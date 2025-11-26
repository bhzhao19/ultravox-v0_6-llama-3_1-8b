import logging
import re
from typing import Any, Dict, Generator, Optional, Set, Tuple, TypeVar, Union

import accelerate
import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.activations
import transformers.modeling_outputs
import transformers.models
from transformers.generation.utils import GenerationMixin
from transformers.models.whisper import modeling_whisper as whisper

# We must use relative import in this directory to allow uploading to HF Hub
# Even "from . import X" pattern doesn't work (undocumented and unclear why)
from .ultravox_config import LossConfig
from .ultravox_config import LossFunction
from .ultravox_config import UltravoxConfig

FROM_PRETRAINED_KWARGS = {}
SHARED_PRETRAINED_KWARGS = [
    "tp_plan",
    "device_map",
    "torch_dtype",
    "attn_implementation",
    "use_flash_attention_2",
]


class UltravoxModel(transformers.LlamaPreTrainedModel, GenerationMixin):
    """
    ============================================================================
    ULTRAVOX MODEL ARCHITECTURE
    ============================================================================
    
    Ultravox is a multi-modal model that combines audio understanding with language
    modeling. It processes audio and text inputs together to enable audio-conditioned
    text generation.
    
    ARCHITECTURE OVERVIEW:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Raw Audio   │ --> │ Audio Tower  │ --> │  Projector  │ --> │   Merged    │
    │ Waveforms   │     │ (Whisper/    │     │ (Linear     │     │ Embeddings  │
    │             │     │  Wav2Vec2)   │     │  Layers)    │     │             │
    └─────────────┘     └──────────────┘     └─────────────┘     └──────┬──────┘
                                                                        │
    ┌─────────────┐     ┌─────────────────┐                             │
    │ Text Tokens │ --> │ Text Embeddings │ ----------------------------┘
    │             │     │ (LLM Embedding) │           |
    └─────────────┘     └─────────────────┘           |
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │ Language Model  │
                                            │   (Llama LLM)   │
                                            └─────────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │  Output Logits  │
                                            │   / Loss        │
                                            └─────────────────┘
    
    KEY COMPONENTS:
    1. Audio Tower: Encodes raw audio waveforms into feature embeddings
       - Supports Whisper or Wav2Vec2 encoders
       - Output: (batch, audio_frames, audio_hidden_dim)
    
    2. Multi-modal Projector: Bridges audio and text embedding spaces
       - Stacks audio frames (reduces sequence length by stack_factor)
       - Projects from audio_dim to text_dim via 2 linear layers
       - Output: (batch, reduced_frames, text_hidden_dim)
    
    3. Language Model: Processes fused audio-text embeddings
       - Uses Llama architecture for causal language modeling
       - Receives merged embeddings where audio replaces <|audio|> tokens
       - Output: Next token predictions and loss
    
    DATA FLOW:
    1. Audio waveforms → Audio Tower → Audio embeddings (audio_dim)
    2. Audio embeddings → Stack frames → Projector → Text-space embeddings (text_dim)
    3. Text tokens → Text embeddings (text_dim)
    4. Merge: Replace <|audio|> token positions with projected audio embeddings
    5. Fused embeddings → LLM → Predictions
    
    SPECIAL TOKEN:
    - <|audio|>: Marks where audio embeddings should be inserted in the text sequence
      The projector output replaces this token's embedding in the sequence.
    
    Parameters:
        config: Model configuration class with all the parameters of the model.
    """

    config_class = UltravoxConfig
    config: UltravoxConfig  # for type hinting
    # Usually we load encoder and LLM weights from a pretrained model separately, so they are allowed to be missing
    _keys_to_ignore_on_load_missing = ["audio_tower.*", "language_model.*"]
    # Since we have kwargs in forward, we need to set this to False, otherwise grad_accum_steps will cause incorrect train loss to be reported
    # see https://github.com/huggingface/transformers/issues/35856 and https://github.com/huggingface/trl/pull/2615/files
    accepts_loss_kwargs = False

    def __init__(self, config: UltravoxConfig):
        """
        Initialize the Ultravox model architecture.
        
        This method constructs the three main components of the model:
        1. Audio Tower (encoder)
        2. Multi-modal Projector
        3. Language Model (LLM)
        
        Each component can be loaded from pretrained weights or initialized from config.
        """
        logging.info(f"__init__:config: {config}")
        super().__init__(config)
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

        self.keep_params: Set[str] = set()
        self.vocab_size = config.vocab_size

        # ========================================================================
        # COMPONENT 1: Audio Tower (Encoder)
        # ========================================================================
        # Encodes raw audio waveforms into feature embeddings.
        # - Input: Raw audio waveforms (mel spectrograms for Whisper)
        # - Output: Audio feature embeddings of shape (batch, frames, audio_hidden_dim)
        # - Supports: Whisper encoder or Wav2Vec2-based models
        # - Can be frozen or fine-tuned with LoRA
        self.audio_tower = self._create_audio_tower(config)
        self.audio_tower_context_length: Optional[int] = None
        self.audio_tower_context_length = self.audio_tower.max_context_length

        # ========================================================================
        # COMPONENT 2: Multi-modal Projector
        # ========================================================================
        # Projects audio embeddings from audio space to text embedding space.
        # - Input: Audio embeddings (batch, frames, audio_hidden_dim)
        # - Process: Stack frames → Linear1 → Activation → Linear2
        # - Output: Text-space embeddings (batch, reduced_frames, text_hidden_dim)
        # - Key: Reduces sequence length by stack_factor while increasing channel dim
        self.multi_modal_projector = self._create_multi_modal_projector(config)
        
        # ========================================================================
        # COMPONENT 3: Language Model (LLM)
        # ========================================================================
        # Processes fused audio-text embeddings for generation.
        # - Input: Merged embeddings where audio replaces <|audio|> tokens
        # - Architecture: Llama-based causal language model
        # - Output: Next token logits and loss
        # - Can be frozen or fine-tuned with LoRA
        self.language_model = self._create_language_model(config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in self.language_model._tied_weights_keys
            ]

        # Determine no_split_modules dynamically to use with FSDP auto_wrap policy.
        # FSDP throws an error if some of the layer types are not found in the model.
        # This would be something like ["LlamaDecoderLayer"] as we don't split audio encoder layers.
        self._no_split_modules = self.language_model._no_split_modules
        self.loss_config = LossConfig()
        self.post_init()

    def _init_weights(self, module):
        if module is self:
            if self.config.text_model_id is not None:
                self.language_model = self._create_language_model(self.config)
            if self.config.audio_model_id is not None:
                self.audio_tower = self._create_audio_tower(self.config)
        elif module in self.language_model.modules():
            pass
        elif module in self.audio_tower.modules():
            pass
        else:
            super()._init_weights(module)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        global FROM_PRETRAINED_KWARGS
        FROM_PRETRAINED_KWARGS = {
            k: v for k, v in kwargs.items() if k in SHARED_PRETRAINED_KWARGS
        }
        model = super().from_pretrained(*args, **kwargs)
        FROM_PRETRAINED_KWARGS = {}
        return model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def set_loss_config(self, loss_config: LossConfig):
        self.loss_config = loss_config

    def _setup_cache(
        self, cache_cls, max_batch_size: int, max_cache_len: Optional[int] = None
    ):
        self.language_model._setup_cache(cache_cls, max_batch_size, max_cache_len)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _get_prediction_mask(
        self, labels: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get boolean masks for positions where we want to compute KL divergence.

        For each label position, we want the position before it since that's where
        the model makes the prediction for that label.

        Additionally, we want to identify the position right before the EOT token
        (the last token with label != -100).

        Args:
            labels: Tensor of shape (B, T) where B is batch size and T is sequence length,
                   with -100 for masked positions and token ids for label positions

        Returns:
            Tuple containing:
            - pred_mask: Boolean tensor of shape (B, T) that's True for positions where we want to compute KL divergence
            - eot_mask: Boolean tensor of shape (B, T) that's True only for the last prediction position in each sequence
        """
        if labels is None:
            raise ValueError("labels must be provided")

        # Shift the label mask right by 1 along the sequence dimension
        # This gives us positions where we make predictions for the next token
        label_mask = labels != -100
        pred_mask = torch.zeros_like(label_mask)
        pred_mask[:, :-1] = label_mask[
            :, 1:
        ]  # shift right by 1 along sequence dimension

        # Create EOT mask - identify only the last prediction position in each sequence
        eot_mask = torch.zeros_like(pred_mask)
        batch_size = labels.shape[0]

        for i in range(batch_size):
            # Find positions where we make predictions
            pred_positions = torch.where(pred_mask[i])[0]
            if len(pred_positions) > 0:
                # Only mark the last prediction position
                eot_mask[i, pred_positions[-1]] = True

        return pred_mask, eot_mask

    def _compute_kl_loss(
        self,
        lm_output: transformers.modeling_outputs.CausalLMOutputWithPast,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        alt_input_ids: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        alt_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        ========================================================================
        KL DIVERGENCE LOSS: Knowledge Distillation
        ========================================================================
        
        This implements knowledge distillation where:
        - Student: Audio-text model (current model with audio inputs)
        - Teacher: Text-only model (same model but without audio)
        
        GOAL: Make the audio-text model produce similar predictions to the
              text-only model, ensuring audio enhances rather than changes outputs.
        
        PROCESS:
        1. Run teacher (text-only) forward pass (no gradients)
        2. Compare student and teacher logit distributions
        3. Compute KL divergence loss
        4. Optionally weight EOT (end-of-turn) token positions more heavily
        
        WHY KL DIVERGENCE?
        - Aligns probability distributions, not just predictions
        - Captures uncertainty and confidence levels
        - Temperature scaling makes distributions softer for better learning
        
        TEMPERATURE SCALING:
        - Divides logits by temperature before softmax
        - Higher temperature = softer distribution (more uncertainty)
        - Helps with knowledge transfer
        """
        # ========================================================================
        # STEP 1: Compute Teacher (Text-Only) Model Outputs
        # ========================================================================
        # The teacher model is the same architecture but processes text-only inputs
        # We disable gradients because we're using it as a fixed reference
        with torch.no_grad():
            # Embed text tokens (no audio here - this is the text-only version)
            alt_inputs_embeds = self.get_input_embeddings().forward(alt_input_ids)
            # Forward through language model (text-only, no audio merged)
            alt_lm_output = self.language_model.forward(
                inputs_embeds=alt_inputs_embeds,
                labels=alt_labels,
                attention_mask=alt_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # ========================================================================
        # STEP 2: Get Prediction Masks
        # ========================================================================
        # We need masks to identify:
        # - pred_mask: All positions where we make predictions (before each label)
        # - eot_mask: Last prediction position in each sequence (end-of-turn)
        # The EOT position is important because it's where the model decides to stop
        pred_mask, eot_mask = self._get_prediction_mask(labels)
        alt_pred_mask, alt_eot_mask = self._get_prediction_mask(alt_labels)

        # ========================================================================
        # STEP 3: Compute KL Divergence for Regular Tokens
        # ========================================================================
        # KL divergence measures how different two probability distributions are
        # We want to minimize this difference between student and teacher
        # 
        # Formula: KL(P_student || P_teacher) = sum(P_student * log(P_student / P_teacher))
        # 
        # Temperature scaling:
        # - Divides logits by temperature before softmax
        # - Makes distributions "softer" (more uniform)
        # - Helps with knowledge transfer
        kl_loss = F.kl_div(
            # Student distribution (audio-text model)
            F.log_softmax(
                lm_output.logits[pred_mask] / self.loss_config.kl_temperature,
                dim=-1,
            ),
            # Teacher distribution (text-only model)
            F.softmax(
                alt_lm_output.logits[alt_pred_mask] / self.loss_config.kl_temperature,
                dim=-1,
            ),
            reduction="batchmean",  # Average over batch
        )

        # ========================================================================
        # STEP 4: Compute Additional KL Loss for EOT Tokens (Optional)
        # ========================================================================
        # EOT (end-of-turn) tokens are critical because they indicate when to stop
        # We can weight these positions more heavily to ensure proper stopping behavior
        if self.loss_config.eot_loss_weight > 0:
            eot_loss = F.kl_div(
                F.log_softmax(
                    lm_output.logits[eot_mask] / self.loss_config.kl_temperature,
                    dim=-1,
                ),
                F.softmax(
                    alt_lm_output.logits[alt_eot_mask]
                    / self.loss_config.kl_temperature,
                    dim=-1,
                ),
                reduction="batchmean",
            )
            # Add weighted EOT loss to main KL loss
            kl_loss += self.loss_config.eot_loss_weight * eot_loss

        return kl_loss

    def _audio_iter(
        self, audio_batch_size: torch.Tensor
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Helper iterator for mapping audio samples to text samples in batches.
        
        This handles the case where multiple audio samples can map to one text sample.
        For example, a conversation might have multiple audio turns but one text sequence.
        
        Args:
            audio_batch_size: Tensor of shape (B,) indicating how many audio samples
                            each text sample has. Example: [2, 1, 3] means:
                            - Text sample 0 has 2 audio samples
                            - Text sample 1 has 1 audio sample
                            - Text sample 2 has 3 audio samples
        
        Yields:
            (batch_idx, audio_idx): Tuple mapping text batch index to audio index
            Example: For [2, 1, 3], yields:
                    (0, 0), (0, 1), (1, 2), (2, 3), (2, 4), (2, 5)
        """
        audio_index = 0
        for i_b, batch_count in enumerate(audio_batch_size):
            for _ in range(batch_count):
                yield i_b, audio_index
                audio_index += 1

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        # the alt_* fields are needed for KL divergence loss
        alt_input_ids: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        alt_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        """
        ========================================================================
        FORWARD PASS: How the Model Processes Audio and Text
        ========================================================================
        
        This is the core forward pass that processes audio and text inputs together.
        
        PROCESSING PIPELINE:
        
        STEP 1: Embed Text Tokens
        ─────────────────────────
        - Input: input_ids (batch, seq_len) - tokenized text
        - Process: Lookup in LLM's embedding table
        - Output: inputs_embeds (batch, seq_len, text_hidden_dim)
        - Note: If inputs_embeds already provided, skip this step
        
        STEP 2: Process Audio (if provided)
        ───────────────────────────────────
        - Input: audio_values (batch, audio_length) - raw audio waveforms
        - Process through Audio Tower:
          * Converts raw audio to mel spectrograms (for Whisper)
          * Encodes through transformer layers
          * Output: (batch, audio_frames, audio_hidden_dim)
        - Process through Projector:
          * Stacks frames (reduces length by stack_factor)
          * Projects to text embedding space
          * Output: (batch, reduced_frames, text_hidden_dim)
        
        STEP 3: Merge Audio and Text Embeddings
        ────────────────────────────────────────
        - Find <|audio|> token positions in text sequence
        - Replace those token embeddings with projected audio embeddings
        - Result: Unified sequence with audio "injected" at specific positions
        - Example: [text_tokens...] <|audio|> [more_text...]
                  becomes: [text_embeds...] [audio_embeds...] [text_embeds...]
        
        STEP 4: Process Through Language Model
        ───────────────────────────────────────
        - Input: Merged embeddings (batch, total_seq_len, text_hidden_dim)
        - Process: Standard causal language model forward pass
        - Output: Logits for next token prediction + loss (if labels provided)
        
        STEP 5: Compute Loss (if training)
        ──────────────────────────────────
        - CrossEntropy: Standard next-token prediction loss
        - KL Divergence: Distillation loss between audio-text model and text-only model
        
        KEY PARAMETERS:
        - audio_token_start_idx: Position in text sequence where audio embeddings start
        - audio_token_len: How many tokens the audio embeddings span
        - audio_lens: Actual audio lengths (for masking padding)
        - audio_batch_size: How many audio samples per text sample (for batching)
        
        Args:
            input_ids: Tokenized text input (batch, seq_len)
            audio_values: Raw audio waveforms (list of tensors, variable length)
            inputs_embeds: Pre-computed text embeddings (optional, batch, seq_len, dim)
            labels: Ground truth tokens for loss computation (batch, seq_len)
            attention_mask: Mask for valid positions (batch, seq_len)
            audio_token_start_idx: Start positions for audio in text sequence
            audio_lens: Actual lengths of audio samples (for padding)
            audio_token_len: Number of tokens each audio sample produces
            audio_batch_size: Number of audio samples per text sample
            past_key_values: Cached key-value pairs for generation
            alt_*: Alternative text-only inputs for KL divergence loss
            **kwargs: Additional arguments passed to language model
        """
        # ========================================================================
        # STEP 1: Embed Text Tokens
        # ========================================================================
        # Convert token IDs to embeddings using the LLM's embedding layer
        # Shape: (batch, seq_len) -> (batch, seq_len, text_hidden_dim)
        if inputs_embeds is None:
            # B x T  ->  B x T x D
            inputs_embeds = self.get_input_embeddings().forward(input_ids)

        # ========================================================================
        # STEP 2-3: Process and Merge Audio Embeddings (if audio provided)
        # ========================================================================
        if audio_values is not None and len(audio_values) > 0:
            # Validate that all required audio metadata is provided
            assert (
                audio_token_start_idx is not None
                and audio_token_len is not None
                and audio_lens is not None
                and audio_batch_size is not None
            ), "audio_token_start_idx/audio_token_len/audio_lens must be provided if audio_values are provided."
            assert (
                len(audio_token_start_idx)
                == len(audio_token_len)
                == len(audio_lens)
                == len(audio_values)
            ), "audio_token_start_idx/audio_token_len/audio_lens/audio_values must have the same batch size."
            assert len(audio_batch_size) == len(
                inputs_embeds
            ), "audio_batch_size and inputs_embeds must have the same batch size."

            # STEP 2a: Audio Tower - Encode raw audio to feature embeddings
            # Input: audio_values (list of variable-length audio tensors)
            # Output: (batch, audio_frames, audio_hidden_dim)
            # The audio tower handles:
            # - Converting raw audio to mel spectrograms (Whisper) or features (Wav2Vec2)
            # - Processing through convolutional layers
            # - Encoding through transformer encoder layers
            # - Applying attention masks based on actual audio lengths
            # B x A/3200 x (D=max-audio-length-in-batch)
            audio_tower_output = self.audio_tower.forward(
                audio_values.to(self.audio_tower.dtype),
                audio_len=audio_lens,
            ).last_hidden_state
            audio_tower_output = audio_tower_output.to(inputs_embeds.dtype)
            
            # STEP 2b: Multi-modal Projector - Project audio embeddings to text space
            # Input: (batch, audio_frames, audio_hidden_dim)
            # Process: Stack frames → Linear1 → Activation → Linear2
            # Output: (batch, reduced_frames, text_hidden_dim)
            # This aligns audio embeddings with text embedding dimensions
            audio_embeds = self.multi_modal_projector.forward(audio_tower_output)

            # STEP 3: Merge audio embeddings into text embeddings at <|audio|> positions
            # For each text sample, find where <|audio|> tokens are and replace them
            # with the corresponding projected audio embeddings
            # 
            # Example merging:
            #   Text sequence: [token1, token2, <|audio|>, token3, token4]
            #   Text embeddings: [emb1, emb2, audio_token_emb, emb3, emb4]
            #   After merge:    [emb1, emb2, audio_emb1, audio_emb2, ..., emb3, emb4]
            #
            # The loop handles batching where multiple audio samples map to one text sample
            for i_b, i_a in self._audio_iter(audio_batch_size):
                start_idx = audio_token_start_idx[i_a]  # Where to insert in text sequence
                token_len = audio_token_len[i_a]        # How many tokens to insert
                item_embedding = audio_embeds[i_a][:token_len]  # Get audio embeddings
                # Replace text embeddings at <|audio|> positions with audio embeddings
                inputs_embeds[i_b][start_idx : start_idx + token_len] = item_embedding

        # ========================================================================
        # STEP 4: Process Through Language Model
        # ========================================================================
        # Now we have a unified sequence with audio embeddings merged into text
        # Process through the LLM as a standard causal language model
        # The LLM doesn't know which tokens are audio vs text - it just sees embeddings
        lm_output = self.language_model.forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        # ========================================================================
        # STEP 5: Compute Loss (if training)
        # ========================================================================
        # The model supports two loss functions:
        # 1. CrossEntropy: Standard next-token prediction loss (default)
        # 2. KL Divergence: Distillation loss that aligns audio-text model with text-only model
        if self.loss_config.loss_function == LossFunction.CrossEntropy:
            # Standard cross-entropy loss already computed by language_model.forward()
            # No additional processing needed
            pass
        elif self.loss_config.loss_function == LossFunction.KL_Divergence:
            # KL divergence loss for knowledge distillation
            # Compares predictions from:
            # - Student: Audio-text model (current forward pass)
            # - Teacher: Text-only model (alt_input_ids without audio)
            # This helps the model learn to generate similar outputs with/without audio
            lm_output.loss = self._compute_kl_loss(
                lm_output=lm_output,
                labels=labels,
                past_key_values=past_key_values,
                alt_input_ids=alt_input_ids,
                alt_attention_mask=alt_attention_mask,
                alt_labels=alt_labels,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported loss function: {self.loss_config.loss_function}"
            )
        return lm_output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_input = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        # include audio information in model_input only when it is needed during prefilling
        # audio_token_start_idx should always be relative to the current cache position
        prefill_start_idx: int | torch.Tensor = (
            0 if cache_position is None else cache_position[0]
        )
        if (
            audio_values is not None
            and audio_token_start_idx is not None
            and prefill_start_idx <= torch.max(audio_token_start_idx)
        ):
            model_input["audio_values"] = audio_values
            model_input["audio_token_start_idx"] = (
                audio_token_start_idx - prefill_start_idx
            )
            model_input["audio_token_len"] = audio_token_len
            model_input["audio_batch_size"] = audio_batch_size
            model_input["audio_lens"] = audio_lens

        return model_input

    @classmethod
    def _create_multi_modal_projector(
        cls, config: UltravoxConfig
    ) -> "UltravoxProjector":
        projector = UltravoxProjector(config)
        dtype = config.torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        projector.to(dtype)
        return projector

    @classmethod
    def _create_audio_tower(
        cls, config: UltravoxConfig
    ) -> Union[transformers.Wav2Vec2Model, "ModifiedWhisperEncoder"]:
        # We probably don't want to pass tp_plan or device_map to the audio tower
        # But potentially other kwargs can be passed in. TODO
        kwargs = {"torch_dtype": config.torch_dtype}
        if (
            transformers.modeling_utils._init_weights
            and config.audio_model_id is not None
        ):
            if "whisper" in config.audio_model_id.lower():
                audio_tower = ModifiedWhisperEncoder.from_pretrained(
                    config.audio_model_id, **kwargs
                )
                audio_tower.init_latency_mask(
                    config.audio_latency_block_size, dtype=config.torch_dtype
                )
            else:
                assert config.audio_latency_block_size in (
                    None,
                    0,
                ), "only whisper audio tower supports audio latency masking, got non-zero value for 'audio_latency_block_size'"
                audio_tower = transformers.AutoModel.from_pretrained(
                    config.audio_model_id, **kwargs
                )
        else:
            with accelerate.init_empty_weights():
                if "whisper" in config.audio_config._name_or_path.lower():
                    audio_tower = ModifiedWhisperEncoder(config.audio_config)
                    audio_tower.init_latency_mask(
                        config.audio_latency_block_size,
                        dtype=config.torch_dtype,
                    )
                else:
                    assert config.audio_latency_block_size in (
                        None,
                        0,
                    ), "only whisper audio tower supports audio latency masking, got non-zero value for 'audio_latency_block_size'"
                    # we only ever use from_config if the weights are retrained, hence initializing is not
                    # required. This makes the model quite creation faster since init on CPU is quite slow.
                    audio_tower = transformers.AutoModel.from_config(
                        config.audio_config, **kwargs
                    )

        if isinstance(
            audio_tower,
            (transformers.Wav2Vec2BertModel, transformers.WhisperModel),
        ):
            # For these models we only need the encoder part
            # Wav2Vec2BertModel -> Wav2Vec2BertEncoder
            # WhisperModel -> WhisperEncoder
            audio_tower = audio_tower.encoder

        audio_tower = apply_lora(audio_tower, config.audio_model_lora_config)
        return audio_tower

    @classmethod
    def _create_language_model(
        cls, config: UltravoxConfig
    ) -> transformers.LlamaForCausalLM:
        if (
            transformers.modeling_utils._init_weights
            and config.text_model_id is not None
        ):
            language_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.text_model_id,
                **{
                    "attn_implementation": config.text_config._attn_implementation,
                    "torch_dtype": config.torch_dtype,
                    **FROM_PRETRAINED_KWARGS,
                },
            )
            if not language_model._no_split_modules:
                if hasattr(language_model, "model") and hasattr(
                    language_model.model, "_no_split_modules"
                ):
                    language_model._no_split_modules = (
                        language_model.model._no_split_modules
                    )
                    logging.info(
                        f"_create_language_model: setting language_model._no_split_modules ([]) from language_model.model._no_split_modules: {language_model._no_split_modules}"
                    )
                else:
                    raise ValueError(
                        f"_no_split_modules is not set for {config.text_model_id}"
                    )
            logging.info(
                f"_create_language_model: language_model device type: {language_model.device.type}"
            )
        else:
            with accelerate.init_empty_weights():
                # we only ever use from_config if the weights are retrained, hence initializing is not
                # required. This makes the model quite creation faster since init on CPU is quite slow.
                language_model = transformers.AutoModelForCausalLM.from_config(
                    config.text_config,
                    attn_implementation=config.text_config._attn_implementation,
                    torch_dtype=config.torch_dtype,
                )

        language_model = apply_lora(language_model, config.text_model_lora_config)
        return language_model

    def merge_and_unload(self):
        if isinstance(self.language_model, peft.PeftModel):
            self.language_model = self.language_model.merge_and_unload()
            # no need to download base language model weights anymore, so we can remove the id
            self.config.text_model_id = None
            self.keep_params.update(
                set(
                    [
                        f"language_model.{name}"
                        for name, _ in self.language_model.named_parameters()
                    ]
                )
            )

        if isinstance(self.audio_tower, peft.PeftModel):
            self.audio_tower = self.audio_tower.merge_and_unload()
            # no need to download base audio model weights anymore, so we can remove the id
            self.config.audio_model_id = None
            self.keep_params.update(
                set(
                    [
                        f"audio_tower.{name}"
                        for name, _ in self.audio_tower.named_parameters()
                    ]
                )
            )

        for param in ["text_model_lora_config", "audio_model_lora_config"]:
            if hasattr(self.config, param):
                delattr(self.config, param)

    def push_to_hub(self, *args, **kwargs):
        self.merge_and_unload()
        return super().push_to_hub(*args, **kwargs)

    def diff_state_dict(
        self, state_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if state_dict is None:
            state_dict = super().state_dict()

        trainable_params = {k for k, v in self.named_parameters() if v.requires_grad}
        # normalize the keys to match the original model
        # Example: audio_tower.base_model.model.layers.0._fsdp_wrapped_module.self_attn.k_proj.lora_B.default.weight
        trainable_params = {
            k.replace("_fsdp_wrapped_module.", "") for k in trainable_params
        }

        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.keep_params or k in trainable_params
        }

        return state_dict

    def save_pretrained(
        self, *args, state_dict: Optional[Dict[str, Any]] = None, **kwargs
    ):
        state_dict = self.diff_state_dict(state_dict)

        super().save_pretrained(*args, state_dict=state_dict, **kwargs)

    def _pre_load_state_dict_hook(self, state_dict: Dict[str, Any], *args, **kwargs):
        self.keep_params.update(set(state_dict.keys()))

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model (reuses Peft model's method)
        """
        count_params = peft.peft_model.PeftModel.get_nb_trainable_parameters

        trainable_params, all_param = count_params(self)

        logging.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
            f" || trainable%: {100 * trainable_params / all_param:.1f}%"
        )

        lm_trainable_params, lm_all_params = count_params(self.language_model)
        audio_trainable_params, audio_all_params = count_params(self.audio_tower)

        projector_trainable_params = (
            trainable_params - lm_trainable_params - audio_trainable_params
        )
        projector_all_params = all_param - lm_all_params - audio_all_params

        logging.info(
            f"Trainable%:   "
            f" LLM: {100 * lm_trainable_params / lm_all_params:.1f}%"
            f" || Audio Encoder: {100 * audio_trainable_params / audio_all_params:.1f}%"
            f" || Projector: {100 * projector_trainable_params / projector_all_params:.1f}%"
        )


def get_checkpoint_files(
    model_id: str,
) -> tuple[list[str], dict | None, list[str]]:
    resolved_archive_file = transformers.utils.cached_file(
        model_id,
        transformers.utils.SAFE_WEIGHTS_NAME,
        _raise_exceptions_for_missing_entries=False,
    )

    if resolved_archive_file is not None:
        # not sharded
        sharded_metadata = None
        state_dict = transformers.modeling_utils.load_state_dict(resolved_archive_file)
        loaded_state_dict_keys = list(state_dict.keys())
    else:
        # sharded
        resolved_archive_file = transformers.utils.cached_file(
            model_id, transformers.utils.SAFE_WEIGHTS_INDEX_NAME
        )
        resolved_archive_file, sharded_metadata = (
            transformers.modeling_utils.get_checkpoint_shard_files(
                model_id,
                resolved_archive_file,
            )
        )
        loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]

    if isinstance(resolved_archive_file, str):
        resolved_archive_file = [resolved_archive_file]

    return resolved_archive_file, sharded_metadata, loaded_state_dict_keys


# TODO: refactor common parts to a shared module
def is_cache_empty(
    past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]],
) -> bool:
    """
    Check if the cache is empty.
    """
    if past_key_values is None:
        return True
    if isinstance(past_key_values, tuple):
        return all(len(c) == 0 for c in past_key_values)
    return past_key_values.get_seq_length() == 0


T = TypeVar("T", bound=torch.nn.Module)


def apply_lora(model: T, lora_config: dict) -> T:
    """
    Applies LoRA finetuning to the model. If the `r` parameter is set to 0, the model is frozen instead.
    """
    unfreeze_layers = lora_config.pop("unfreeze_layers", None)
    lora_config = peft.LoraConfig(**lora_config or {})

    if lora_config.r == 0:
        # freeze the model entirely, except for the specified layers
        for name, param in model.named_parameters():
            if not unfreeze_layers or not any(
                re.match(layer, name) for layer in unfreeze_layers
            ):
                param.requires_grad = False
            else:
                logging.info(f"Unfreezing layer: {name} with #{param.numel()} params")
    else:
        model = peft.get_peft_model(model, lora_config)

    return model


class StackAudioFrames(nn.Module):
    """
    ============================================================================
    STACK AUDIO FRAMES: Reducing Sequence Length
    ============================================================================
    
    This module reduces the sequence length of audio embeddings by stacking
    consecutive frames together. This is crucial for aligning audio frame rate
    with text token rate.
    
    WHY STACKING?
    - Audio encoders produce high frame rates (e.g., 50 frames/second)
    - Text models work with lower token rates (e.g., 2-5 tokens/second)
    - Stacking balances these rates by combining multiple frames into one token
    
    TRANSFORMATION:
    Input:  (batch, frames, channels)
            Example: (2, 80, 512)  [80 frames, 512-dim embeddings]
    
    Process:
    1. Pad to multiple of stack_factor: (2, 80, 512) -> (2, 80, 512) [already multiple of 8]
    2. Reshape: (2, 80, 512) -> (2, 10, 4096)  [80/8=10, 512*8=4096]
    
    Output: (batch, frames/stack_factor, channels*stack_factor)
            Example: (2, 10, 4096)  [10 tokens, 4096-dim stacked embeddings]
    
    Example with stack_factor=8:
    - Before: 80 frames × 512 dims = 80 tokens
    - After:  10 tokens × 4096 dims = 10 tokens
    - Sequence length reduced by 8x, channel dimension increased by 8x
    """
    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        """
        Stack audio frames to reduce sequence length.
        
        Process:
        1. Pad sequence to be divisible by stack_factor
        2. Reshape: (B, T, C) -> (B, T//S, C*S)
           where S = stack_factor
        
        Example:
          Input:  (2, 77, 512) with stack_factor=8
          Pad:    (2, 80, 512)  [pad to next multiple of 8]
          Reshape: (2, 10, 4096) [80/8=10, 512*8=4096]
        """
        B, T, C = audio_embeds.shape
        # Calculate padding needed to make T divisible by stack_factor
        T_pad = (T + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        # Pad along sequence dimension (last dimension is channels, don't pad that)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T))
        B, T, C = audio_embeds.shape
        # Reshape: combine stack_factor frames into one, increase channels
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
        return audio_embeds


class RMSNorm(transformers.models.llama.modeling_llama.LlamaRMSNorm):
    def __init__(self, hidden_size: int, init: float = 1, eps: float = 1e-6):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.weight.data.fill_(init)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class UltravoxProjector(nn.Module):
    """
    ============================================================================
    MULTI-MODAL PROJECTOR: Bridging Audio and Text Embedding Spaces
    ============================================================================
    
    This is a critical architectural component that projects audio embeddings
    from the audio encoder's space into the text language model's embedding space.
    
    ARCHITECTURE:
    ┌─────────────────┐
    │ Audio Embeddings│ (batch, frames, audio_hidden_dim)
    │ from Audio Tower │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Stack Frames     │ Reduces sequence length by stack_factor
    │ (stack_factor)   │ Increases channel dim by stack_factor
    └────────┬────────┘ (batch, frames/stack_factor, audio_hidden_dim*stack_factor)
             │
             ▼
    ┌─────────────────┐
    │ Pre-Norm (RMS)   │ Normalize stacked features
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear 1        │ Project to intermediate dimension
    │ (dim_in → H)     │ (batch, frames/stack_factor, hidden_dim)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Activation       │ SwiGLU or other activation
    │ (SwiGLU halves)  │ (batch, frames/stack_factor, hidden_dim/2)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Mid-Norm (RMS)   │ Normalize after activation (v0.5.0+)
    │ or Post-Norm     │ or after Linear2 (v0.4.1-)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear 2        │ Project to text embedding dimension
    │ (H/2 → text_dim)│ (batch, frames/stack_factor, text_hidden_dim)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Post-Norm (RMS) │ Final normalization
    └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Text-Space       │ Ready to merge with text embeddings
    │ Embeddings       │
    └─────────────────┘
    
    KEY DESIGN DECISIONS:
    1. Stacking: Reduces sequence length to match text token density
       - Audio: High frame rate (e.g., 50 frames/sec)
       - Text: Lower token rate (e.g., 2-5 tokens/sec)
       - Stacking balances the rates
    
    2. Two-stage projection: 
       - Linear1: Maps to intermediate dimension (config.hidden_size)
       - Linear2: Maps to text dimension (text_config.hidden_size)
       - Allows flexible dimension matching
    
    3. SwiGLU activation: Gated activation that halves dimension
       - More expressive than ReLU
       - Common in modern LLMs (Llama uses it)
    
    4. Layer norm placement: Varies by version
       - v0.5.0+: After Linear1 (mid-norm)
       - v0.4.1-: After Linear2 (post-norm)
    """
    def __init__(self, config: UltravoxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        
        # Stack audio frames to reduce sequence length
        # Example: stack_factor=8 means 8 frames become 1 token
        # Input: (batch, 80, 512) -> Output: (batch, 10, 4096)
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        
        # After stacking, input dimension is audio_hidden_size * stack_factor
        # Example: 512 * 8 = 4096
        dim_in = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim_in, init=config.norm_init)
        
        # First linear projection: stacked_audio_dim -> hidden_dim
        # Example: 4096 -> 2048
        self.linear_1 = nn.Linear(dim_in, self.hidden_dim, bias=False)
        dim_mid = self.hidden_dim
        
        # Activation function (SwiGLU halves the dimension)
        self.act = transformers.activations.get_activation(config.projector_act)
        dim_mid = dim_mid // 2 if config.projector_act == "swiglu" else dim_mid
        # Example: 2048 -> 1024 (if SwiGLU)
        
        # Second linear projection: hidden_dim/2 -> text_hidden_dim
        # Example: 1024 -> 4096 (if text model is 4096-dim)
        dim_out = config.text_config.hidden_size
        self.linear_2 = nn.Linear(dim_mid, dim_out, bias=False)

        # Layer norm placement varies by version:
        # Ultravox v0.4.1 and below uses layer_norm after the second linear layer,
        # while v0.5.0 and above uses layer_norm after the first linear layer.
        if config.projector_ln_mid:
            # v0.5.0+: Normalize after Linear1 (before Linear2)
            self.ln_mid: nn.Module = RMSNorm(dim_mid, init=config.norm_init)
            self.ln_post: nn.Module = nn.Identity()
        else:
            # v0.4.1-: Normalize after Linear2 (final output)
            self.ln_mid = nn.Identity()
            self.ln_post = RMSNorm(dim_out, init=config.norm_init)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projector.
        
        TRANSFORMATION PIPELINE:
        Input:  (batch, audio_frames, audio_hidden_dim)
                Example: (2, 80, 512)
        
        Step 1 - Stack: (batch, frames/stack_factor, audio_dim*stack_factor)
                 Example: (2, 10, 4096)  [80 frames -> 10 tokens, 512 -> 4096]
        
        Step 2 - Linear1: (batch, frames/stack_factor, hidden_dim)
                 Example: (2, 10, 2048)
        
        Step 3 - Activation: (batch, frames/stack_factor, hidden_dim/2)
                 Example: (2, 10, 1024)  [SwiGLU halves dimension]
        
        Step 4 - Linear2: (batch, frames/stack_factor, text_hidden_dim)
                 Example: (2, 10, 4096)
        
        Output: (batch, reduced_frames, text_hidden_dim)
                Ready to merge with text embeddings!

        Input shape:
            audio_features: B, F, C
                B: batch size
                F: number of frames from audio tower
                C: audio_hidden_dim (e.g., 512 for Whisper)
        
        Output shape:
            hidden_states: B, T, D
                B: batch size
                T: number of output embeddings = ceil(F / stack_factor)
                D: text_hidden_dim (e.g., 4096 for Llama)
        
        Where:
            S: stack_factor (e.g., 8)
            H: hidden_dim (intermediate dimension, e.g., 2048)
        """
        # STEP 1: Stack frames to reduce sequence length
        # Input: (B, F, C) -> Output: (B, T, C*S) where T = ceil(F/S)
        # Example: (2, 80, 512) -> (2, 10, 4096) with stack_factor=8
        # Padding is applied if F is not divisible by stack_factor
        audio_features = self._pad_and_stack(audio_features)
        
        # STEP 2: Pre-normalization
        # Normalize the stacked features before first linear layer
        audio_features = self.ln_pre(audio_features)
        
        # STEP 3: First linear projection
        # (B, T, C*S) -> (B, T, H)
        # Example: (2, 10, 4096) -> (2, 10, 2048)
        hidden_states = self.linear_1(audio_features)
        
        # STEP 4: Activation (SwiGLU halves dimension)
        # (B, T, H) -> (B, T, H/2)
        # Example: (2, 10, 2048) -> (2, 10, 1024)
        hidden_states = self.act(hidden_states)
        
        # STEP 5: Mid normalization (v0.5.0+) or skip (v0.4.1-)
        hidden_states = self.ln_mid(hidden_states)
        
        # STEP 6: Second linear projection to text dimension
        # (B, T, H/2) -> (B, T, D)
        # Example: (2, 10, 1024) -> (2, 10, 4096)
        hidden_states = self.linear_2(hidden_states)
        
        # STEP 7: Post normalization (v0.4.1-) or skip (v0.5.0+)
        hidden_states = self.ln_post(hidden_states)
        
        return hidden_states


class ModifiedWhisperEncoder(
    whisper.WhisperEncoder, transformers.modeling_utils.ModuleUtilsMixin
):
    """
    ============================================================================
    AUDIO TOWER: Whisper Encoder for Audio Processing
    ============================================================================
    
    This is the audio encoding component that processes raw audio waveforms
    into feature embeddings. It's based on OpenAI's Whisper encoder architecture.
    
    ARCHITECTURE:
    ┌─────────────────┐
    │ Raw Audio        │ Audio waveforms or mel spectrograms
    │ (mel features)   │ Shape: (batch, n_mels, seq_len)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Conv1 + GELU     │ First convolutional layer
    │ (stride reduction)│ Reduces sequence length
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Conv2 + GELU     │ Second convolutional layer
    │ (stride reduction)│ Further reduces sequence length
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Positional      │ Add positional embeddings
    │ Embeddings      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Transformer     │ Stack of encoder layers
    │ Encoder Layers  │ Self-attention + FFN
    │ (6 layers)      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Layer Norm      │ Final normalization
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Audio Embeddings│ Output: (batch, frames, 512)
    └─────────────────┘
    
    KEY MODIFICATIONS from standard Whisper:
    1. base_model_prefix: Updated to allow direct `.from_pretrained()` on encoder
    2. Flexible length: Allows audio shorter than 30 seconds (original requires exactly 30s)
    3. Latency masking: Optional streaming mask for real-time applications
    
    PROCESSING DETAILS:
    - Input: Mel spectrograms (80 mel bins, variable length)
    - Conv layers: Reduce sequence length by stride factors
    - Encoder layers: 6 transformer layers with self-attention
    - Output: Feature embeddings of dimension 512
    
    This implementation is a slightly modified version of HF Transformers' Whisper Encoder:
    1. base_model_prefix updated to allow for doing `.from_pretrained` directly on the encoder
    2. allow less than 30 second of audio padding to be passed in:
        - relaxed ValueError check for `input_features` length to be less than or equal to `expected_seq_length` instead of strictly equal
        - embed_pos is now sliced to match the length of `inputs_embeds`

    Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    """

    base_model_prefix = "model.encoder"
    _no_split_modules = ["WhisperEncoderLayer"]
    _keys_to_ignore_on_load_unexpected = ["model.decoder.*"]

    def __init__(self, config: transformers.WhisperConfig):
        super().__init__(config)
        self.config.is_decoder = False

    @property
    def max_context_length(self):
        return (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )

    def init_latency_mask(
        self, audio_latency_block_size: int | None, dtype: torch.dtype
    ):
        if audio_latency_block_size is None:
            self.audio_streaming_mask = None
            return

        # Use max_context_length directly in the calculation
        max_seqlen = self.max_context_length
        assert (
            max_seqlen > 0
        ), f"maximum sequence length must be positive, got {max_seqlen}"
        assert (
            max_seqlen % audio_latency_block_size == 0
        ), f"audio_latency_block_size {audio_latency_block_size} must divide {max_seqlen} evenly."
        # Given the block size, we calculate number of blocks.
        audio_latency_nblocks = max_seqlen // audio_latency_block_size
        audio_streaming_mask = (
            torch.tril(
                torch.ones(audio_latency_nblocks, audio_latency_nblocks),
                diagonal=0,
            )
            .repeat_interleave(audio_latency_block_size, dim=0)
            .repeat_interleave(audio_latency_block_size, dim=1)
        )
        audio_streaming_mask = (1.0 - audio_streaming_mask) * torch.finfo(dtype).min
        audio_streaming_mask = audio_streaming_mask[None, None, :, :]
        self.register_buffer(
            "audio_streaming_mask", audio_streaming_mask, persistent=False
        )

    def forward(
        self,
        input_features,
        audio_len=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = self.max_context_length
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length} or less, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[: inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Create attention mask based on audio lengths to mask out padding tokens
        # For each sample in batch:
        # - Convert raw audio length to feature length after convolutions
        # - Create boolean mask that is True for valid positions and False for padding
        # - Convert to extended attention mask format expected by transformer layers
        #   (1.0 for positions to attend to, large negative for positions to ignore)
        # This masking ensures consistent behavior between training and inference
        # by preventing the model from attending to padding tokens in both cases
        attention_mask = None
        if audio_len is not None:
            audio_feature_len = self._get_feat_extract_output_lengths(audio_len)
            max_seq_len = hidden_states.shape[1]
            attention_mask = torch.arange(max_seq_len, device=hidden_states.device)[
                None, :
            ].lt(audio_feature_len.view(-1, 1))
            attention_mask = self.get_extended_attention_mask(
                attention_mask,
                None,
                dtype=hidden_states.dtype,
            )

        if self.audio_streaming_mask is not None:
            seqlen = hidden_states.size(-2)
            if attention_mask is not None:
                attention_mask = torch.minimum(
                    self.audio_streaming_mask[:, :, :seqlen, :seqlen], attention_mask
                )  # merge
            else:
                attention_mask = self.audio_streaming_mask[:, :, :seqlen, :seqlen]
            attention_mask = attention_mask.to(hidden_states.dtype)

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


# ============================================================================
# HUGGING FACE INTEGRATION: Model Registration
# ============================================================================
# These registrations allow the Ultravox model to be used with Hugging Face's
# AutoModel and AutoConfig classes, making it easy to load from the Hub.

# Register for automatic class discovery (used by Hugging Face Hub)
# This enables the model to be automatically discovered when uploaded to HF Hub
UltravoxConfig.register_for_auto_class()
UltravoxModel.register_for_auto_class()

# Register with AutoConfig: enables AutoConfig.from_pretrained("model-name")
# When config.json has "model_type": "ultravox", it will use UltravoxConfig
transformers.AutoConfig.register("ultravox", UltravoxConfig)

# Register with AutoModel: enables AutoModel.from_pretrained("model-name")
# When loading a model with UltravoxConfig, it automatically uses UltravoxModel
transformers.AutoModel.register(UltravoxConfig, UltravoxModel)

# Register custom activation function: makes SwiGLU available to transformers
# This allows the projector to use "swiglu" as an activation function name in configs
transformers.activations.ACT2FN["swiglu"] = SwiGLU
