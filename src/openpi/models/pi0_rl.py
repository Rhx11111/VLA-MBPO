import logging
from typing import Literal

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from einops import rearrange, reduce

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.training.utils import concat_observations


logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0RL(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0RLConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.gamma = config.gamma
        self.noise_method = config.noise_method
        self.noise_level = config.noise_level
        self.num_steps = config.num_steps
        
        # Initialize main model components
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Value head (after VLM): support an ensemble of ValueHeads.
        # Read ensemble size from config if present, otherwise default to 1.
        self.value_ensemble_size = getattr(config, "value_ensemble_size", 1)
        self.value_heads = nnx.Dict({
            f"head_{i}": _gemma.ValueHead(
                input_dim=paligemma_config.width,
                hidden_sizes=(1024, 512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
                rngs=rngs,
            )
            for i in range(self.value_ensemble_size)
        })

    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _get_last_token_output(self, output: at.Array, input_mask: at.Bool[at.Array, "b s"]) -> at.Array:
        """Get the output corresponding to the last real (non-padding) token for each sequence."""
        # Find the last position where input_mask is True for each sequence
        # input_mask shape: [batch_size, seq_len]
        # We want to find the last True position for each batch item
        
        # Convert mask to indices: True -> position index, False -> -1
        # Then take the maximum index for each sequence (which gives us the last real token position)
        seq_len = input_mask.shape[1]
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        
        # Create a tensor where True positions have their index, False positions have -1
        masked_positions = jnp.where(input_mask, positions, -1)
        
        # Find the maximum position for each sequence (last real token position)
        last_real_positions = jnp.max(masked_positions, axis=1)  # Shape: [batch_size]
        
        # Use these positions to index into the output
        # We need to handle the case where all tokens might be padding (last_real_positions = -1)
        # In that case, we'll use position 0 as a fallback
        last_real_positions = jnp.maximum(last_real_positions, 0)
        
        # Index into output using the last real positions
        # output shape: [batch_size, seq_len, hidden_dim]
        # last_real_positions shape: [batch_size]
        # We want: output[batch_idx, last_real_positions[batch_idx], :] for each batch_idx
        
        batch_size = output.shape[0]
        batch_indices = jnp.arange(batch_size)
        
        # Use advanced indexing to get the last real token output for each sequence
        last_real_output = output[batch_indices, last_real_positions]  # Shape: [batch_size, hidden_dim]
        
        return last_real_output

    def get_value_from_vlm(
        self,
        prefix_output: at.Float[at.Array, "b s emb"],
        mode: Literal["mean_token", "last_token", "first_token"] = "mean_token",
        return_all: bool = False,
    ):

        # -------- token pooling --------
        if mode == "mean_token":
            first_part = prefix_output[:, :512]
            second_part = prefix_output[:, 768:768 + (200 if self.pi05 else 48)]
            pooled = jnp.mean(jnp.concatenate([first_part, second_part], axis=1), axis=1)
        elif mode == "last_token":
            pooled = prefix_output[:, -1]
        elif mode == "first_token":
            pooled = prefix_output[:, 0]
        else:
            raise ValueError(mode)

        # pooled: [B, D]

        # -------- ensemble forward (NNX + einops) --------
        # values_list: List[(B,)]
        values_list = [
            head(pooled)[..., 0]
            for head in self.value_heads.values()
        ]

        # Stack -> [E, B] -> rearrange -> [B, E]
        values = rearrange(
            jnp.stack(values_list, axis=0),
            'e b -> b e'
        )

        if return_all:
            return values

        # Conservative reduction
        return reduce(values, 'b e -> b', 'min')

    def compute_values(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        mode: Literal["mean_token", "last_token", "first_token"] = "mean_token",
        return_all: bool = False,
    ) -> at.Float[at.Array, "b"]:
        """Compute value estimates for observations (for PPO).
        
        This method computes the value function V(s) which estimates the expected return
        from a given state. Used in PPO for advantage estimation via GAE.
        
        Args:
            rng: Random key for preprocessing
            observation: Observations to compute values for
            mode: Pooling mode for value head (default: "mean_token")
            
        Returns:
            Value estimates, shape (batch_size,)
        """
        # Preprocess observation
        observation = _model.preprocess_observation(rng, observation, train=False)
        
        # Backwards-compat: some callers pass `True` as the 3rd positional arg
        # intending `return_all`. If `mode` is boolean, treat it as `return_all`.
        if isinstance(mode, (bool, jnp.bool_)):
            return_all = bool(mode)
            mode = "mean_token"

        # Get prefix embeddings
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Forward pass through VLM to get representations
        (prefix_output, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], 
            mask=prefix_attn_mask, 
            positions=prefix_positions
        )
        
        # Compute values from prefix output using value head
        values = self.get_value_from_vlm(prefix_output, mode=mode, return_all=return_all)
        
        return values

    def compute_value_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prev_values_all: at.Float[at.Array, "b e"],
        returns: at.Float[at.Array, "b"],
        value_clip: float = 0.2,
        huber_delta: float = 10.0,
        mode: Literal["mean_token", "last_token", "first_token"] = "mean_token",
        loss_mask: at.Bool[at.Array, "b"] | None = None,
    ):

        observation = _model.preprocess_observation(rng, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_output, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        # [B, E]
        current_values_all = self.get_value_from_vlm(
            jax.lax.stop_gradient(prefix_output),
            mode=mode,
            return_all=True,
        )

        # PPO value clipping
        value_pred_clipped_all = prev_values_all + jnp.clip(
            current_values_all - prev_values_all,
            -value_clip,
            value_clip,
        )

        # -------- vectorized Huber loss --------
        def huber(error, delta):
            abs_e = jnp.abs(error)
            quad = jnp.minimum(abs_e, delta)
            lin = abs_e - quad
            return 0.5 * quad** 2 + delta * lin

        errors = returns[:, None] - current_values_all
        errors_clipped = returns[:, None] - value_pred_clipped_all

        loss_orig = huber(errors, huber_delta)
        loss_clip = huber(errors_clipped, huber_delta)

        value_loss = jnp.maximum(loss_orig, loss_clip)  # [B, E]

        if loss_mask is not None:
            value_loss = value_loss * loss_mask[:, None]
            critic_loss = reduce(value_loss, 'b e ->', 'sum') / jnp.sum(loss_mask)
        else:
            critic_loss = reduce(value_loss, 'b e ->', 'mean')

        # Average over ensemble
        critic_loss /= self.value_ensemble_size

        # Clip statistics
        clip_indicator = jnp.abs(value_pred_clipped_all - prev_values_all) > value_clip
        if loss_mask is not None:
            clip_fraction = (
                reduce(clip_indicator * loss_mask[:, None], 'b e ->', 'sum')
                / jnp.sum(loss_mask)
            )
        else:
            clip_fraction = reduce(clip_indicator, 'b e ->', 'mean')

        info = {
            "value_loss": critic_loss,
            "value_clip_fraction": clip_fraction,
            "returns": jnp.mean(returns),
            "current_values_mean": jnp.mean(current_values_all),
            "prev_values_mean": jnp.mean(prev_values_all),
        }

        return critic_loss, info

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        return jnp.mean(jnp.square(v_t - u_t))

    def compute_sft_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        use_advantages: bool = False,
        advantages: at.Float[at.Array, ""] = None,
        train: bool = False
    ) -> tuple[at.Float[at.Array, ""], dict]:
        """Compute supervised fine-tuning loss using diffusion denoising.
        
        Args:
            rng: Random key for sampling.
            observation: Current observation.
            actions: Expert actions to learn from.
            train: Whether we're in training mode.
            
        Returns:
            Tuple of (loss, info_dict) where info_dict contains detailed metrics.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        
        # Preprocess observation
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        
        # Generate noise and time for diffusion process
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Get prefix embeddings and compute kv_cache (frozen)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # PaliGemma has two configs, so we need to pass [tokens, None] for the two experts
        _, prefix_kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)
        prefix_kv_cache = jax.lax.stop_gradient(prefix_kv_cache)
        
        # Compute suffix tokens for noisy actions
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        
        # Run the model to predict the noise using kv_cache
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=prefix_kv_cache,
            adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        
        # Compute MSE loss between predicted and actual noise
        if not use_advantages:
            advantages = jnp.ones_like(v_t[..., 0])
        elif advantages is None:
            advantages = jax.lax.stop_gradient(self.get_advantages(observation, actions))
        else:
            advantages = jax.lax.stop_gradient(advantages)
        
        mse_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        weighted_loss = jnp.mean(advantages * mse_loss)
        
        info = {
            "loss": weighted_loss,
            "advantages": jnp.mean(advantages),
        }
        
        return weighted_loss, info
    
    def _compute_logprob_gaussian(
        self, 
        sample: at.Float[at.Array, "b ah ad"],
        mu: at.Float[at.Array, "b ah ad"],
        sigma: at.Float[at.Array, "b ah ad"],
    ) -> at.Float[at.Array, "b ah ad"]:
        """Compute log probability of a Gaussian distribution.
        
        Args:
            sample: Sample value
            mu: Mean
            sigma: Standard deviation
            
        Returns:
            Log probability: log p(sample | mu, sigma)
        """
        mask = sigma == 0
        sigma_safe = jnp.where(mask, jnp.ones_like(sigma), sigma)
        constant_term = -jnp.log(sigma_safe) - 0.5 * jnp.log(2.0 * jnp.pi * jnp.ones_like(sample))
        exponent_term = -0.5 * jnp.square((sample - mu) / sigma_safe)
        log_prob = constant_term + exponent_term
        log_prob = jnp.where(mask, jnp.zeros_like(log_prob), log_prob)   
        return log_prob

    def recompute_logprobs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        chains: at.Float[at.Array, "b steps ah ad"],
        denoise_inds: at.Int[at.Array, "b num_steps"],
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Recompute log-probs.
        
        Args:
            rng: Random key for preprocessing
            observation: Current observation
            chains: Sequence of states x_t with shape (batch, num_steps+1, action_horizon, action_dim)
            denoise_inds: Denoising step indices, shape (batch, num_steps)
            
        Returns:
            Log-probs with shape (batch, action_horizon, action_dim)
        """
        # Ensure observation is preprocessed
        observation = _model.preprocess_observation(rng, observation, train=False)
        if params is not None:
            self = nnx.merge(nnx.graphdef(self), params)
        
        batch_size = observation.state.shape[0]
        total_steps = chains.shape[1] - 1  # chains has num_steps+1 elements
        
        # Build KV cache for prefix once (same as get_log_prob_value)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_output, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], 
            mask=prefix_attn_mask, 
            positions=prefix_positions
        )
        kv_cache = jax.lax.stop_gradient(kv_cache)
        
        # Setup timesteps and sigmas for flow_sde (matching sample_mean_var_val)
        timesteps = jnp.linspace(1.0, 1.0 / total_steps, total_steps)
        timesteps = jnp.concatenate([timesteps, jnp.array([0.0])])
        sigmas = self.noise_level * jnp.sqrt(
            timesteps / (1.0 - jnp.where(timesteps == 1.0, timesteps[1], timesteps))
        )[:-1]
        batch_indices = jnp.arange(batch_size)
        denoise_ind = denoise_inds[:, 0]
        x_t_mean, x_t_std = self._sample_mean_var_val_jax(
            chains[batch_indices, denoise_ind], denoise_inds[:, 0], observation, timesteps, sigmas,
            kv_cache, prefix_mask, "train"
        )

        chains_next = chains[batch_indices, denoise_ind + 1]
        
        # Compute log probability (matching get_logprob_norm)
        log_probs = self._compute_logprob_gaussian(chains_next, x_t_mean, x_t_std)
        
        return log_probs  # (batch, action_horizon, action_dim)
    
    def compute_ppo_actor_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        chains: at.Float[at.Array, "b steps ah ad"],
        denoise_inds: at.Int[at.Array, "b num_steps"],
        old_logprobs: at.Float[at.Array, "b steps ah ad"],
        advantages: at.Float[at.Array, "b"],
        clip_ratio: float = 0.2,
        clip_ratio_negative: float | None = None,
        valid_action_dim: int = 7,
        loss_mask: at.Bool[at.Array, "b"] | None = None,
    ) -> tuple[at.Float[at.Array, ""], dict]:
        """Compute PPO actor loss with clipped surrogate objective.
        
        Args:
            rng: Random key
            observation: Current observation
            chains: Action chains from buffer
            denoise_inds: Denoise indices from buffer
            old_logprobs: Old log probabilities from buffer
            advantages: PPO advantages (from GAE), shape (batch,)
            clip_ratio: PPO clipping ratio for positive advantages
            clip_ratio_negative: PPO clipping ratio for negative advantages. 
                If None, uses clip_ratio for both positive and negative advantages.
            valid_action_dim: Number of valid action dimensions to use
            
        Returns:
            Tuple of (actor_loss, info_dict)
        """
        batch_size = observation.state.shape[0]
        
        # Select old_logprobs at denoise_inds (matching PyTorch forward logic)
        # For non-joint_logprob case, use denoise_inds[:, 0]
        batch_indices = jnp.arange(batch_size)
        old_logprobs = old_logprobs[batch_indices, denoise_inds[:, 0]]  # (batch, ah, ad)
        
        # Recompute logprobs with current policy
        new_logprobs = self.recompute_logprobs(rng, observation, chains, denoise_inds)  # (batch, ah, ad)
        
        # Only use valid action dimensions
        old_logprobs = old_logprobs[:, :self.action_horizon, :valid_action_dim].sum(axis=[1,2])  # (batch, ah, valid_action_dim)
        new_logprobs = new_logprobs[:, :self.action_horizon, :valid_action_dim].sum(axis=[1,2])  # (batch, ah, valid_action_dim)
        
        loss_mask_count = jnp.maximum(jnp.sum(loss_mask), 1.0)
        
        # For numerical stability: mask out invalid entries when computing ratio
        logprob_diff = new_logprobs - old_logprobs
        ratio = jnp.where(loss_mask, jnp.exp(logprob_diff), 0.0)
        approx_kl = jnp.where(loss_mask, jax.lax.stop_gradient(logprob_diff), 0.0)
        
        # Use clip_ratio_negative if provided, otherwise use clip_ratio for both
        if clip_ratio_negative is None:
            clip_ratio_negative = clip_ratio
        
        # Asymmetric clipping: different clip ratios for positive and negative advantages
        # For positive advantages: clip ratio in [1 - clip_ratio, 1 + clip_ratio]
        # For negative advantages: clip ratio in [1 - clip_ratio_negative, 1 + clip_ratio_negative]
        positive_mask = advantages >= 0.0
        
        # Compute clipped ratio based on advantage sign
        clipped_ratio_pos = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        clipped_ratio_neg = jnp.clip(ratio, 1.0 - clip_ratio_negative, 1.0 + clip_ratio_negative)
        clipped_ratio = jnp.where(positive_mask, clipped_ratio_pos, clipped_ratio_neg)
        
        # Compute policy losses
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * clipped_ratio
        
        # PPO loss: max(policy_loss1, policy_loss2)
        # Compute clipping statistics before applying mask (for correct clip_mask calculation)
        clip_mask = policy_loss1 < policy_loss2
        
        # Apply mask to losses before computing final loss
        policy_loss = jnp.sum(jnp.maximum(policy_loss1, policy_loss2) * loss_mask) / loss_mask_count
        clip_mask = jnp.logical_and(clip_mask, loss_mask)
        clip_fraction = jnp.sum(clip_mask) / loss_mask_count
        approx_kl_mean = -jnp.sum(approx_kl) / loss_mask_count
        
        # Additional statistics for positive/negative advantages
        pos_mask = jnp.logical_and(positive_mask, loss_mask)
        neg_mask = jnp.logical_and(~positive_mask, loss_mask)
        pos_count = jnp.maximum(jnp.sum(pos_mask), 1.0)
        neg_count = jnp.maximum(jnp.sum(neg_mask), 1.0)
        
        info = {
            "policy_loss": policy_loss,
            "ratio": jnp.sum(ratio * loss_mask) / loss_mask_count,
            "clipped_ratio": jnp.sum(clipped_ratio * loss_mask) / loss_mask_count,
            "approx_kl": approx_kl_mean,
            "clip_fraction": clip_fraction,
            "advantages": jnp.sum(advantages * loss_mask) / loss_mask_count,
            "log_probs_mean": jnp.sum(old_logprobs * loss_mask) / loss_mask_count,
            "new_log_probs_mean": jnp.sum(new_logprobs * loss_mask) / loss_mask_count,
            "advantages_positive": jnp.sum(advantages * pos_mask) / pos_count,
            "advantages_negative": jnp.sum(advantages * neg_mask) / neg_count,
            "ratio_positive": jnp.sum(ratio * pos_mask) / pos_count,
            "ratio_negative": jnp.sum(ratio * neg_mask) / neg_count,
            "clip_fraction_positive": jnp.sum(jnp.logical_and(clip_mask, positive_mask)) / pos_count,
            "clip_fraction_negative": jnp.sum(jnp.logical_and(clip_mask, ~positive_mask)) / neg_count,
        }
        
        return policy_loss, info

    def compute_kl_divergence(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        chains: at.Float[at.Array, "b steps ah ad"],
        denoise_inds: at.Int[at.Array, "b num_steps"],
        ref_params: at.Params | None,
        valid_action_dim: int = 7,
        loss_mask: at.Bool[at.Array, "b"] | None = None,
        kl_div_weight: float = 0.0,
    ) -> tuple[at.Float[at.Array, ""], dict]:
        """Compute a KL regularization term between current policy and ref_model.
        
        KL is approximated similarly to `approx_kl`, but using ref_model logprobs:
        D_KL(pi || pi_ref) ≈ E[log pi(a|s) - log pi_ref(a|s)].
        """
        if kl_div_weight == 0.0 or ref_params is None:
            zero = jnp.array(0.0, dtype=jnp.float32)
            return zero, {"kl_div": zero}
        
        # Current policy logprobs
        cur_logprobs_full = self.recompute_logprobs(
            rng,
            observation,
            chains,
            denoise_inds,
        )  
        
        # Ref model logprobs
        ref_logprobs_full = jax.lax.stop_gradient(self.recompute_logprobs(
            rng,
            observation,
            chains,
            denoise_inds,
            ref_params,
        ))
        
        # Only use valid action dimensions and sum over (ah, ad)
        cur_logprobs = cur_logprobs_full[:, :self.action_horizon, :valid_action_dim].sum(axis=[1, 2])
        ref_logprobs = ref_logprobs_full[:, :self.action_horizon, :valid_action_dim].sum(axis=[1, 2])
        
        if loss_mask is None:
            loss_mask = jnp.ones_like(cur_logprobs, dtype=jnp.bool_)
        loss_mask_count = jnp.maximum(jnp.sum(loss_mask), 1.0)
        
        logprob_diff = cur_logprobs - ref_logprobs
        approx_kl = jnp.where(
            loss_mask,
            logprob_diff,
            0.0,
        )
        approx_kl_mean = jnp.sum(approx_kl) / loss_mask_count
        
        kl_loss = kl_div_weight * approx_kl_mean
        
        info = {
            "kl_div": approx_kl_mean,
        }
        return kl_loss, info

    def _sample_mean_var_val_jax(
        self,
        x_t: at.Float[at.Array, "b ah ad"],
        step_idx: int,
        observation: _model.Observation,
        timesteps: at.Float[at.Array, "steps"],
        sigmas: at.Float[at.Array, "steps"],
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
        step_mode: str,
    ) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, "b ah ad"]]:
        """sample_mean_var_val method."""
        batch_size = x_t.shape[0]
        t_current = timesteps[step_idx]
        t_next = timesteps[step_idx + 1]
        delta = jnp.broadcast_to((t_current - t_next)[:, None, None], x_t.shape)
        sigma_i = jnp.broadcast_to(sigmas[step_idx][:, None, None], x_t.shape)

        # Get model predictions
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, t_current
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        # Compute x0 and x1 predictions
        x0_pred = x_t - v_t * t_current[:, None, None]
        x1_pred = x_t + v_t * (1.0 - t_current[:, None, None])
        t_current = jnp.broadcast_to(t_current[:, None, None], x_t.shape)

        # Compute weights and std based on mode
        if step_mode == "eval":
            # Deterministic sampling
            x0_weight = 1.0 - (t_current - delta)
            x1_weight = (t_current - delta)
            x_t_std = jnp.zeros_like(x_t)
        else:  # step_mode == "train"
            # Stochastic sampling for flow_sde
            x0_weight = jnp.ones_like(t_current) - (t_current - delta)
            x1_weight = t_current - delta - sigma_i**2 * delta / (2.0 * t_current)
            x_t_std = jnp.sqrt(delta) * sigma_i

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def compute_ratio(
        self,
        observation: _model.Observation,
        chains: at.Float[at.Array, "b steps ah ad"],
        old_logprobs: at.Float[at.Array, "b steps ah ad"],
        valid_action_dim: int = 7,
    ) -> at.Float[at.Array, "b"]:
        """Compute ratios.
        
        Args:
            observation: Current observation
            chains: Action chains from buffer
            old_logprobs: Old log probabilities from buffer
            valid_action_dim: Number of valid action dimensions
            
        Returns:
            log ratios array of shape (b,)
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        total_steps = chains.shape[1] - 1
        
        # Build KV cache for prefix once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_output, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], 
            mask=prefix_attn_mask, 
            positions=prefix_positions
        )
        kv_cache = jax.lax.stop_gradient(kv_cache)
        
        # Setup timesteps and sigmas
        timesteps = jnp.linspace(1.0, 1.0 / total_steps, total_steps)
        timesteps = jnp.concatenate([timesteps, jnp.array([0.0])])
        sigmas = self.noise_level * jnp.sqrt(
            timesteps / (1.0 - jnp.where(timesteps == 1.0, timesteps[1], timesteps))
        )[:-1]
        logprobs = []
        for step_idx in range(total_steps):
            # x_t is input at step_idx
            x_t = chains[:, step_idx]
            
            # So we must broadcast step_idx to (batch_size,)
            step_idx_broadcast = jnp.full((batch_size,), step_idx, dtype=jnp.int32)
            
            x_t_mean, x_t_std = self._sample_mean_var_val_jax(
                x_t, step_idx_broadcast, observation, timesteps, sigmas,
                kv_cache, prefix_mask, "train"
            )
            
            # x_next is target at step_idx + 1
            x_next = chains[:, step_idx + 1]
            
            step_logprob = self._compute_logprob_gaussian(x_next, x_t_mean, x_t_std)
            logprobs.append(step_logprob)
        old_logprobs = old_logprobs[:, :, :self.action_horizon, :valid_action_dim].sum(axis=[1,2,3])
        logprobs = jnp.stack(logprobs, axis=1)[:, :, :self.action_horizon, :valid_action_dim].sum(axis=[1,2,3])
        ratios = logprobs - old_logprobs
        
        return ratios

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 3,
        mode: Literal["train", "eval"] = "eval",
        return_logprobs: bool = False,
        return_values: bool = False,
        joint_logprob: bool = False,
    ) -> tuple[_model.Actions, dict[str, at.Array]] | _model.Actions:
        """Sample actions."""
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        noise_rng, step_rng = jax.random.split(rng)

        # Build KV cache for prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_output, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        kv_cache = jax.lax.stop_gradient(kv_cache)

        # Setup timesteps and sigmas
        timesteps = jnp.linspace(1.0, 1.0 / num_steps, num_steps)
        timesteps = jnp.concatenate([timesteps, jnp.array([0.0])])
        sigmas = (
            self.noise_level 
            * jnp.sqrt(
                timesteps 
                / (1.0 - jnp.where(timesteps == 1.0, timesteps[1], timesteps))
            )[:-1]
        )
        # Initial noise
        x_t = jax.random.normal(noise_rng, (batch_size, self.action_horizon, self.action_dim))
        
        # Initialize storage lists
        chains = [x_t]
        log_probs = []

        # Compute VLM-based values if needed
        if return_values:
            values = self.get_value_from_vlm(prefix_output, return_all=True)

        # Determine denoising indices based on mode
        if mode == "train":
            # Random selection for flow_sde, same for all batch items
            selected_step = jax.random.randint(step_rng, (), 0, num_steps)
            denoise_inds = jnp.full((batch_size, num_steps), selected_step, dtype=jnp.int32)
        else:
            # In eval mode, no step is selected for stochastic sampling
            selected_step = -jnp.ones(())
            denoise_inds = jnp.full((batch_size, num_steps), selected_step, dtype=jnp.int32)

        # Denoising loop
        for step_idx in range(num_steps):
            idx = jnp.broadcast_to(step_idx, (batch_size))
            def train_branch(_):
                return self._sample_mean_var_val_jax(
                    x_t, idx, observation, timesteps, sigmas,
                    kv_cache, prefix_mask, "train"
                )

            def eval_branch(_):
                return self._sample_mean_var_val_jax(
                    x_t, idx, observation, timesteps, sigmas,
                    kv_cache, prefix_mask, "eval"
                )

            x_t_mean, x_t_std = jax.lax.cond(
                jnp.logical_or(jnp.equal(selected_step, step_idx), joint_logprob),
                train_branch,
                eval_branch,
                operand=None
            )
            
            # Sample next state - always add noise (matching PyTorch: x_t = x_t_mean + noise * x_t_std)
            step_noise_rng = jax.random.fold_in(step_rng, step_idx)
            step_noise = jax.random.normal(step_noise_rng, x_t.shape)
            x_t = x_t_mean + step_noise * x_t_std

            # Compute log probability for this step
            if return_logprobs:
                step_logprob = self._compute_logprob_gaussian(x_t, x_t_mean, x_t_std)
                log_probs.append(step_logprob)

            chains.append(x_t)

        x_0 = x_t
        chains = jnp.stack(chains, axis=1)
        
        if return_logprobs:
            log_probs = jnp.stack(log_probs, axis=1)
        
        if not return_logprobs and not return_values:
            return x_0
            
        # Return format matching PyTorch implementation
        result = {
            "actions": x_0,
            "chains": chains,
            "denoise_inds": denoise_inds,
        }
        
        if return_logprobs:
            result["prev_logprobs"] = log_probs
        
        if return_values:
            result["prev_values"] = values

        return x_0, result
