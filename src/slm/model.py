from .layers import RoPE, DecoderLayer, RMSNorm
from .config import ModelConfig

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Shaped


class SLModel(nn.Module):
    def __init__(self, config: ModelConfig):
        """Small Language Model

        Args:
            config (ModelConfig): Model Config
        """
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model, padding_idx=config.pad_token_id)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [DecoderLayer(config=config, layer_idx=layer_idx) for layer_idx in range(config.n_layers)]
        )
        self.norm = RMSNorm(hidden_size=config.d_model, eps=config.rmsnorm_eps)
        self.rotary_emb = RoPE(config=config)

    def forward(
        self, 
        input_ids: Float[Tensor, "batch sequence"],
        # TODO will we use custom attention masks or auto generate them with sdpa?
        attention_mask: Shaped[Tensor, "batch sequence"] | None = None 
    ) -> Float[Tensor, "batch sequence d_model"]:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)

        # all position ids start from 0
        # TODO add support for sequence packing; this changes position ids with regiular restarts from 0
        # also impacking RoPE
        position_ids = torch.arange(S, dtype = torch.long).unsqueeze(0).expand(B, -1) 
        position_embeds = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, attention_mask, position_embeds)
        x = self.norm(x)
        return x
    
class CausalSLM(nn.Module):
    def __init__(self, config: ModelConfig):
        """ Causal Small Language Model

        Args:
            config (ModelConfig): Model Config
        """
        super().__init__()
        self.config = config
        self.model = SLModel(config=config)
        self.lm_head = nn.Linear(in_features=config.d_model, out_features=config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        # print(f"NUMBER OF PARAMETERS IN QWEN 2.5 0.5B IS {self.count_params()} i.e. {self.count_params() / 1e6}")

    def forward(
        self, 
        input_ids: Float[Tensor, "batch sequence"], 
        attention_mask: Shaped[Tensor, "batch sequence"] | None = None
    )->tuple[
        Float[Tensor, "batch sequence d_model"], 
        Float[Tensor, "batch sequence d_model"]
    ]:
        hidden_state = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_state)
        return logits, hidden_state

    @torch.no_grad()
    def generate(
        self,
        input_ids: Float[Tensor, "batch sequence"],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None
    ) -> torch.Tensor:
        """
        Autoregressively generates tokens given a prompt.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            top_k: Only sample from the top_k logits.
            eos_token_id: Optional token ID at which to stop generation.

        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens) with generated sequences.
        """
        self.eval()
        B, S = input_ids.shape

        for _ in range(max_new_tokens):
            attention_mask = torch.ones((B, 1, 1, S), device=input_ids.device)  # assumes all tokens are valid
            logits, _ = self.forward(input_ids, attention_mask)  # (batch, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # take the last token's logits

            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-8) # avoid division by 0 

            # Top-k filtering
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, -float('Inf')),
                    next_token_logits
                )

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Stop if all sequences have generated eos_token_id
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return input_ids