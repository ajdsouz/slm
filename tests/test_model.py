import pytest
import torch
from slm.config import ModelConfig
from slm.model import SLModel, CausalSLM

@pytest.fixture
def make_model_config():
    config= ModelConfig(
        block_size=32,
        d_model=128,
        d_ffn=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        max_positional_embed=512,
        rmsnorm_eps=1e-6,
        rope_theta=1e-6,
        dropout=0.1,
        vocab_size=1024,
        activation='silu',
        pad_token_id=0,
        tie_word_embeddings=True
    )

    return config

def test_model_instantiation(make_model_config):
    model = CausalSLM(config = make_model_config)
    print(model)

def test_model_forward(make_model_config):

    B, S = 2, 16
    input_ids = torch.randint(low=0, high=make_model_config.vocab_size, size = (B, S))
    model = CausalSLM(make_model_config)
    logits, hidden_states = model(input_ids) 

    print("Input IDS Shape", input_ids.shape)
    print("Logits Shape", logits.shape)
    print("Hidden State Shape", hidden_states.shape)

    assert logits.shape == (B, S, make_model_config.vocab_size), "Logits shape does not match predicted shape"
    assert hidden_states.shape == (B, S, make_model_config.d_model), "Hidden States shape does not match predicted shape"

    assert torch.isfinite(logits).all()
    assert torch.isfinite(hidden_states).all()

    print("Forward pass successful!")