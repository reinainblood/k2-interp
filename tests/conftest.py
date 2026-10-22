from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def native_model(tmp_path_factory):
    """Real pinned upstream model code, random tiny weights; never scientific evidence."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM

    from k2_interp.registry import CHECKPOINTS, REPO_ID

    path = Path(
        snapshot_download(
            REPO_ID,
            revision=CHECKPOINTS["sft_3_1500"],
            allow_patterns=["*.py", "config.json"],
            cache_dir=tmp_path_factory.mktemp("hub"),
        )
    )
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    for key, value in dict(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0],
        vocab_size=40,
        num_shared_experts=1,
    ).items():
        setattr(cfg, key, value)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True, attn_implementation="eager", dtype=torch.float32
    ).eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def tokenizer():
    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [2 + (ord(c) % 35) for c in text] or [2]

        def apply_chat_template(self, messages, **kwargs):
            return self.encode(" ".join(m["content"] for m in messages))

        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids))

        def convert_ids_to_tokens(self, ids):
            return list(map(str, ids))

    return Tokenizer()


@pytest.fixture(scope="session")
def native_tokenizer(tmp_path_factory):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from k2_interp.registry import CHECKPOINTS, REPO_ID

    path = snapshot_download(
        REPO_ID,
        revision=CHECKPOINTS["sft_3_1500"],
        allow_patterns=[
            "config.json",
            "*.py",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ],
        cache_dir=tmp_path_factory.mktemp("tokenizer"),
    )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
