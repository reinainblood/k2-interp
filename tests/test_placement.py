import json
import struct

import pytest

from k2_interp.weights import plan_placement


def test_full_k2_geometry_is_balanced_without_loading_weights(tmp_path):
    # Real-dimensional headers, no 758GB payload: placement reads only metadata.
    tensors = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [250624, 6144]},
        "lm_head.weight": {"dtype": "BF16", "shape": [250624, 6144]},
        "model.norm.weight": {"dtype": "BF16", "shape": [6144]},
    }
    for layer in range(61):
        if layer < 3:
            tensors[f"model.layers.{layer}.mlp.weight"] = {
                "dtype": "BF16",
                "shape": [3, 6144, 16384],
            }
        else:
            tensors[f"model.layers.{layer}.mlp.weight"] = {
                "dtype": "BF16",
                "shape": [193, 3, 6144, 1792],
            }
        tensors[f"model.layers.{layer}.attention.weight"] = {
            "dtype": "BF16",
            "shape": [6144, 6144 * 2 + 1024 * 2],
        }
    header = json.dumps(tensors).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 61}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "weights.safetensors" for k in tensors}})
    )
    mapping, bytes_per_gpu = plan_placement(tmp_path, {i: 128_000_000_000 for i in range(8)})
    assert set(mapping.values()) == set(range(8))
    assert max(bytes_per_gpu.values()) / min(bytes_per_gpu.values()) < 1.25
    assert max(bytes_per_gpu.values()) < 110_000_000_000
    with pytest.raises(MemoryError):
        plan_placement(tmp_path, {i: 70_000_000_000 for i in range(8)})
