# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoPE cos/sin caches for GQA graph surgery."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="transformers required for RoPE cache tests")

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from modelopt.onnx.graph_surgery.utils.rope_cache import get_rope_caches

HEAD_DIM = 64
MAX_SEQ_LEN = 512
LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}


@pytest.mark.parametrize("rope_scaling", [None, LLAMA3_SCALING], ids=["default", "llama3"])
def test_rope_caches_match_hf_rotary_embedding(tmp_path, rope_scaling):
    """Caches equal HF's rotary embedding for a checkpoint-style config.json (top-level rope_theta)."""
    config = {
        "model_type": "llama",
        "hidden_size": 4 * HEAD_DIM,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 512,
        "vocab_size": 64,
        "max_position_embeddings": 4096,
        "rope_theta": 500000.0,
        "rope_scaling": rope_scaling,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    cos, sin, _ = get_rope_caches(str(tmp_path), MAX_SEQ_LEN, io_dtype="float32")

    rotary = LlamaRotaryEmbedding(config=AutoConfig.from_pretrained(str(tmp_path)))
    ref_cos, ref_sin = rotary(
        torch.zeros(1, MAX_SEQ_LEN, HEAD_DIM), torch.arange(MAX_SEQ_LEN)[None]
    )
    np.testing.assert_allclose(cos, ref_cos[0, :, : HEAD_DIM // 2].numpy(), atol=1e-5)
    np.testing.assert_allclose(sin, ref_sin[0, :, : HEAD_DIM // 2].numpy(), atol=1e-5)
