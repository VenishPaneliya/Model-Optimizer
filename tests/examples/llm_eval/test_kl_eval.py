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

"""Offline correctness cases for the KL example; no checkpoints or datasets are downloaded."""

import copy
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BloomConfig,
    BloomForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)


@pytest.fixture
def kl_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "llm_eval"))
    return importlib.import_module("kl_eval")


@pytest.mark.parametrize("chunk_size", [1, 2, 32])
def test_kl_direction_shared_support_and_chunk_reduction(kl_eval, chunk_size):
    p = [0.5, 0.3, 0.2]
    q = [0.25, 0.25, 0.5]  # Candidate's favorite token is outside the reference top-2.
    reference = torch.tensor([p, p, q]).log()
    quantized = torch.tensor([q, q, q]).log()
    actual = kl_eval.mean_kl(
        reference, quantized, top_k=2, chunk_size=chunk_size, full_vocab_kl=True
    )
    expected_full = sum(a * math.log(a / b) for a, b in zip(p, q)) * 2 / 3
    expected_head = (0.625 * math.log(0.625 / 0.5) + 0.375 * math.log(0.375 / 0.5)) * 2 / 3
    assert actual["full_vocab_kl"] == pytest.approx(expected_full, abs=1e-6)
    assert actual["conditional_topk_kl"] == pytest.approx(expected_head, abs=1e-6)
    entire_vocab = kl_eval.mean_kl(reference, quantized, top_k=3)
    assert entire_vocab["conditional_topk_kl"] == pytest.approx(expected_full, abs=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_spaced_full_kl_uses_entire_response_and_short_eos(kl_eval, chunk_size):
    reference = torch.tensor([[0.5, 0.5]] * 6).log()
    candidate = torch.tensor(
        [[0.5, 0.5], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.05, 0.95]]
    ).log()
    actual = kl_eval.mean_kl(
        reference,
        candidate,
        top_k=2,
        score_tokens=3,
        chunk_size=chunk_size,
        position_counts=[1, 3, 6, 8192],
        full_vocab_kl=True,
    )
    per_token = (reference.exp() * (reference - candidate)).sum(-1).tolist()
    # EAR's endpoint grid for six positions and three samples is [0, 2, 5].
    assert actual["spaced_full_vocab_kl_3"] == pytest.approx(
        (per_token[0] + per_token[2] + per_token[5]) / 3, abs=1e-6
    )
    assert actual["conditional_topk_kl"] == pytest.approx(
        actual["spaced_full_vocab_kl_3"], abs=1e-6
    )
    conditional = kl_eval.mean_kl(
        reference, candidate, top_k=2, score_tokens=3, chunk_size=chunk_size
    )
    assert conditional == pytest.approx(
        {"conditional_topk_kl": actual["conditional_topk_kl"]}, abs=1e-6
    )
    assert actual["spaced_full_vocab_kl_1"] == pytest.approx(per_token[0], abs=1e-6)
    assert actual["spaced_full_vocab_kl_6"] == actual["full_vocab_kl"]
    assert actual["spaced_full_vocab_kl_8192"] == actual["full_vocab_kl"]
    with pytest.raises(ValueError):
        kl_eval.mean_kl(reference, candidate, position_counts=[0], top_k=None, full_vocab_kl=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_identical_logits_have_zero_kl(kl_eval, dtype):
    logits = torch.tensor([[1, 2, -3], [8, 4, 1]], dtype=dtype)
    metrics = kl_eval.mean_kl(logits, logits.clone(), top_k=2, full_vocab_kl=True)
    assert metrics == pytest.approx({"full_vocab_kl": 0, "conditional_topk_kl": 0}, abs=1e-7)


def _tiny_model(tuple_default=False):
    if tuple_default:
        return BloomForCausalLM(
            BloomConfig(
                vocab_size=16,
                hidden_size=8,
                n_layer=1,
                n_head=1,
                bos_token_id=1,
                eos_token_id=0,
                pad_token_id=0,
                return_dict=False,
                hidden_dropout=0,
                attention_dropout=0,
            )
        ).eval()
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    ).eval()


@pytest.mark.parametrize("tuple_default", [False, True])
@pytest.mark.parametrize(("eos_token_id", "end"), [(None, 7), (0, 6), ([0, 8], 5)])
def test_scores_all_and_only_continuation_predictions(kl_eval, eos_token_id, end, tuple_default):
    torch.manual_seed(0)
    reference = _tiny_model(tuple_default)
    reference.generation_config.eos_token_id = eos_token_id
    quantized = copy.deepcopy(reference)
    with torch.no_grad():
        quantized.lm_head.weight[:, 0].add_(torch.linspace(-3, 3, reference.config.vocab_size))
    # EOS in the prompt is ignored; scalar/list EOS settings stop at different positions.
    sequence = torch.tensor([[0, 5, 6, 7, 8, 0, 0]])
    with torch.inference_mode():
        if tuple_default:
            assert isinstance(reference(sequence[:, :-1], use_cache=False), tuple)
        p = (
            reference(sequence[:, :-1], use_cache=False, return_dict=True)
            .logits[0, 2 : end - 1]
            .float()
        )
        q = (
            quantized(sequence[:, :-1], use_cache=False, return_dict=True)
            .logits[0, 2 : end - 1]
            .float()
        )
        per_position = (p.softmax(-1) * (p.log_softmax(-1) - q.log_softmax(-1))).sum(-1)
        expected = per_position.mean().item()
        ids = p.topk(4, dim=-1).indices
        p_head, q_head = p.gather(-1, ids), q.gather(-1, ids)
        per_head_position = (
            p_head.softmax(-1) * (p_head.log_softmax(-1) - q_head.log_softmax(-1))
        ).sum(-1)
        expected_head = (per_head_position[0].item() + per_head_position[-1].item()) / 2
    assert expected > 1e-6
    assert expected_head > 1e-6
    actual = kl_eval.score_continuation(
        reference,
        quantized,
        sequence,
        prompt_tokens=3,
        top_k=4,
        full_vocab_kl=True,
        position_counts=[1, 2, 64],
        score_tokens=2,
    )
    assert actual == pytest.approx(
        {
            "full_vocab_kl": expected,
            "conditional_topk_kl": expected_head,
            "spaced_full_vocab_kl_1": per_position[0].item(),
            "spaced_full_vocab_kl_2": (per_position[0].item() + per_position[-1].item()) / 2,
            "spaced_full_vocab_kl_64": expected,
        },
        abs=1e-6,
    )


@pytest.mark.parametrize("detailed_results", [False, True])
@pytest.mark.parametrize(("top_k", "full_vocab_kl"), [(4, False), (None, True), (4, True)])
def test_real_generation_stops_at_eos_and_scores_it(
    kl_eval, monkeypatch, detailed_results, top_k, full_vocab_kl
):
    argv = ["kl_eval.py", "--model", "local-test-model", "--recipe", "local-test-recipe"]
    if top_k is not None:
        argv.extend(["--top_k", str(top_k)])
    if full_vocab_kl:
        argv.append("--full_vocab_kl")
    if detailed_results:
        argv.append("--detailed_results")
    monkeypatch.setattr(sys, "argv", argv)
    args = kl_eval._parse_args()
    assert args.full_vocab_kl == full_vocab_kl
    assert args.eval_batch_size == 8
    assert args.max_new_tokens == 1024
    assert args.top_k == (2048 if top_k is None else top_k)
    assert args.score_tokens == 512
    assert args.num_examples == 300
    assert args.eval_revision == kl_eval.EVALUATION_REVISION
    reference = _tiny_model()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.zero_()
    # Equal logits greedily choose token 0, which is this model's EOS.
    # Checkpoint decoding settings must not override the evaluation's plain greedy policy.
    reference.generation_config.suppress_tokens = [0]
    reference.generation_config.return_dict_in_generate = True
    original_generation = copy.deepcopy(reference.generation_config)
    quantized = copy.deepcopy(reference)
    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)
    prompts = [{"id": "fixture-3", "source": "fixture", "cohort_rank": 3, "input_ids": [4, 5, 6]}]
    result = kl_eval.evaluate(
        reference,
        quantized,
        tokenizer,
        prompts,
        max_new_tokens=4,
        top_k=top_k,
        detailed_results=args.detailed_results,
        position_counts=[1, 64, 64] if full_vocab_kl else [],
        full_vocab_kl=args.full_vocab_kl,
    )
    assert reference.generation_config == original_generation
    expected = {}
    if full_vocab_kl:
        expected.update(
            {"full_vocab_kl": 0, "spaced_full_vocab_kl_1": 0, "spaced_full_vocab_kl_64": 0}
        )
    if top_k is not None:
        expected["conditional_topk_kl"] = 0
    if detailed_results:
        assert result["examples"] == [
            {
                "example": 0,
                "id": "fixture-3",
                "source": "fixture",
                "cohort_rank": 3,
                "prompt_ids": [4, 5, 6],
                "generated_ids": [0],
                "generated_tokens": 1,
                **expected,
            }
        ]
        assert result["summary"] == pytest.approx(expected, abs=1e-7)
    else:
        assert result == pytest.approx(expected, abs=1e-7)


def test_invalid_logits_fail_instead_of_dropping_positions(kl_eval):
    with pytest.raises(ValueError, match="Non-finite KL"):
        kl_eval.mean_kl(torch.zeros(2, 4), torch.full((2, 4), float("nan")), top_k=2)


@pytest.fixture
def blended_fixture(kl_eval, monkeypatch):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "user": 1, "assistant": 2, "alpha": 3, "beta": 4}))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    )
    dataset = Dataset.from_list(
        [
            {
                "id": str(index),
                "source": "fixture",
                "cohort_rank": index,
                "messages": [{"role": "user", "content": content}],
                "intrinsic_score": 999,
                "gold_answer": "must not enter prompt",
            }
            for index, content in enumerate(["alpha beta", "beta", "alpha " * 128])
        ]
    )
    # Replace only the external Hub read; exercise real dataset selection and chat tokenization.
    monkeypatch.setattr("datasets.load_dataset", lambda *_a, **_kw: dataset)
    return tokenizer, dataset


def test_blended_prompts_preserve_messages_order_and_identity(kl_eval, blended_fixture):
    tokenizer, dataset = blended_fixture
    prompts, fingerprint = kl_eval._blended_prompts(tokenizer, 3, "revision", 256, 4)
    assert prompts == [
        {"id": "0", "source": "fixture", "cohort_rank": 0, "input_ids": [1, 3, 4, 2]},
        {"id": "1", "source": "fixture", "cohort_rank": 1, "input_ids": [1, 4, 2]},
        {"id": "2", "source": "fixture", "cohort_rank": 2, "input_ids": [1] + [3] * 128 + [2]},
    ]
    assert fingerprint == dataset._fingerprint


@pytest.mark.parametrize(
    ("num_examples", "context_limit", "error"),
    [(0, 256, "dataset rows"), (4, 256, "dataset rows"), (3, 133, "not truncated")],
)
def test_blended_prompts_refuse_short_banks_and_context_overflow(
    kl_eval, blended_fixture, num_examples, context_limit, error
):
    tokenizer, _ = blended_fixture
    with pytest.raises(ValueError, match=error):
        kl_eval._blended_prompts(tokenizer, num_examples, "revision", context_limit, 4)


def test_full_vocab_only_skips_topk(kl_eval, monkeypatch):
    def unexpected_topk(*_args, **_kwargs):
        pytest.fail("Full-vocabulary evaluation must not calculate a conditional head")

    monkeypatch.setattr(torch.Tensor, "topk", unexpected_topk)
    logits = torch.randn(3, 16)
    result = kl_eval.mean_kl(logits, logits.clone(), top_k=None, full_vocab_kl=True)
    assert result == pytest.approx({"full_vocab_kl": 0}, abs=1e-7)


@pytest.mark.parametrize("top_k", [None, 4])
def test_batched_scoring_excludes_padding_and_matches_single_rows(kl_eval, top_k):
    reference = _tiny_model()
    quantized = copy.deepcopy(reference)
    with torch.no_grad():
        quantized.lm_head.weight[3, 0].add_(3)
    sequences = [[4, 5, 6, 7, 8], [4, 5, 0], [4, 5, 6, 7, 8, 9]]
    lengths = [3, 2, 2]
    counts = [1, 3, 8192]
    actual = kl_eval._score_batch(
        reference, quantized, sequences, lengths, 0, top_k, counts, full_vocab_kl=True
    )
    for sequence, prompt_length, metrics in zip(sequences, lengths, actual):
        # Independent unpadded forwards verify alignment, EOS inclusion, and per-row means.
        inputs = torch.tensor([sequence[:-1]])
        with torch.inference_mode():
            p = reference(inputs, use_cache=False).logits[0, prompt_length - 1 :]
            q = quantized(inputs, use_cache=False).logits[0, prompt_length - 1 :]
        expected = kl_eval.mean_kl(p, q, top_k, position_counts=counts, full_vocab_kl=True)
        assert metrics == pytest.approx(expected, abs=1e-6)


def test_generation_and_scoring_use_eight_rows_then_partial_batch(kl_eval):
    reference = _tiny_model()
    quantized = copy.deepcopy(reference)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.zero_()
    prompts = [
        {"id": str(i), "source": "fixture", "cohort_rank": i, "input_ids": [4, 5] + [6] * (i % 3)}
        for i in range(10)
    ]
    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)
    sizes = {"reference": [], "candidate": []}

    def record(name):
        def hook(_module, _args, kwargs):
            sizes[name].append(kwargs["input_ids"].shape[0])

        return hook

    hooks = [
        reference.register_forward_pre_hook(record("reference"), with_kwargs=True),
        quantized.register_forward_pre_hook(record("candidate"), with_kwargs=True),
    ]
    try:
        batched = kl_eval.evaluate(
            reference,
            quantized,
            tokenizer,
            prompts,
            max_new_tokens=4,
            top_k=4,
            detailed_results=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert set(sizes["reference"]) == {8, 2}
    assert sizes["candidate"] == [8, 2]
    singles = kl_eval.evaluate(
        reference,
        quantized,
        tokenizer,
        prompts,
        max_new_tokens=4,
        detailed_results=True,
        eval_batch_size=1,
        top_k=4,
    )
    assert batched["summary"] == pytest.approx(singles["summary"], abs=1e-6)
    for actual, expected in zip(batched["examples"], singles["examples"]):
        assert actual["generated_ids"] == expected["generated_ids"] == [0]
        assert actual["conditional_topk_kl"] == pytest.approx(
            expected["conditional_topk_kl"], abs=1e-6
        )


@pytest.mark.parametrize("full_vocab_kl", [False, True])
def test_fast_check_uses_top_2048_at_512_spaced_positions(kl_eval, full_vocab_kl):
    reference = torch.tensor([[0.0] * 2048 + [-10.0]]).repeat(1024, 1)
    candidate = reference.clone()
    selected = [i * 1023 // 511 for i in range(512)]
    candidate[selected] = torch.tensor([0.0] * 1024 + [math.log(2)] * 1024 + [8.0])
    actual = kl_eval.mean_kl(reference, candidate, full_vocab_kl=full_vocab_kl)
    p_normalizer = 2048 + math.exp(-10)
    q_normalizer = 3072 + math.exp(8)
    p = 1 / p_normalizer
    p_tail = math.exp(-10) / p_normalizer
    expected_full = 0.5 * (
        1024 * p * math.log(q_normalizer / p_normalizer)
        + 1024 * p * math.log(q_normalizer / (2 * p_normalizer))
        + p_tail * math.log(p_tail / (math.exp(8) / q_normalizer))
    )
    expected_head = 0.5 * math.log(1.5) + 0.5 * math.log(0.75)
    expected = {"conditional_topk_kl": expected_head}
    if full_vocab_kl:
        expected["full_vocab_kl"] = expected_full
    assert actual == pytest.approx(expected, abs=1e-6)
    assert kl_eval.mean_kl(reference, candidate, score_tokens=0) == pytest.approx(
        {"conditional_topk_kl": expected_head / 2}, abs=1e-6
    )
    if not full_vocab_kl:
        # An unused vocabulary tail must not enter conditional normalization or KL.
        candidate[:, -1] = float("nan")
        assert kl_eval.mean_kl(reference, candidate) == pytest.approx(expected, abs=1e-6)


def test_position_counts_require_full_vocab_opt_in(kl_eval, monkeypatch):
    argv = ["kl_eval.py", "--model", "fixture", "--recipe", "fixture"]
    monkeypatch.setattr(sys, "argv", argv)
    args = kl_eval._parse_args()
    assert args.max_new_tokens == 1024
    assert args.top_k == 2048
    assert args.score_tokens == 512
    assert not args.full_vocab_kl
    monkeypatch.setattr(sys, "argv", [*argv, "--position_counts", "2"])
    with pytest.raises(SystemExit):
        kl_eval._parse_args()
    monkeypatch.setattr(sys, "argv", [*argv, "--position_counts", "2", "--full_vocab_kl"])
    assert kl_eval._parse_args().position_counts == [2]
    monkeypatch.setattr(sys, "argv", [*argv, "--score_tokens", "0"])
    assert kl_eval._parse_args().score_tokens == 0
    monkeypatch.setattr(sys, "argv", [*argv, "--score_tokens", "-1"])
    with pytest.raises(SystemExit):
        kl_eval._parse_args()
    logits = torch.zeros(3, 4)
    for kwargs in (
        {"position_counts": [2], "top_k": 2},
        {"top_k": None},
        {"top_k": 2, "score_tokens": -1},
    ):
        with pytest.raises(ValueError):
            kl_eval.mean_kl(logits, logits, **kwargs)
        with pytest.raises(ValueError):
            kl_eval.evaluate(None, None, None, [{"input_ids": [1]}], **kwargs)
