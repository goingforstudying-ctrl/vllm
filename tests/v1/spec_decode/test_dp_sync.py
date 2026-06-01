# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for DP synchronization in speculative decoding to prevent NCCL deadlock.

See https://github.com/vllm-project/vllm/issues/44185 for the bug report.
"""

from unittest import mock

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.load import LoadConfig
from vllm.platforms import current_platform
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

DEVICE = current_platform.device_type


def _create_runner_with_spec_decode(
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
    max_model_len: int = 512,
) -> GPUModelRunner:
    """Create a GPUModelRunner with speculative decoding enabled."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
    )

    draft_model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
        max_model_len=max_model_len,
    )
    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=parallel_config,
        model="facebook/opt-125m",
        method="draft_model",
        num_speculative_tokens=3,
        draft_model_config=draft_model_config,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE),
        load_config=LoadConfig(),
    )

    with set_current_vllm_config(vllm_config):
        runner = GPUModelRunner(vllm_config, DEVICE)
        # We don't call initialize_kv_cache here because it requires a real
        # attention backend. The test targets the sample_tokens logic only.
        return runner


def test_input_fits_in_drafter_sync_skips_when_dp_disabled():
    """When DP is disabled, no synchronization should happen."""
    runner = _create_runner_with_spec_decode(data_parallel_size=1)
    runner.effective_drafter_max_model_len = 100
    runner.num_spec_tokens = 3

    # Mock the drafter so we can tell if it was called
    runner.drafter = mock.MagicMock(spec=DraftModelProposer)
    runner.drafter.prepare_next_token_ids_padded.return_value = (
        torch.zeros(1, device=DEVICE, dtype=torch.int32),
        torch.zeros(1, device=DEVICE, dtype=torch.int32),
    )

    # Build a fake execute_model_state
    from vllm.v1.attention.backend import CommonAttentionMetadata

    common_attn_meta = mock.MagicMock(spec=CommonAttentionMetadata)
    common_attn_meta.max_seq_len = 50  # 50 + 3 <= 100, fits

    sampler_output = mock.MagicMock()
    sampler_output.sampled_token_ids = torch.zeros(
        (1, 1), device=DEVICE, dtype=torch.int32
    )

    runner.execute_model_state = (
        mock.MagicMock(),  # scheduler_output
        mock.MagicMock(),  # logits
        mock.MagicMock(),  # spec_decode_metadata
        common_attn_meta,  # spec_decode_common_attn_metadata
        mock.MagicMock(),  # hidden_states
        mock.MagicMock(),  # sample_hidden_states
        None,  # aux_hidden_states
        None,  # ec_connector_output
        None,  # cudagraph_stats
        None,  # slot_mappings
    )

    # We need to mock _sample and _bookkeeping_sync to avoid full execution
    with (
        mock.patch.object(runner, "_sample", return_value=sampler_output),
        mock.patch.object(runner, "_update_states_after_model_execute"),
        mock.patch.object(runner, "_bookkeeping_sync", return_value=(0, [], None, {}, [], {}, [])),
        mock.patch.object(runner, "finalize_kv_connector"),
        mock.patch.object(runner, "eplb_step"),
    ):
        runner.sample_tokens(grammar_output=None)

    # With DP size 1, the drafter should have been called because input fits
    assert runner.drafter.prepare_next_token_ids_padded.called


def test_input_fits_in_drafter_sync_with_dp():
    """When DP is enabled with a draft model, all ranks must agree on fit."""
    # This test verifies that the synchronization code path is present.
    # A full multi-rank test requires a real distributed environment,
    # so we mock the collective call and assert it is invoked.

    runner = _create_runner_with_spec_decode(
        data_parallel_size=2, data_parallel_rank=0
    )
    runner.effective_drafter_max_model_len = 100
    runner.num_spec_tokens = 3

    runner.drafter = mock.MagicMock(spec=DraftModelProposer)
    runner.drafter.prepare_next_token_ids_padded.return_value = (
        torch.zeros(1, device=DEVICE, dtype=torch.int32),
        torch.zeros(1, device=DEVICE, dtype=torch.int32),
    )

    from vllm.v1.attention.backend import CommonAttentionMetadata

    common_attn_meta = mock.MagicMock(spec=CommonAttentionMetadata)
    common_attn_meta.max_seq_len = 50  # fits locally

    sampler_output = mock.MagicMock()
    sampler_output.sampled_token_ids = torch.zeros(
        (1, 1), device=DEVICE, dtype=torch.int32
    )

    runner.execute_model_state = (
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        common_attn_meta,
        mock.MagicMock(),
        mock.MagicMock(),
        None,
        None,
        None,
        None,
    )

    with (
        mock.patch.object(runner, "_sample", return_value=sampler_output),
        mock.patch.object(runner, "_update_states_after_model_execute"),
        mock.patch.object(runner, "_bookkeeping_sync", return_value=(0, [], None, {}, [], {}, [])),
        mock.patch.object(runner, "finalize_kv_connector"),
        mock.patch.object(runner, "eplb_step"),
        mock.patch(
            "vllm.v1.worker.gpu_model_runner.torch.distributed.all_reduce"
        ) as mock_all_reduce,
    ):
        runner.sample_tokens(grammar_output=None)

    # The all_reduce should have been called because DP size > 1 and we use
    # a draft model.
    assert mock_all_reduce.called, (
        "Expected all_reduce to synchronize input_fits_in_drafter across DP ranks"
    )

    # Verify the tensor value is 1 (fits) and MIN op is used
    call_args = mock_all_reduce.call_args
    tensor_arg = call_args[0][0]
    assert tensor_arg.item() == 1, "Expected fits_tensor to be 1 (True)"
    assert call_args[1]["op"] == torch.distributed.ReduceOp.MIN
