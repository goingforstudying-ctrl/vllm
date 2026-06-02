"""Tests for SchedulerConfig validation."""

import pytest

from vllm.config.scheduler import SchedulerConfig


class TestMaxNumScheduledTokensValidation:
    """Test validation of max_num_scheduled_tokens field."""

    def test_negative_max_num_scheduled_tokens_raises(self):
        """Test that negative max_num_scheduled_tokens raises ValueError.
        
        Regression test for: https://github.com/vllm-project/vllm/issues/44123
        """
        with pytest.raises(ValueError, match="max_num_scheduled_tokens must be positive"):
            SchedulerConfig(
                max_num_batched_tokens=2048,
                max_num_scheduled_tokens=-1,
                max_model_len=8192,
                is_encoder_decoder=False,
            )

    def test_zero_max_num_scheduled_tokens_raises(self):
        """Test that zero max_num_scheduled_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_num_scheduled_tokens must be positive"):
            SchedulerConfig(
                max_num_batched_tokens=2048,
                max_num_scheduled_tokens=0,
                max_model_len=8192,
                is_encoder_decoder=False,
            )

    def test_positive_max_num_scheduled_tokens_ok(self):
        """Test that positive max_num_scheduled_tokens is accepted."""
        config = SchedulerConfig(
            max_num_batched_tokens=2048,
            max_num_scheduled_tokens=1024,
            max_model_len=8192,
            is_encoder_decoder=False,
        )
        assert config.max_num_scheduled_tokens == 1024

    def test_none_max_num_scheduled_tokens_ok(self):
        """Test that None max_num_scheduled_tokens is accepted (default behavior)."""
        config = SchedulerConfig(
            max_num_batched_tokens=2048,
            max_num_scheduled_tokens=None,
            max_model_len=8192,
            is_encoder_decoder=False,
        )
        assert config.max_num_scheduled_tokens is None
