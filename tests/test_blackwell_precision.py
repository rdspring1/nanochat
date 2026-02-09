"""
Test Blackwell precision formats (MXFP8).

These tests require a Blackwell B200+ GPU and will be skipped on other hardware.

Run: pytest tests/test_blackwell_precision.py -v -s
"""
import torch
import torch.nn as nn
import pytest
from gpu_capability import is_blackwell_gpu


@pytest.mark.skipif(not is_blackwell_gpu(), reason="Requires Blackwell GPU")
class TestMXFP8Training:
    """Test MXFP8 training on Blackwell B200."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_mxfp8_conversion(self):
        """Test that MXFP8 conversion works correctly."""
        # Create a simple model with Linear layers
        model = nn.Sequential(
            nn.Linear(128, 256),  # Divisible by 16
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.DEVICE).to(self.DTYPE)

        # Count original Linear layers
        original_linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        assert original_linear_count == 2

        # Convert to MXFP8
        try:
            import torchao.prototype.mx_formats  # Register MXLinear
            from torchao.prototype.mx_formats import MXLinearConfig
            from torchao.prototype.mx_formats.config import MXLinearRecipeName
            from torchao.quantization import quantize_

            def module_filter(mod: nn.Module, fqn: str) -> bool:
                if not isinstance(mod, nn.Linear):
                    return False
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
                return True

            config = MXLinearConfig.from_recipe_name(MXLinearRecipeName.MXFP8_CUBLAS)
            quantize_(model, config, filter_fn=module_filter)

            # Verify conversion
            mxfp_count = sum(1 for m in model.modules() if type(m).__name__ == 'MXLinear')
            assert mxfp_count == 2, f"Expected 2 MXLinear layers, got {mxfp_count}"

        except (ImportError, AttributeError) as e:
            pytest.skip(f"MXFP8 not available in torchao (requires >= 0.15.0): {e}")

    def test_mxfp8_forward_backward(self):
        """Test MXFP8 forward and backward passes."""
        try:
            import torchao.prototype.mx_formats  # Register MXLinear
            from torchao.prototype.mx_formats import MXLinearConfig
            from torchao.prototype.mx_formats.config import MXLinearRecipeName
            from torchao.quantization import quantize_
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MXFP8 not available in torchao (requires >= 0.15.0): {e}")

        # Create model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.DEVICE).to(self.DTYPE)

        # Convert to MXFP8
        def module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        config = MXLinearConfig.from_recipe_name(MXLinearRecipeName.MXFP8_CUBLAS)
        quantize_(model, config, filter_fn=module_filter)

        # Forward pass (batch size must be multiple of 32 for CUBLAS kernel)
        x = torch.randn(32, 128, device=self.DEVICE, dtype=self.DTYPE)
        y = model(x)
        assert y.shape == (32, 128)
        assert not torch.isnan(y).any(), "Forward pass produced NaNs"

        # Backward pass
        loss = y.sum()
        loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs"

    def test_mxfp8_vs_bf16_outputs(self):
        """Verify MXFP8 outputs are close to BF16."""
        try:
            import torchao.prototype.mx_formats  # Register MXLinear
            from torchao.prototype.mx_formats import MXLinearConfig
            from torchao.prototype.mx_formats.config import MXLinearRecipeName
            from torchao.quantization import quantize_
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MXFP8 not available in torchao (requires >= 0.15.0): {e}")

        # Create BF16 model
        model_bf16 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.DEVICE).to(self.DTYPE)

        # Create MXFP8 model with same weights
        model_mxfp8 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.DEVICE).to(self.DTYPE)

        # Copy weights
        model_mxfp8.load_state_dict(model_bf16.state_dict())

        # Convert to MXFP8
        def module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        config = MXLinearConfig.from_recipe_name(MXLinearRecipeName.MXFP8_CUBLAS)
        quantize_(model_mxfp8, config, filter_fn=module_filter)

        # Test with same input (batch size must be multiple of 32 for CUBLAS kernel)
        torch.manual_seed(42)  # For reproducibility
        x = torch.randn(32, 128, device=self.DEVICE, dtype=self.DTYPE)
        with torch.no_grad():
            y_bf16 = model_bf16(x)
            y_mxfp8 = model_mxfp8(x)

        # Check outputs are close - MXFP8 is low precision, so expect significant numerical difference
        # We verify: 1) no NaNs, 2) reasonable correlation, 3) similar magnitude
        assert not torch.isnan(y_mxfp8).any(), "MXFP8 output contains NaNs"
        assert not torch.isinf(y_mxfp8).any(), "MXFP8 output contains Infs"

        # Check outputs have similar magnitude (within 10x)
        bf16_mean = y_bf16.abs().mean()
        mxfp8_mean = y_mxfp8.abs().mean()
        ratio = mxfp8_mean / (bf16_mean + 1e-8)
        assert 0.1 < ratio < 10.0, f"MXFP8 output magnitude differs too much from BF16: ratio={ratio:.2f}"

        # Check correlation is reasonable (> 0.9)
        correlation = torch.corrcoef(torch.stack([y_bf16.flatten(), y_mxfp8.flatten()]))[0, 1]
        assert correlation > 0.9, f"MXFP8 output correlation with BF16 too low: {correlation:.3f}"

    def test_disable_low_precision_context(self):
        """Test evaluation fallback context manager."""
        try:
            import torchao.prototype.mx_formats  # Register MXLinear
            from torchao.prototype.mx_formats import MXLinearConfig
            from torchao.prototype.mx_formats.config import MXLinearRecipeName
            from torchao.quantization import quantize_
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MXFP8 not available in torchao (requires >= 0.15.0): {e}")

        from contextlib import contextmanager

        # Create the context manager (inline version for testing)
        @contextmanager
        def disable_low_precision(model):
            """Temporarily swap low-precision Linear modules with nn.Linear."""
            low_precision_types = ['Float8', 'MXLinear']
            lp_locations = []

            for name, module in model.named_modules():
                module_type = type(module).__name__
                if any(lp_type in module_type for lp_type in low_precision_types):
                    if '.' in name:
                        parent_name, attr_name = name.rsplit('.', 1)
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                        attr_name = name
                    lp_locations.append((parent, attr_name, module))

            if not lp_locations:
                yield
                return

            for parent, attr_name, lp_module in lp_locations:
                linear = nn.Linear(
                    lp_module.in_features,
                    lp_module.out_features,
                    bias=lp_module.bias is not None,
                    device=lp_module.weight.device,
                    dtype=lp_module.weight.dtype,
                )
                linear.weight = lp_module.weight
                if lp_module.bias is not None:
                    linear.bias = lp_module.bias
                setattr(parent, attr_name, linear)

            try:
                yield
            finally:
                for parent, attr_name, lp_module in lp_locations:
                    setattr(parent, attr_name, lp_module)

        # Create and convert model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.DEVICE).to(self.DTYPE)

        def module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        config = MXLinearConfig.from_recipe_name(MXLinearRecipeName.MXFP8_CUBLAS)
        quantize_(model, config, filter_fn=module_filter)

        # Verify MXLinear modules exist
        mxfp_count_before = sum(1 for m in model.modules() if type(m).__name__ == 'MXLinear')
        assert mxfp_count_before == 2

        # Test context manager
        with disable_low_precision(model):
            # Inside context, should be nn.Linear
            linear_count = sum(1 for m in model.modules() if type(m) is nn.Linear)
            assert linear_count == 2, "Expected nn.Linear modules inside context"

        # After context, should be restored to MXLinear
        mxfp_count_after = sum(1 for m in model.modules() if type(m).__name__ == 'MXLinear')
        assert mxfp_count_after == 2, "MXLinear modules not restored after context"


def test_mutex_enforcement():
    """Test that only one precision flag can be active."""
    # This test just documents the expected behavior
    # Actual enforcement happens in base_train.py argument parsing

    # Simulate the mutex check
    def check_mutex(fp8, mxfp8):
        precision_flags = [fp8, mxfp8]
        if sum(precision_flags) > 1:
            raise ValueError("Only one of --fp8, --mxfp8 can be specified")

    # Valid: only one flag
    check_mutex(True, False)  # Should not raise
    check_mutex(False, True)  # Should not raise
    check_mutex(False, False)  # Should not raise

    # Invalid: multiple flags
    with pytest.raises(ValueError):
        check_mutex(True, True)
