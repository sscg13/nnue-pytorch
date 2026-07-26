import torch

from model.quantize import QuantizationConfig, QuantizationManager


def _quantmoid4_reference(values: torch.Tensor) -> torch.Tensor:
    integer_values = torch.floor(values * 128.0 + 1e-5).clamp(-127, 127)
    distance = 127 - integer_values.abs()
    lower_half = torch.floor(distance.square() / 256.0)
    return torch.where(integer_values < 0, lower_half, 126 - lower_half) / 128.0


def test_fake_quantmoid4_matches_signed_integer_reference() -> None:
    quantization = QuantizationManager(QuantizationConfig())
    values = torch.tensor(
        [-2.0, -1.0, -0.503, -0.5, -1.0 / 128.0, 0.0, 1.0 / 128.0, 0.5, 1.0, 2.0]
    )

    result = quantization.fake_quantmoid4(values)

    torch.testing.assert_close(result, _quantmoid4_reference(values), atol=0.0, rtol=0.0)


def test_fake_quantmoid4_uses_surrogate_gradient() -> None:
    quantization = QuantizationManager(QuantizationConfig())
    value = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)

    quantization.fake_quantmoid4(value).sum().backward()

    assert torch.all(value.grad > 0)
