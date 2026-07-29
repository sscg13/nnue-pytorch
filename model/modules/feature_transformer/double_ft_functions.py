import torch

from .fused_ft_functions import _HAS_CUPY_KERNELS
from .sparse_linear_functions import SparseLinearFunction


def double_feature_transform(
    us: torch.Tensor,
    them: torch.Tensor,
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    max_ft_activation: float,
    l1_size: int,
    backend: str = "auto",
) -> torch.Tensor:
    # Resolve backend
    cupy_available = _HAS_CUPY_KERNELS
    all_cuda = (
        us.is_cuda
        and them.is_cuda
        and white_indices.is_cuda
        and black_indices.is_cuda
        and weight.is_cuda
        and bias.is_cuda
    )
    cuda_capable = cupy_available and all_cuda

    if backend == "auto":
        impl = "sparse" if cuda_capable else "torch"
    else:
        impl = backend

    if impl == "fused":
        if not cupy_available:
            raise RuntimeError("Fused double FT backend requested, but CuPy kernels are not available.")
        if not all_cuda:
            raise RuntimeError("Fused double FT backend requested, but not all tensors/parameters are on CUDA.")
        impl = "sparse"

    if impl in ("sparse", "torch"):
        if impl == "sparse":
            if not cupy_available:
                raise RuntimeError("Sparse backend requested, but CuPy kernels are not available.")
            if not all_cuda:
                raise RuntimeError("Sparse backend requested, but not all tensors/parameters are on CUDA.")

        assert l1_size % 2 == 0

        wp = SparseLinearFunction.apply(white_indices, weight, bias, backend=impl)
        bp = SparseLinearFunction.apply(black_indices, weight, bias, backend=impl)

        l0_ = (us * torch.cat([wp, bp], dim=1)) + (them * torch.cat([bp, wp], dim=1))
        # do not fake quantize sum of (quantized) weights
        l0_ = torch.clamp(l0_, 0.0, max_ft_activation)

        l0_s = torch.split(l0_, l1_size // 2, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
        l0_ = torch.cat(l0_s1, dim=1)

        return l0_
    else:
        raise ValueError(f"Invalid double FT implementation mode: {backend}")
