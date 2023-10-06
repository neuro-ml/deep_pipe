import numpy as np
import pytest
import torch
from torch import nn
from dpipe.torch import inference_step


class AssertDtype(nn.Conv2d):
    def __init__(self, *args, assert_dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.assert_dtype = assert_dtype

    def forward(self, x):
        assert x.dtype == self.assert_dtype
        return x


torch_dtype_to_np_dtype = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


@pytest.fixture(params=[torch.float16, torch.float32, torch.float64, None], ids=['fp16', 'fp32', 'fp64', None])
def in_dtype(request):
    return request.param


@pytest.fixture(params=[torch.float16, torch.float32, torch.float64, None], ids=['fp16', 'fp32', 'fp64', None])
def out_dtype(request):
    return request.param


@pytest.fixture(params=[False, True])
def amp(request):
    return request.param


def test_inference_step_dtypes(in_dtype, out_dtype, amp):
    x = np.ones((128, 128)).astype(np.float64)
    # in torch module assert `in_dtype` if specified -> fp16 if amp -> `x.dtype` if nothing specified
    net = AssertDtype(1, 1, kernel_size=3, padding=1, assert_dtype=in_dtype or (torch.float16 if amp else torch.float64))

    out = inference_step(x[None, None], architecture=net, in_dtype=in_dtype, out_dtype=out_dtype, amp=amp)

    # `out_dtype` if specified -> `in_dtype` if specified -> fp16 if amp -> `x.dtype` if nothing specified
    check_dtype = torch_dtype_to_np_dtype.get(out_dtype or (in_dtype or (torch.float16 if amp else in_dtype)), x.dtype)
    assert check_dtype == out.dtype
