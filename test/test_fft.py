import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.fft import fft1d, ifft1d, fft2d, ifft2d, fft_plan, autotune_fft_thresholds, rfft1d, irfft1d, rfft2d, irfft2d


def _complex_close(a, b, tol=1e-3):
  return np.max(np.abs(a - b)) < tol


def test_fft1d_matches_numpy():
  n = 8
  x = np.random.randn(n).astype(np.float32)
  xt = Tensor(x)
  out = fft1d(xt).numpy()
  np_out = np.fft.fft(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_fft1d_mixed_radix():
  n = 12
  x = np.random.randn(n).astype(np.float32)
  xt = Tensor(x)
  out = fft1d(xt).numpy()
  np_out = np.fft.fft(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_ifft1d_roundtrip():
  n = 8
  x = np.random.randn(n).astype(np.float32)
  xt = Tensor(x)
  out = ifft1d(fft1d(xt)).numpy()
  out_r = out[..., 0]
  assert np.max(np.abs(out_r - x)) < 1e-3


def test_fft2d_matches_numpy():
  n = 8
  x = np.random.randn(n, n).astype(np.float32)
  xt = Tensor(x)
  out = fft2d(xt).numpy()
  np_out = np.fft.fft2(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_fft2d_mixed_radix():
  n = 6
  x = np.random.randn(n, n).astype(np.float32)
  xt = Tensor(x)
  out = fft2d(xt).numpy()
  np_out = np.fft.fft2(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_ifft2d_roundtrip():
  n = 8
  x = np.random.randn(n, n).astype(np.float32)
  xt = Tensor(x)
  out = ifft2d(fft2d(xt)).numpy()
  out_r = out[..., 0]
  assert np.max(np.abs(out_r - x)) < 1e-3


def test_fft_plan_1d():
  n = 16
  x = np.random.randn(n).astype(np.float32)
  xt = Tensor(x)
  plan = fft_plan(xt, kind="1d")
  out = plan(xt).numpy()
  np_out = np.fft.fft(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_fft_plan_2d():
  n = 8
  x = np.random.randn(n, n).astype(np.float32)
  xt = Tensor(x)
  plan = fft_plan(xt, kind="2d")
  out = plan(xt).numpy()
  np_out = np.fft.fft2(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_fft_autotune_threshold():
  thr = autotune_fft_thresholds(sizes=(16, 32), thresholds=(32, 64), iters=1)
  assert thr in (32, 64)


def test_rfft1d_matches_numpy():
  n = 16
  x = np.random.randn(n).astype(np.float32)
  xt = Tensor(x)
  out = rfft1d(xt).numpy()
  np_out = np.fft.rfft(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_irfft1d_roundtrip():
  n = 16
  x = np.random.randn(n).astype(np.float32)
  xt = Tensor(x)
  out = irfft1d(rfft1d(xt), n=n).numpy()
  assert np.max(np.abs(out - x)) < 1e-3


def test_rfft2d_matches_numpy():
  n = 8
  x = np.random.randn(n, n).astype(np.float32)
  xt = Tensor(x)
  out = rfft2d(xt).numpy()
  np_out = np.fft.rfft2(x)
  out_c = out[..., 0] + 1j * out[..., 1]
  assert _complex_close(out_c, np_out)


def test_irfft2d_roundtrip():
  n = 8
  x = np.random.randn(n, n).astype(np.float32)
  xt = Tensor(x)
  out = irfft2d(rfft2d(xt), n=(n, n)).numpy()
  assert np.max(np.abs(out - x)) < 1e-3
