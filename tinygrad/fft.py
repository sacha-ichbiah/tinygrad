import os, time, math
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.uop import Ops
from tinygrad.helpers import getenv
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import capturing
from tinygrad.engine.jit import TinyJit, JitError


def _as_complex(x: Tensor) -> Tensor:
  if x.ndim >= 1 and x.shape[-1] == 2:
    return x
  return Tensor.stack([x, x.zeros_like()], dim=-1)


def _complex_mul(a: Tensor, b: Tensor) -> Tensor:
  ar, ai = a[..., 0], a[..., 1]
  br, bi = b[..., 0], b[..., 1]
  real = ar * br - ai * bi
  imag = ar * bi + ai * br
  return Tensor.stack([real, imag], dim=-1)


def _complex_add(a: Tensor, b: Tensor) -> Tensor:
  return Tensor.stack([a[..., 0] + b[..., 0], a[..., 1] + b[..., 1]], dim=-1)


def _complex_sub(a: Tensor, b: Tensor) -> Tensor:
  return Tensor.stack([a[..., 0] - b[..., 0], a[..., 1] - b[..., 1]], dim=-1)


def _complex_conj(a: Tensor) -> Tensor:
  return Tensor.stack([a[..., 0], -a[..., 1]], dim=-1)


def _is_power_of_two(n: int) -> bool:
  return n > 0 and (n & (n - 1)) == 0


def _bit_reverse_indices(n: int) -> list[int]:
  if n <= 1:
    return list(range(n))
  bits = int(math.log2(n))
  out = [0] * n
  for i in range(n):
    v = i
    r = 0
    for _ in range(bits):
      r = (r << 1) | (v & 1)
      v >>= 1
    out[i] = r
  return out


def _bit_reverse_tensor(n: int, device: str) -> Tensor:
  radices = [2] * int(math.log2(n))
  return _digit_reverse_tensor(n, radices, device)


def _digit_reverse_tensor(n: int, radices: list[int], device: str) -> Tensor:
  key = (n, tuple(radices), device)
  cached = _digit_reverse_tensor_cache.get(key)
  if cached is not None:
    return cached
  base_key = (n, tuple(radices))
  base = _digit_reverse_cache.get(base_key)
  if base is None:
    base = _digit_reverse_indices(n, radices)
    _digit_reverse_cache[base_key] = base
  cached = Tensor(base, device=device, dtype=dtypes.int32)
  _digit_reverse_tensor_cache[key] = cached
  return cached


def _digit_reverse_view(x: Tensor, n: int, radices: list[int]) -> Tensor:
  if len(radices) <= 1:
    return x
  prefix = x.shape[:-2]
  x = x.reshape(*prefix, *radices, 2)
  nd = len(prefix)
  rev_axes = [nd + i for i in range(len(radices) - 1, -1, -1)]
  perm = list(range(nd)) + rev_axes + [nd + len(radices)]
  x = x.permute(*perm)
  return x.reshape(*prefix, n, 2)


def _rev8_tensor(device: str) -> Tensor:
  cached = _rev8_cache.get(device)
  if cached is not None:
    return cached
  t = Tensor([0, 4, 2, 6, 1, 5, 3, 7], device=device, dtype=dtypes.int32)
  _rev8_cache[device] = t
  return t




_twiddle_cache: dict[tuple[int, int, int, bool, str, object], Tensor] = {}
_dft_cache: dict[tuple[int, bool, str, object], Tensor] = {}
_split_radix_cache: dict[tuple[int, bool, str, object], tuple[Tensor, Tensor]] = {}
_base8_twiddle_cache: dict[tuple[bool, str, object], Tensor] = {}
_fft_plan_cache: dict[tuple[tuple[int, ...], bool, str, object, str], TinyJit] = {}
_radix_plan_cache: dict[int, list[int]] = {}
_factor_plan_cache: dict[int, list[int]] = {}
_fft_threshold_cache: dict[tuple[str, object], int] = {}
_fft_plan_obj_cache: dict[tuple[tuple[int, ...], bool, str, object, str], "FFTPlan"] = {}
_fft_autotuned: set[tuple[str, object]] = set()
_fft_autotune_active: set[tuple[str, object]] = set()
_stage_twiddle_cache: dict[tuple[int, bool, str, object], Tensor] = {}
_stage_twiddle4_cache: dict[tuple[int, bool, str, object], tuple[Tensor, Tensor, Tensor]] = {}
_stage_twiddle_broadcast_cache: dict[tuple[int, bool, str, object, int], Tensor] = {}
_stage_twiddle4_broadcast_cache: dict[tuple[int, bool, str, object, int], tuple[Tensor, Tensor, Tensor]] = {}
_rev8_cache: dict[str, Tensor] = {}
_fft_split_radix_threshold_cache: dict[tuple[str, object], int] = {}
_digit_reverse_cache: dict[tuple[int, tuple[int, ...]], list[int]] = {}
_digit_reverse_tensor_cache: dict[tuple[int, tuple[int, ...], str], Tensor] = {}
_swap_last2_perm_cache: dict[int, tuple[int, ...]] = {}
_axis_perm_cache: dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...] | None]] = {}
_fft3d_axis_order_cache: dict[tuple[tuple[int, ...], bool], tuple[int, int, int]] = {}
_permute_last3_cache: dict[tuple[int, tuple[int, int, int]], tuple[int, ...]] = {}
_fft3d_contig_threshold_cache: dict[tuple[str, object], int] = {}
_fft3d_plan_cache: dict[tuple[tuple[int, ...], str, object],
                        tuple[bool, bool, tuple[int, int, int] | None, tuple[bool, bool, bool] | None, int, tuple[bool, bool]]] = {}
_twiddle128_cache: dict[tuple[bool, str, object], Tensor] = {}


def _twiddle(n: int, r: int, m: int, inverse: bool, device: str, dtype) -> Tensor:
  key = (n, r, m, inverse, device, dtype)
  cached = _twiddle_cache.get(key)
  if cached is not None:
    return cached
  sign = 1.0 if inverse else -1.0
  ar = Tensor.arange(r, device=device, dtype=dtype, requires_grad=False).reshape(r, 1)
  am = Tensor.arange(m, device=device, dtype=dtype, requires_grad=False).reshape(1, m)
  ang = (ar * am) * (sign * 2 * math.pi / n)
  t = Tensor.stack([ang.cos(), ang.sin()], dim=-1).realize()
  _twiddle_cache[key] = t
  return t


def _twiddle_128(inverse: bool, device: str, dtype) -> Tensor:
  key = (inverse, device, dtype)
  cached = _twiddle128_cache.get(key)
  if cached is not None:
    return cached
  sign = 1.0 if inverse else -1.0
  ar = Tensor.arange(128, device=device, dtype=dtype, requires_grad=False)
  ang = ar * (sign * 2 * math.pi / 128.0)
  t = Tensor.stack([ang.cos(), ang.sin()], dim=-1).realize()
  _twiddle128_cache[key] = t
  return t


def _dft_matrix(r: int, inverse: bool, device: str, dtype) -> Tensor:
  key = (r, inverse, device, dtype)
  cached = _dft_cache.get(key)
  if cached is not None:
    return cached
  sign = 1.0 if inverse else -1.0
  ar = Tensor.arange(r, device=device, dtype=dtype, requires_grad=False).reshape(r, 1)
  am = Tensor.arange(r, device=device, dtype=dtype, requires_grad=False).reshape(1, r)
  ang = (ar * am) * (sign * 2 * math.pi / r)
  t = Tensor.stack([ang.cos(), ang.sin()], dim=-1).realize()
  _dft_cache[key] = t
  return t


def _dft_small(x: Tensor, r: int, inverse: bool, scale: bool = True) -> Tensor:
  permuted = False
  if x.shape[-2] != r:
    if x.ndim < 3 or x.shape[-3] != r:
      raise ValueError("dft_small expects r on the last or third-to-last axis")
    axis = x.ndim - 3
    perm = list(range(x.ndim))
    perm.pop(axis)
    perm.insert(x.ndim - 2, axis)
    x = x.permute(*perm)
    inv_perm = [0] * x.ndim
    for i, p in enumerate(perm):
      inv_perm[p] = i
    permuted = True
  W = _dft_matrix(r, inverse, x.device, x.dtype)
  W = W.reshape((1,) * (x.ndim - 2) + (r, r, 2))
  x_exp = x.unsqueeze(-3)
  prod = _complex_mul(x_exp, W)
  out = prod.sum(axis=-2)
  if inverse and scale:
    scale = 1.0 / r
    out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
  if permuted:
    out = out.permute(*inv_perm)
  return out


def _factor_radix(n: int) -> int:
  for r in (2, 3, 5):
    if n % r == 0:
      return r
  return n


def _factor_list(n: int) -> list[int]:
  cached = _factor_plan_cache.get(n)
  if cached is not None:
    return cached
  out = []
  n0 = n
  while n > 1:
    r = _factor_radix(n)
    out.append(r)
    n //= r
  _factor_plan_cache[n0] = out
  return out


def _radix_plan_pow2(n: int) -> list[int]:
  cached = _radix_plan_cache.get(n)
  if cached is not None:
    return cached
  n0 = n
  rads: list[int] = []
  while n % 8 == 0:
    rads.append(8)
    n //= 8
  while n % 4 == 0:
    rads.append(4)
    n //= 4
  while n % 2 == 0:
    rads.append(2)
    n //= 2
  _radix_plan_cache[n0] = rads
  return rads


def _digit_reverse_indices(n: int, radices: list[int]) -> list[int]:
  if n <= 1:
    return list(range(n))
  if all(r == 2 for r in radices):
    bits = int(math.log2(n))
    out = [0] * n
    for i in range(n):
      v = i
      r = 0
      for _ in range(bits):
        r = (r << 1) | (v & 1)
        v >>= 1
      out[i] = r
    return out
  rem = list(range(n))
  digits: list[list[int]] = []
  for r in radices:
    digits.append([v % r for v in rem])
    rem = [v // r for v in rem]
  rev = [0] * n
  base = 1
  for d, r in zip(digits[::-1], radices[::-1]):
    for i in range(n):
      rev[i] += d[i] * base
    base *= r
  return rev


def _fft1d_recursive(x: Tensor, n: int, inverse: bool) -> Tensor:
  if n == 1:
    return x
  r = _factor_radix(n)
  if r == n:
    return _dft_small(x, n, inverse)
  m = n // r
  prefix = x.shape[:-2]
  x = x.reshape(*prefix, m, r, 2)
  x = x.permute(*range(x.ndim - 3), x.ndim - 2, x.ndim - 3, x.ndim - 1)
  x = _fft1d_recursive(x, m, inverse)
  tw = _twiddle(n, r, m, inverse, x.device, x.dtype).reshape((1,) * (x.ndim - 2) + (r, m, 2))
  x = _complex_mul(x, tw)
  x = _dft_small(x, r, inverse)
  x = x.reshape(*prefix, n, 2)
  return x


def _mul_i(z: Tensor, inverse: bool) -> Tensor:
  if inverse:
    return Tensor.stack([-z[..., 1], z[..., 0]], dim=-1)
  return Tensor.stack([z[..., 1], -z[..., 0]], dim=-1)


def _fft_pow2_base4(x: Tensor, inverse: bool) -> Tensor:
  x0, x1, x2, x3 = x[..., 0, :], x[..., 1, :], x[..., 2, :], x[..., 3, :]
  t0 = _complex_add(x0, x2)
  t1 = _complex_sub(x0, x2)
  t2 = _complex_add(x1, x3)
  t3 = _complex_sub(x1, x3)
  y0 = _complex_add(t0, t2)
  y2 = _complex_sub(t0, t2)
  yi = _mul_i(t3, inverse)
  y1 = _complex_add(t1, yi)
  y3 = _complex_sub(t1, yi)
  return Tensor.stack([y0, y1, y2, y3], dim=-2)


def _base8_twiddle(inverse: bool, device: str, dtype) -> Tensor:
  key = (inverse, device, dtype)
  cached = _base8_twiddle_cache.get(key)
  if cached is not None:
    return cached
  sign = 1.0 if inverse else -1.0
  angles = Tensor.arange(4, device=device, dtype=dtype, requires_grad=False) * (sign * 2 * math.pi / 8)
  t = Tensor.stack([angles.cos(), angles.sin()], dim=-1).realize()
  _base8_twiddle_cache[key] = t
  return t


def _fft_pow2_base8(x: Tensor, inverse: bool) -> Tensor:
  even = _fft_pow2_base4(x[..., ::2, :], inverse)
  odd = _fft_pow2_base4(x[..., 1::2, :], inverse)
  tw = _base8_twiddle(inverse, x.device, x.dtype).reshape((1,) * (x.ndim - 2) + (4, 2))
  t = _complex_mul(odd, tw)
  top = _complex_add(even, t)
  bottom = _complex_sub(even, t)
  return Tensor.cat(top, bottom, dim=-2)


def _split_radix_twiddles(n: int, inverse: bool, device: str, dtype) -> tuple[Tensor, Tensor]:
  key = (n, inverse, device, dtype)
  cached = _split_radix_cache.get(key)
  if cached is not None:
    return cached
  quarter = n // 4
  sign = 1.0 if inverse else -1.0
  k = Tensor.arange(quarter, device=device, dtype=dtype, requires_grad=False)
  w1a = k * (sign * 2 * math.pi / n)
  w3a = k * (sign * 6 * math.pi / n)
  W1 = Tensor.stack([w1a.cos(), w1a.sin()], dim=-1).realize()
  W3 = Tensor.stack([w3a.cos(), w3a.sin()], dim=-1).realize()
  _split_radix_cache[key] = (W1, W3)
  return W1, W3


def _fft_pow2_split_radix(x: Tensor, n: int, inverse: bool) -> Tensor:
  if n == 4:
    return _fft_pow2_base4(x, inverse)
  if n == 8:
    return _fft_pow2_base8(x, inverse)
  if n < 8:
    return _dft_small(x, n, inverse, scale=False)
  half = n // 2
  quarter = n // 4
  even = _fft_pow2_split_radix(x[..., ::2, :], half, inverse)
  odd1 = _fft_pow2_split_radix(x[..., 1::4, :], quarter, inverse)
  odd2 = _fft_pow2_split_radix(x[..., 3::4, :], quarter, inverse)

  W1, W3 = _split_radix_twiddles(n, inverse, x.device, x.dtype)
  W1 = W1.reshape((1,) * (x.ndim - 2) + (quarter, 2))
  W3 = W3.reshape((1,) * (x.ndim - 2) + (quarter, 2))

  t1 = _complex_mul(odd1, W1)
  t2 = _complex_mul(odd2, W3)
  t12 = _complex_add(t1, t2)
  tdiff = _complex_sub(t1, t2)

  top = _complex_add(even[..., :quarter, :], t12)
  bottom = _complex_sub(even[..., :quarter, :], t12)
  mid = _complex_sub(even[..., quarter:, :], _mul_i(tdiff, inverse))
  mid2 = _complex_add(even[..., quarter:, :], _mul_i(tdiff, inverse))

  out = Tensor.cat(top, mid2, bottom, mid, dim=-2)
  return out


def _fft_pow2_radix8_plan(x: Tensor, n: int, inverse: bool) -> Tensor:
  radices = _radix_plan_pow2(n)
  if not radices:
    return x

  # DIT: apply digit reversal at INPUT
  x = _digit_reverse_view(x, n, radices)

  prefix = x.shape[:-2]
  m = 1  # current size of each DFT'd group

  for r in radices:
    m_new = m * r
    num_supergroups = n // m_new

    if m == 1:
      # First stage: just do r-point DFT
      x = x.reshape(*prefix, num_supergroups, r, 2)
      x = _dft_small(x, r, inverse, scale=False)
    else:
      # Subsequent stages: reshape to access r groups of m
      x = x.reshape(*prefix, num_supergroups, r, m, 2)

      # Apply twiddles: W_{m_new}^{k*j} - uses m_new, not n!
      tw = _twiddle(m_new, r, m, inverse, x.device, x.dtype)
      tw = tw.reshape((1,) * (x.ndim - 3) + (r, m, 2))
      x = _complex_mul(x, tw)

      # Transpose to (supergroups, m, r, 2) for DFT
      x = x.permute(*range(x.ndim - 4), x.ndim - 4, x.ndim - 2, x.ndim - 3, x.ndim - 1)
      x = _dft_small(x, r, inverse, scale=False)
      # Transpose back
      x = x.permute(*range(x.ndim - 4), x.ndim - 4, x.ndim - 2, x.ndim - 3, x.ndim - 1)

    x = x.reshape(*prefix, n, 2)
    m = m_new

  if inverse:
    scale = 1.0 / n
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x


def _fft_pow2_iterative_radix8(x: Tensor, n: int, inverse: bool) -> Tensor:
  threshold = _get_radix8_threshold(x.device, x.dtype)
  use_radix8_base = n >= 8 and n >= threshold
  if use_radix8_base:
    bits = int(math.log2(n))
    radices = [8] + [2] * (bits - 3)
    x = _digit_reverse_view(x, n, radices)
    prefix0 = x.shape[:-2]
    x = x.reshape(*prefix0, -1, 8, 2)
    x = _fft_pow2_base8(x, inverse)
    x = x.reshape(*prefix0, n, 2)
    m = 16
  else:
    bits = int(math.log2(n))
    if n >= 4:
      radices = [4] + [2] * (bits - 2)
      x = _digit_reverse_view(x, n, radices)
      prefix0 = x.shape[:-2]
      x = x.reshape(*prefix0, -1, 4, 2)
      x = _fft_pow2_base4(x, inverse)
      x = x.reshape(*prefix0, n, 2)
      m = 8
    else:
      radices = [2] * bits
      x = _digit_reverse_view(x, n, radices)
      prefix0 = x.shape[:-2]
      m = 2

  ndim = len(prefix0) + 2
  while m <= n:
    # The radix-4 shortcut is only compatible with the radix-4 base initialization path.
    # For radix-8 base, the data layout after digit reversal with mixed radices [8, 2, 2, ...]
    # is incompatible with the radix-4 butterfly combining logic.
    if not use_radix8_base and m >= 4 and m * 4 <= n and (len(prefix0) <= 2 or n >= 1024):
      m4 = m * 4
      quarter = m4 // 4
      x = x.reshape(*prefix0, -1, m4, 2)
      x0 = x[..., :quarter, :]
      x1 = x[..., quarter:2 * quarter, :]
      x2 = x[..., 2 * quarter:3 * quarter, :]
      x3 = x[..., 3 * quarter:, :]
      tw1, tw2, tw3 = _stage_twiddle4_broadcast(m4, inverse, x.device, x.dtype, ndim)
      t1 = _complex_mul(x1, tw1)
      t2 = _complex_mul(x2, tw2)
      t3 = _complex_mul(x3, tw3)
      y = _fft_pow2_base4(Tensor.stack([x0, t1, t2, t3], dim=-2), inverse)
      y = y.permute(*range(y.ndim - 3), y.ndim - 2, y.ndim - 3, y.ndim - 1)
      x = y.reshape(*prefix0, n, 2)
      m = m4
      continue
    half = m // 2
    x = x.reshape(*prefix0, -1, m, 2)
    even = x[..., :half, :]
    odd = x[..., half:, :]
    if m == 2:
      x = Tensor.cat(_complex_add(even, odd), _complex_sub(even, odd), dim=-2)
    else:
      tw = _stage_twiddle_broadcast(m, inverse, x.device, x.dtype, ndim)
      t = _complex_mul(odd, tw)
      x = Tensor.cat(_complex_add(even, t), _complex_sub(even, t), dim=-2)
    x = x.reshape(*prefix0, n, 2)
    m *= 2

  if inverse:
    scale = 1.0 / n
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x


def _fft_mixed_radix_plan(x: Tensor, n: int, inverse: bool) -> Tensor:
  radices = _factor_list(n)
  if radices.count(2) >= 2:
    compressed: list[int] = []
    i = 0
    while i < len(radices):
      if i + 1 < len(radices) and radices[i] == 2 and radices[i + 1] == 2:
        compressed.append(4)
        i += 2
      else:
        compressed.append(radices[i])
        i += 1
    radices = compressed
  if len(radices) <= 1:
    return _dft_small(x, n, inverse)
  m = 1
  prefix = x.shape[:-2]
  for r in radices:
    m_new = m * r
    blocks = n // m_new
    x = x.reshape(*prefix, blocks, m, r, 2)
    x = x.permute(*range(x.ndim - 4), x.ndim - 4, x.ndim - 2, x.ndim - 3, x.ndim - 1)
    tw = _twiddle(n, r, m, inverse, x.device, x.dtype).reshape((1,) * (x.ndim - 3) + (r, m, 2))
    x = _complex_mul(x, tw)
    x = _dft_small(x, r, inverse, scale=False)
    x = x.reshape(*prefix, blocks, m_new, 2)
    m = m_new
  x = x.reshape(*prefix, n, 2)
  if inverse:
    scale = 1.0 / n
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x


def _get_radix8_threshold(device: str, dtype) -> int:
  env = os.getenv("TINYGRAD_FFT_RADIX8_THRESHOLD")
  if env is not None:
    return int(env)
  cached = _fft_threshold_cache.get((device, dtype))
  if cached is not None:
    return cached
  return 8


def _get_split_radix_threshold(device: str, dtype) -> int:
  env = os.getenv("TINYGRAD_FFT_SPLIT_RADIX_THRESHOLD")
  if env is not None:
    return int(env)
  cached = _fft_split_radix_threshold_cache.get((device, dtype))
  if cached is not None:
    return cached
  return 32


def _get_fft3d_contig_threshold(device: str, dtype) -> int:
  env = os.getenv("TINYGRAD_FFT_CONTIGUOUS_3D_THRESHOLD")
  if env is not None:
    return int(env)
  cached = _fft3d_contig_threshold_cache.get((device, dtype))
  if cached is not None:
    return cached
  return 4096


def _maybe_contiguous(x: Tensor, threshold: int) -> Tensor:
  if threshold <= 0 or x.device != "CPU":
    return x
  try:
    numel = int(x.numel())
  except Exception:
    return x
  if numel >= threshold and not x.uop.is_contiguous():
    return x.contiguous()
  return x


def _ensure_autotuned(device: str, dtype):
  if (device, dtype) in _fft_autotuned:
    return
  if (device, dtype) in _fft_autotune_active:
    return
  env = os.getenv("TINYGRAD_FFT_AUTOTUNE")
  if env is not None and getenv("TINYGRAD_FFT_AUTOTUNE", 0):
    autotune_fft_thresholds(device=device, dtype=dtype)
    autotune_split_radix_thresholds(device=device, dtype=dtype)
  _fft_autotuned.add((device, dtype))


def _stage_twiddle(m: int, inverse: bool, device: str, dtype) -> Tensor:
  key = (m, inverse, device, dtype)
  cached = _stage_twiddle_cache.get(key)
  if cached is not None:
    return cached
  sign = 1.0 if inverse else -1.0
  half = m // 2
  angles = Tensor.arange(half, device=device, dtype=dtype, requires_grad=False) * (sign * 2 * math.pi / m)
  tw = Tensor.stack([angles.cos(), angles.sin()], dim=-1).realize()
  _stage_twiddle_cache[key] = tw
  return tw


def _stage_twiddle_broadcast(m: int, inverse: bool, device: str, dtype, ndim: int) -> Tensor:
  key = (m, inverse, device, dtype, ndim)
  cached = _stage_twiddle_broadcast_cache.get(key)
  if cached is not None:
    return cached
  half = m // 2
  tw = _stage_twiddle(m, inverse, device, dtype)
  tw = tw.reshape((1,) * (ndim - 2) + (half, 2))
  _stage_twiddle_broadcast_cache[key] = tw
  return tw


def _stage_twiddle4(m: int, inverse: bool, device: str, dtype) -> tuple[Tensor, Tensor, Tensor]:
  key = (m, inverse, device, dtype)
  cached = _stage_twiddle4_cache.get(key)
  if cached is not None:
    return cached
  sign = 1.0 if inverse else -1.0
  quarter = m // 4
  angles = Tensor.arange(quarter, device=device, dtype=dtype, requires_grad=False) * (sign * 2 * math.pi / m)
  tw1 = Tensor.stack([angles.cos(), angles.sin()], dim=-1)
  tw2 = _complex_mul(tw1, tw1)
  tw3 = _complex_mul(tw2, tw1)
  tw1, tw2, tw3 = tw1.realize(), tw2.realize(), tw3.realize()
  _stage_twiddle4_cache[key] = (tw1, tw2, tw3)
  return tw1, tw2, tw3


def _stage_twiddle4_broadcast(m: int, inverse: bool, device: str, dtype, ndim: int) -> tuple[Tensor, Tensor, Tensor]:
  key = (m, inverse, device, dtype, ndim)
  cached = _stage_twiddle4_broadcast_cache.get(key)
  if cached is not None:
    return cached
  quarter = m // 4
  tw1, tw2, tw3 = _stage_twiddle4(m, inverse, device, dtype)
  tw1 = tw1.reshape((1,) * (ndim - 2) + (quarter, 2))
  tw2 = tw2.reshape((1,) * (ndim - 2) + (quarter, 2))
  tw3 = tw3.reshape((1,) * (ndim - 2) + (quarter, 2))
  _stage_twiddle4_broadcast_cache[key] = (tw1, tw2, tw3)
  return tw1, tw2, tw3


def _fft_pow2_iterative_radix4(x: Tensor, n: int, inverse: bool) -> Tensor:
  bits = int(math.log2(n))
  radices = [4] * (bits // 2)
  x = _digit_reverse_view(x, n, radices)
  prefix0 = x.shape[:-2]
  ndim = len(prefix0) + 2
  m = 4
  while m <= n:
    quarter = m // 4
    x = x.reshape(*prefix0, -1, m, 2)
    prefix = x.shape[:-2]
    x0 = x[..., :quarter, :]
    x1 = x[..., quarter:2 * quarter, :]
    x2 = x[..., 2 * quarter:3 * quarter, :]
    x3 = x[..., 3 * quarter:, :]
    if m == 4:
      t1, t2, t3 = x1, x2, x3
    else:
      tw1, tw2, tw3 = _stage_twiddle4_broadcast(m, inverse, x.device, x.dtype, ndim)
      t1 = _complex_mul(x1, tw1)
      t2 = _complex_mul(x2, tw2)
      t3 = _complex_mul(x3, tw3)
    y = _fft_pow2_base4(Tensor.stack([x0, t1, t2, t3], dim=-2), inverse)
    y = y.permute(*range(y.ndim - 3), y.ndim - 2, y.ndim - 3, y.ndim - 1)
    x = y.reshape(*prefix, m, 2)
    m *= 4
  x = x.reshape(*prefix0, n, 2)
  if inverse:
    scale = 1.0 / n
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x


def _fft_pow2_unrolled_16(x: Tensor, inverse: bool) -> Tensor:
  x = _digit_reverse_view(x, 16, [8, 2])
  prefix = x.shape[:-2]
  x = x.reshape(*prefix, -1, 8, 2)
  x = _fft_pow2_base8(x, inverse)
  x = x.reshape(*prefix, 16, 2)
  x = x.reshape(*prefix, -1, 16, 2)
  even = x[..., :8, :]
  odd = x[..., 8:, :]
  tw = _stage_twiddle_broadcast(16, inverse, x.device, x.dtype, len(prefix) + 2)
  t = _complex_mul(odd, tw)
  x = Tensor.cat(_complex_add(even, t), _complex_sub(even, t), dim=-2)
  x = x.reshape(*prefix, 16, 2)
  if inverse:
    scale = 1.0 / 16
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x


def _fft_base_on_axis_minus3(x: Tensor, r: int, inverse: bool) -> Tensor:
  if r not in (4, 8):
    raise ValueError("base FFT only supports r=4 or r=8")
  perm = list(range(x.ndim))
  perm[-3], perm[-2] = perm[-2], perm[-3]
  x = x.permute(*perm)
  x = _fft_pow2_base4(x, inverse) if r == 4 else _fft_pow2_base8(x, inverse)
  inv = [0] * x.ndim
  for i, p in enumerate(perm):
    inv[p] = i
  return x.permute(*inv)


def _fft_pow2_unrolled_128(x: Tensor, inverse: bool) -> Tensor:
  n = 128
  prefix = x.shape[:-2]
  m = 1
  # stage r=8
  r = 8
  m_new = m * r
  blocks = n // m_new
  x = x.reshape(*prefix, blocks, m, r, 2)
  x = x.permute(*range(x.ndim - 4), x.ndim - 4, x.ndim - 2, x.ndim - 3, x.ndim - 1)
  tw = _twiddle(n, r, m, inverse, x.device, x.dtype).reshape((1,) * (x.ndim - 3) + (r, m, 2))
  x = _complex_mul(x, tw)
  x = _fft_base_on_axis_minus3(x, r, inverse)
  x = x.reshape(*prefix, blocks, m_new, 2)
  m = m_new
  # stage r=4
  r = 4
  m_new = m * r
  blocks = n // m_new
  x = x.reshape(*prefix, blocks, m, r, 2)
  x = x.permute(*range(x.ndim - 4), x.ndim - 4, x.ndim - 2, x.ndim - 3, x.ndim - 1)
  tw = _twiddle(n, r, m, inverse, x.device, x.dtype).reshape((1,) * (x.ndim - 3) + (r, m, 2))
  x = _complex_mul(x, tw)
  x = _fft_base_on_axis_minus3(x, r, inverse)
  x = x.reshape(*prefix, blocks, m_new, 2)
  m = m_new
  # stage r=4
  r = 4
  m_new = m * r
  blocks = n // m_new
  x = x.reshape(*prefix, blocks, m, r, 2)
  x = x.permute(*range(x.ndim - 4), x.ndim - 4, x.ndim - 2, x.ndim - 3, x.ndim - 1)
  tw = _twiddle(n, r, m, inverse, x.device, x.dtype).reshape((1,) * (x.ndim - 3) + (r, m, 2))
  x = _complex_mul(x, tw)
  x = _fft_base_on_axis_minus3(x, r, inverse)
  x = x.reshape(*prefix, blocks, m_new, 2)
  if inverse:
    scale = 1.0 / n
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x.reshape(*prefix, n, 2)


def _fft1d_pow2_fast(x: Tensor, n: int, inverse: bool) -> Tensor:
  if n <= 1:
    return x
  if n == 2:
    x0, x1 = x[..., 0, :], x[..., 1, :]
    out = Tensor.stack([_complex_add(x0, x1), _complex_sub(x0, x1)], dim=-2)
    if inverse:
      scale = 0.5
      out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
    return out
  if n == 4:
    out = _fft_pow2_base4(x, inverse)
    if inverse:
      scale = 0.25
      out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
    return out
  if n == 8:
    out = _fft_pow2_base8(x, inverse)
    if inverse:
      scale = 0.125
      out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
    return out
  if n == 16:
    return _fft_pow2_unrolled_16(x, inverse)
  if n == 128:
    if getenv("TINYGRAD_FFT_128_FUSED", 0):
      return _fft_pow2_unrolled_128(x, inverse)
  if n >= 128:
    return _fft_pow2_radix8_plan(x, n, inverse)
  return _fft_pow2_iterative_radix8(x, n, inverse)


def _fft1d_impl(x: Tensor, inverse: bool = False) -> Tensor:
  n = int(x.shape[-2])
  if n <= 1:
    return x
  if _is_power_of_two(n):
    if n == 2:
      x0, x1 = x[..., 0, :], x[..., 1, :]
      out = Tensor.stack([_complex_add(x0, x1), _complex_sub(x0, x1)], dim=-2)
      if inverse:
        scale = 0.5
        out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
      return out
    if n == 4:
      out = _fft_pow2_base4(x, inverse)
      if inverse:
        scale = 0.25
        out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
      return out
    if n == 8:
      out = _fft_pow2_base8(x, inverse)
      if inverse:
        scale = 0.125
        out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
      return out
    bits = int(math.log2(n))
    split_thr = _get_split_radix_threshold(x.device, x.dtype)
    if split_thr and n <= split_thr:
      out = _fft_pow2_split_radix(x, n, inverse)
      if inverse:
        scale = 1.0 / n
        out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
      return out
    if n % 4 == 0 and (bits % 2 == 0) and (bits <= 6 or bits >= 10):
      return _fft_pow2_iterative_radix4(x, n, inverse)
    return _fft_pow2_iterative_radix8(x, n, inverse)
  if not _is_power_of_two(n):
    if n >= getenv("TINYGRAD_FFT_MIXEDRADIX_THRESHOLD", 1 << 30):
      return _fft_mixed_radix_plan(x, n, inverse)
    return _fft1d_recursive(x, n, inverse)
  if n == 1:
    return x

  rev = _bit_reverse_indices(n)
  idx = Tensor(rev, device=x.device, dtype=dtypes.int32).reshape((1,) * (x.ndim - 2) + (n, 1)).expand(*x.shape)
  x = x.gather(-2, idx)

  sign = 1.0 if inverse else -1.0
  m = 2
  while m <= n:
    half = m // 2
    angles = Tensor.arange(half, device=x.device, dtype=x.dtype, requires_grad=False) * (sign * 2 * math.pi / m)
    tw = Tensor.stack([angles.cos(), angles.sin()], dim=-1).reshape((1,) * (x.ndim - 2) + (half, 2))

    prefix = x.shape[:-2]
    x = x.reshape(*prefix, -1, m, 2)
    even = x[..., :half, :]
    odd = x[..., half:, :]
    t = _complex_mul(odd, tw)
    x = Tensor.cat(_complex_add(even, t), _complex_sub(even, t), dim=-2)
    x = x.reshape(*prefix, n, 2)
    m *= 2

  if inverse:
    scale = 1.0 / n
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x


def _fft2d_impl(x: Tensor, inverse: bool = False) -> Tensor:
  x = _as_complex(x)
  if x.ndim < 2:
    raise ValueError("fft2d requires at least 2D input")
  def _swap_last2(x: Tensor) -> Tensor:
    if x.ndim == 3:
      return x.permute(1, 0, 2)
    perm = _swap_last2_perm_cache.get(x.ndim)
    if perm is None:
      perm = tuple(list(range(x.ndim - 3)) + [x.ndim - 2, x.ndim - 3, x.ndim - 1])
      _swap_last2_perm_cache[x.ndim] = perm
    return x.permute(*perm)
  swap_first = x.shape[-2] < x.shape[-3]
  if swap_first:
    x = _swap_last2(x)
  x = _fft1d_impl(x, inverse=inverse)
  x = _swap_last2(x)
  x = _fft1d_impl(x, inverse=inverse)
  if not swap_first:
    x = _swap_last2(x)
  return x


def _fft2d_pow2_fast(x: Tensor, inverse: bool = False) -> Tensor:
  x = _as_complex(x)
  if x.ndim < 2:
    raise ValueError("fft2d requires at least 2D input")
  def _swap_last2(x: Tensor) -> Tensor:
    if x.ndim == 3:
      return x.permute(1, 0, 2)
    perm = _swap_last2_perm_cache.get(x.ndim)
    if perm is None:
      perm = tuple(list(range(x.ndim - 3)) + [x.ndim - 2, x.ndim - 3, x.ndim - 1])
      _swap_last2_perm_cache[x.ndim] = perm
    return x.permute(*perm)
  swap_first = x.shape[-2] < x.shape[-3]
  if swap_first:
    x = _swap_last2(x)
  n0, n1 = int(x.shape[-3]), int(x.shape[-2])
  x = _fft1d_pow2_fast(x, n1, inverse)
  x = _swap_last2(x)
  x = _fft1d_pow2_fast(x, n0, inverse)
  if not swap_first:
    x = _swap_last2(x)
  return x


def _rfft1d_impl(x: Tensor) -> Tensor:
  n = int(x.shape[-1])
  out = fft1d(_as_complex(x))
  return out[..., : n // 2 + 1, :]


def _permute_axis_to_last2(x: Tensor, axis: int) -> tuple[Tensor, tuple[int, ...] | None]:
  if axis < 0:
    axis += x.ndim
  last = x.ndim - 1
  if axis == last - 1:
    return x, None
  key = (x.ndim, axis)
  cached = _axis_perm_cache.get(key)
  if cached is None:
    dims = list(range(x.ndim - 1))
    dims.remove(axis)
    dims.append(axis)
    perm = tuple(dims + [last])
    inv = [0] * x.ndim
    for i, p in enumerate(perm):
      inv[p] = i
    cached = (perm, tuple(inv))
    _axis_perm_cache[key] = cached
  perm, inv = cached
  return x.permute(*perm), inv


def _fft_axis(x: Tensor, axis: int, inverse: bool) -> Tensor:
  x, inv = _permute_axis_to_last2(x, axis)
  x = _fft1d_impl(x, inverse=inverse)
  return x if inv is None else x.permute(*inv)


def _fft_axis_fft1d(x: Tensor, axis: int, inverse: bool) -> Tensor:
  x, inv = _permute_axis_to_last2(x, axis)
  x = fft1d(x, inverse=inverse)
  return x if inv is None else x.permute(*inv)


def _rfft2d_impl(x: Tensor) -> Tensor:
  x = _rfft1d_impl(x)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    perm = _swap_last2_perm_cache.get(x.ndim)
    if perm is None:
      perm = tuple(list(range(x.ndim - 3)) + [x.ndim - 2, x.ndim - 3, x.ndim - 1])
      _swap_last2_perm_cache[x.ndim] = perm
    x = x.permute(*perm)
  x = fft1d(x)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    perm = _swap_last2_perm_cache.get(x.ndim)
    if perm is None:
      perm = tuple(list(range(x.ndim - 3)) + [x.ndim - 2, x.ndim - 3, x.ndim - 1])
      _swap_last2_perm_cache[x.ndim] = perm
    x = x.permute(*perm)
  return x


def _fft3d_fused_8(x: Tensor, inverse: bool) -> Tensor:
  prefix = x.shape[:-4]
  W = _dft_matrix(8, inverse, x.device, x.dtype)
  x6 = x.reshape(*prefix, 1, 1, 1, 8, 8, 8, 2)
  Wx = W.reshape((1,) * len(prefix) + (8, 1, 1, 8, 1, 1, 2))
  Wy = W.reshape((1,) * len(prefix) + (1, 8, 1, 1, 8, 1, 2))
  Wz = W.reshape((1,) * len(prefix) + (1, 1, 8, 1, 1, 8, 2))
  t = _complex_mul(x6, Wx)
  t = _complex_mul(t, Wy)
  t = _complex_mul(t, Wz)
  out = t.sum(axis=-4).sum(axis=-3).sum(axis=-2)
  if inverse:
    scale = 1.0 / 512
    out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
  return out


def _uop_fft1d_128_stage_axis(x_ptr: UOp, out_ptr: UOp, tw_ptr: UOp, axis: int, inverse: bool, stage: int) -> UOp:
  if axis not in (0, 1, 2):
    raise ValueError("axis must be 0,1,2")
  if stage not in range(7):
    raise ValueError("stage must be 0..6")
  sizes = (128, 128, 128)
  ranges = [UOp.range(s, i) for i, s in enumerate(sizes)]
  idx = [ranges[0], ranges[1], ranges[2]]
  idx_axis = idx[axis]
  shift = stage + 1
  m = 1 << shift
  half = m >> 1
  stride_shift = 6 - stage
  shift_idx = UOp.const(dtypes.index, shift)
  m_mask = UOp.const(dtypes.index, m - 1)
  half_mask = UOp.const(dtypes.index, half - 1)
  half_idx = UOp.const(dtypes.index, half)
  stride_shift_idx = UOp.const(dtypes.index, stride_shift)
  block = idx_axis >> shift_idx
  pos = idx_axis & m_mask
  p = pos & half_mask
  even_idx = (block << shift_idx) + p
  odd_idx = even_idx + half_idx
  idx_even = idx.copy()
  idx_even[axis] = even_idx
  idx_odd = idx.copy()
  idx_odd[axis] = odd_idx
  v_even = x_ptr.vindex(*idx_even)
  v_odd = x_ptr.vindex(*idx_odd)
  even_r, even_i = v_even.gep(0), v_even.gep(1)
  odd_r, odd_i = v_odd.gep(0), v_odd.gep(1)
  tw_idx = p << stride_shift_idx
  w = tw_ptr.vindex(tw_idx)
  w_r, w_i = w.gep(0), w.gep(1)
  t_r = odd_r * w_r - odd_i * w_i
  t_i = odd_r * w_i + odd_i * w_r
  cond = pos < half_idx
  out_r = cond.where(even_r + t_r, even_r - t_r)
  out_i = cond.where(even_i + t_i, even_i - t_i)
  if inverse and stage == 6:
    scale = UOp.const(dtypes.float, 1.0 / 128.0)
    out_r = out_r * scale
    out_i = out_i * scale
  out = UOp.vectorize(out_r, out_i)
  store = out_ptr.index(*idx, ptr=True).store(out)
  return UOp.sink(store).end(*ranges)


def _uop_fft1d_128_stage4_axis(x_ptr: UOp, out_ptr: UOp, tw_ptr: UOp, axis: int, inverse: bool, stage4: int) -> UOp:
  if axis not in (0, 1, 2):
    raise ValueError("axis must be 0,1,2")
  if stage4 not in range(3):
    raise ValueError("stage4 must be 0..2")
  if axis == 2 and stage4 == 0:
    r0 = UOp.range(128, 0)
    r1 = UOp.range(128, 1)
    rblk = UOp.range(32, 2)
    ranges = [r0, r1, rblk]
    base = rblk << UOp.const(dtypes.index, 2)
    ptr = x_ptr.index(r0, r1, base, ptr=True)
    vec_dtype = x_ptr.dtype.base.vec(8)
    vec_ptr = ptr.cast(vec_dtype.ptr(size=ptr.dtype.size, addrspace=ptr.dtype.addrspace))
    v = vec_ptr.load()
    x0r, x0i = v.gep(0), v.gep(1)
    x1r, x1i = v.gep(2), v.gep(3)
    x2r, x2i = v.gep(4), v.gep(5)
    x3r, x3i = v.gep(6), v.gep(7)
    t0r, t0i = x0r + x2r, x0i + x2i
    t1r, t1i = x0r - x2r, x0i - x2i
    t2r, t2i = x1r + x3r, x1i + x3i
    u3r, u3i = x1r - x3r, x1i - x3i
    if inverse:
      t3r, t3i = -u3i, u3r
    else:
      t3r, t3i = u3i, -u3r
    y0r, y0i = t0r + t2r, t0i + t2i
    y1r, y1i = t1r + t3r, t1i + t3i
    y2r, y2i = t0r - t2r, t0i - t2i
    y3r, y3i = t1r - t3r, t1i - t3i
    out_vec = UOp.vectorize(y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i)
    out_ptr_vec = out_ptr.index(r0, r1, base, ptr=True).cast(vec_dtype.ptr(size=ptr.dtype.size, addrspace=ptr.dtype.addrspace))
    store = out_ptr_vec.store(out_vec)
    return UOp.sink(store).end(*ranges)
  if axis == 2 and stage4 == 1:
    r0 = UOp.range(128, 0)
    r1 = UOp.range(128, 1)
    rblk = UOp.range(8, 2)
    rp = UOp.range(4, 3)
    ranges = [r0, r1, rblk, rp]
    base = (rblk << UOp.const(dtypes.index, 4)) + rp
    ptr = x_ptr.index(r0, r1, base, ptr=True)
    vec_dtype = x_ptr.dtype.base.vec(8)
    vec_ptr = ptr.cast(vec_dtype.ptr(size=ptr.dtype.size, addrspace=ptr.dtype.addrspace))
    v = vec_ptr.load()
    x0r, x0i = v.gep(0), v.gep(1)
    x1r, x1i = v.gep(2), v.gep(3)
    x2r, x2i = v.gep(4), v.gep(5)
    x3r, x3i = v.gep(6), v.gep(7)
    t0r, t0i = x0r + x2r, x0i + x2i
    t1r, t1i = x0r - x2r, x0i - x2i
    t2r, t2i = x1r + x3r, x1i + x3i
    u3r, u3i = x1r - x3r, x1i - x3i
    if inverse:
      t3r, t3i = -u3i, u3r
    else:
      t3r, t3i = u3i, -u3r
    y0r, y0i = t0r + t2r, t0i + t2i
    y1r, y1i = t1r + t3r, t1i + t3i
    y2r, y2i = t0r - t2r, t0i - t2i
    y3r, y3i = t1r - t3r, t1i - t3i
    tw_idx = rp << UOp.const(dtypes.index, 3)
    tw_idx2 = tw_idx << UOp.const(dtypes.index, 1)
    tw_idx3 = tw_idx2 + tw_idx
    w1 = tw_ptr.vindex(tw_idx)
    w2 = tw_ptr.vindex(tw_idx2)
    w3 = tw_ptr.vindex(tw_idx3)
    w1r, w1i = w1.gep(0), w1.gep(1)
    w2r, w2i = w2.gep(0), w2.gep(1)
    w3r, w3i = w3.gep(0), w3.gep(1)
    z1r = y1r * w1r - y1i * w1i
    z1i = y1r * w1i + y1i * w1r
    z2r = y2r * w2r - y2i * w2i
    z2i = y2r * w2i + y2i * w2r
    z3r = y3r * w3r - y3i * w3i
    z3i = y3r * w3i + y3i * w3r
    out_vec = UOp.vectorize(y0r, y0i, z1r, z1i, z2r, z2i, z3r, z3i)
    out_ptr_vec = out_ptr.index(r0, r1, base, ptr=True).cast(vec_dtype.ptr(size=ptr.dtype.size, addrspace=ptr.dtype.addrspace))
    store = out_ptr_vec.store(out_vec)
    return UOp.sink(store).end(*ranges)
  if axis == 2 and stage4 == 2:
    r0 = UOp.range(128, 0)
    r1 = UOp.range(128, 1)
    rblk = UOp.range(2, 2)
    rp = UOp.range(16, 3)
    ranges = [r0, r1, rblk, rp]
    base = (rblk << UOp.const(dtypes.index, 6)) + rp
    ptr = x_ptr.index(r0, r1, base, ptr=True)
    vec_dtype = x_ptr.dtype.base.vec(8)
    vec_ptr = ptr.cast(vec_dtype.ptr(size=ptr.dtype.size, addrspace=ptr.dtype.addrspace))
    v = vec_ptr.load()
    x0r, x0i = v.gep(0), v.gep(1)
    x1r, x1i = v.gep(2), v.gep(3)
    x2r, x2i = v.gep(4), v.gep(5)
    x3r, x3i = v.gep(6), v.gep(7)
    t0r, t0i = x0r + x2r, x0i + x2i
    t1r, t1i = x0r - x2r, x0i - x2i
    t2r, t2i = x1r + x3r, x1i + x3i
    u3r, u3i = x1r - x3r, x1i - x3i
    if inverse:
      t3r, t3i = -u3i, u3r
    else:
      t3r, t3i = u3i, -u3r
    y0r, y0i = t0r + t2r, t0i + t2i
    y1r, y1i = t1r + t3r, t1i + t3i
    y2r, y2i = t0r - t2r, t0i - t2i
    y3r, y3i = t1r - t3r, t1i - t3i
    tw_idx = rp << UOp.const(dtypes.index, 1)
    tw_idx2 = tw_idx << UOp.const(dtypes.index, 1)
    tw_idx3 = tw_idx2 + tw_idx
    w1 = tw_ptr.vindex(tw_idx)
    w2 = tw_ptr.vindex(tw_idx2)
    w3 = tw_ptr.vindex(tw_idx3)
    w1r, w1i = w1.gep(0), w1.gep(1)
    w2r, w2i = w2.gep(0), w2.gep(1)
    w3r, w3i = w3.gep(0), w3.gep(1)
    z1r = y1r * w1r - y1i * w1i
    z1i = y1r * w1i + y1i * w1r
    z2r = y2r * w2r - y2i * w2i
    z2i = y2r * w2i + y2i * w2r
    z3r = y3r * w3r - y3i * w3i
    z3i = y3r * w3i + y3i * w3r
    out_vec = UOp.vectorize(y0r, y0i, z1r, z1i, z2r, z2i, z3r, z3i)
    out_ptr_vec = out_ptr.index(r0, r1, base, ptr=True).cast(vec_dtype.ptr(size=ptr.dtype.size, addrspace=ptr.dtype.addrspace))
    store = out_ptr_vec.store(out_vec)
    return UOp.sink(store).end(*ranges)
  m_shift = 2 * (stage4 + 1)
  quarter_shift = 2 * stage4
  stride_shift = 5 - (2 * stage4)
  m = 1 << m_shift
  quarter = 1 << quarter_shift
  if axis == 0:
    r1 = UOp.range(128, 0)
    r2 = UOp.range(128, 1)
    rblk = UOp.range(128 // m, 2)
    rp = UOp.range(quarter, 3)
    ranges = [r1, r2, rblk, rp]
    idx_base = [None, r1, r2]
  elif axis == 1:
    r0 = UOp.range(128, 0)
    r2 = UOp.range(128, 1)
    rblk = UOp.range(128 // m, 2)
    rp = UOp.range(quarter, 3)
    ranges = [r0, r2, rblk, rp]
    idx_base = [r0, None, r2]
  else:
    r0 = UOp.range(128, 0)
    r1 = UOp.range(128, 1)
    rblk = UOp.range(128 // m, 2)
    rp = UOp.range(quarter, 3)
    ranges = [r0, r1, rblk, rp]
    idx_base = [r0, r1, None]
  base = (rblk << UOp.const(dtypes.index, m_shift)) + rp
  quarter_idx = UOp.const(dtypes.index, quarter)
  idx0 = idx_base.copy()
  idx0[axis] = base
  idx1 = idx_base.copy()
  idx1[axis] = base + quarter_idx
  idx2 = idx_base.copy()
  idx2[axis] = base + quarter_idx + quarter_idx
  idx3 = idx_base.copy()
  idx3[axis] = base + quarter_idx + quarter_idx + quarter_idx
  v0 = x_ptr.vindex(*idx0)
  v1 = x_ptr.vindex(*idx1)
  v2 = x_ptr.vindex(*idx2)
  v3 = x_ptr.vindex(*idx3)
  x0r, x0i = v0.gep(0), v0.gep(1)
  x1r, x1i = v1.gep(0), v1.gep(1)
  x2r, x2i = v2.gep(0), v2.gep(1)
  x3r, x3i = v3.gep(0), v3.gep(1)
  t0r, t0i = x0r + x2r, x0i + x2i
  t1r, t1i = x0r - x2r, x0i - x2i
  t2r, t2i = x1r + x3r, x1i + x3i
  u3r, u3i = x1r - x3r, x1i - x3i
  if inverse:
    t3r, t3i = -u3i, u3r
  else:
    t3r, t3i = u3i, -u3r
  y0r, y0i = t0r + t2r, t0i + t2i
  y1r, y1i = t1r + t3r, t1i + t3i
  y2r, y2i = t0r - t2r, t0i - t2i
  y3r, y3i = t1r - t3r, t1i - t3i
  if stage4 == 0:
    z1r, z1i = y1r, y1i
    z2r, z2i = y2r, y2i
    z3r, z3i = y3r, y3i
  else:
    tw_idx = rp << UOp.const(dtypes.index, stride_shift)
    w1 = tw_ptr.vindex(tw_idx)
    tw_idx2 = tw_idx << UOp.const(dtypes.index, 1)
    tw_idx3 = tw_idx2 + tw_idx
    w2 = tw_ptr.vindex(tw_idx2)
    w3 = tw_ptr.vindex(tw_idx3)
    w1r, w1i = w1.gep(0), w1.gep(1)
    w2r, w2i = w2.gep(0), w2.gep(1)
    w3r, w3i = w3.gep(0), w3.gep(1)
    z1r = y1r * w1r - y1i * w1i
    z1i = y1r * w1i + y1i * w1r
    z2r = y2r * w2r - y2i * w2i
    z2i = y2r * w2i + y2i * w2r
    z3r = y3r * w3r - y3i * w3i
    z3i = y3r * w3i + y3i * w3r
  out0 = UOp.vectorize(y0r, y0i)
  out1 = UOp.vectorize(z1r, z1i)
  out2 = UOp.vectorize(z2r, z2i)
  out3 = UOp.vectorize(z3r, z3i)
  s0 = out_ptr.index(*idx0, ptr=True).store(out0)
  s1 = out_ptr.index(*idx1, ptr=True).store(out1)
  s2 = out_ptr.index(*idx2, ptr=True).store(out2)
  s3 = out_ptr.index(*idx3, ptr=True).store(out3)
  return UOp.sink(s0, s1, s2, s3).end(*ranges)


def _uop_fft1d_128_stage_last_axis(x_ptr: UOp, out_ptr: UOp, tw_ptr: UOp, axis: int, inverse: bool) -> UOp:
  if axis not in (0, 1, 2):
    raise ValueError("axis must be 0,1,2")
  if axis == 2:
    r0 = UOp.range(128, 0)
    r1 = UOp.range(128, 1)
    rp = UOp.range(64, 2)
    ranges = [r0, r1, rp]
    half_idx = UOp.const(dtypes.index, 64)
    idx_even = [r0, r1, rp]
    idx_odd = [r0, r1, rp + half_idx]
    v_even = x_ptr.vindex(*idx_even)
    v_odd = x_ptr.vindex(*idx_odd)
    even_r, even_i = v_even.gep(0), v_even.gep(1)
    odd_r, odd_i = v_odd.gep(0), v_odd.gep(1)
    w = tw_ptr.vindex(rp)
    w_r, w_i = w.gep(0), w.gep(1)
    t_r = odd_r * w_r - odd_i * w_i
    t_i = odd_r * w_i + odd_i * w_r
    out0_r = even_r + t_r
    out0_i = even_i + t_i
    out1_r = even_r - t_r
    out1_i = even_i - t_i
    if inverse:
      scale = UOp.const(dtypes.float, 1.0 / 128.0)
      out0_r = out0_r * scale
      out0_i = out0_i * scale
      out1_r = out1_r * scale
      out1_i = out1_i * scale
    out_vec = UOp.vectorize(out0_r, out0_i, out1_r, out1_i)
    out_ptr_vec = out_ptr.index(r0, r1, rp, ptr=True).cast(out_ptr.dtype.base.vec(4).ptr(size=out_ptr.dtype.size, addrspace=out_ptr.dtype.addrspace))
    store = out_ptr_vec.store(out_vec)
    return UOp.sink(store).end(*ranges)
  r_other = [UOp.range(128, 0), UOp.range(128, 1)]
  r_p = UOp.range(64, 2)
  if axis == 0:
    ranges = [r_p, r_other[0], r_other[1]]
    idx = [None, r_other[0], r_other[1]]
  elif axis == 1:
    ranges = [r_other[0], r_p, r_other[1]]
    idx = [r_other[0], None, r_other[1]]
  else:
    ranges = [r_other[0], r_other[1], r_p]
    idx = [r_other[0], r_other[1], None]
  half_idx = UOp.const(dtypes.index, 64)
  idx_even = idx.copy()
  idx_even[axis] = r_p
  idx_odd = idx.copy()
  idx_odd[axis] = r_p + half_idx
  v_even = x_ptr.vindex(*idx_even)
  v_odd = x_ptr.vindex(*idx_odd)
  even_r, even_i = v_even.gep(0), v_even.gep(1)
  odd_r, odd_i = v_odd.gep(0), v_odd.gep(1)
  w = tw_ptr.vindex(r_p)
  w_r, w_i = w.gep(0), w.gep(1)
  t_r = odd_r * w_r - odd_i * w_i
  t_i = odd_r * w_i + odd_i * w_r
  out0_r = even_r + t_r
  out0_i = even_i + t_i
  out1_r = even_r - t_r
  out1_i = even_i - t_i
  if inverse:
    scale = UOp.const(dtypes.float, 1.0 / 128.0)
    out0_r = out0_r * scale
    out0_i = out0_i * scale
    out1_r = out1_r * scale
    out1_i = out1_i * scale
  out0 = UOp.vectorize(out0_r, out0_i)
  out1 = UOp.vectorize(out1_r, out1_i)
  idx_out0 = idx.copy()
  idx_out0[axis] = r_p
  idx_out1 = idx.copy()
  idx_out1[axis] = r_p + half_idx
  s0 = out_ptr.index(*idx_out0, ptr=True).store(out0)
  s1 = out_ptr.index(*idx_out1, ptr=True).store(out1)
  return UOp.sink(s0, s1).end(*ranges)


def _uop_fft1d_128_axis(x_ptr: UOp, tmp_ptr: UOp, out_ptr: UOp, tw_ptr: UOp, axis: int, inverse: bool) -> list[UOp]:
  kernels: list[UOp] = []
  src = x_ptr
  dst = out_ptr
  for stage4 in range(3):
    kernels.append(_uop_fft1d_128_stage4_axis(src, dst, tw_ptr, axis, inverse, stage4))
    src, dst = dst, (tmp_ptr if dst is out_ptr else out_ptr)
  kernels.append(_uop_fft1d_128_stage_last_axis(src, dst, tw_ptr, axis, inverse))
  return kernels


def _uop_fft1d_128_axis_radix2(x_ptr: UOp, tmp_ptr: UOp, out_ptr: UOp, tw_ptr: UOp, axis: int, inverse: bool) -> list[UOp]:
  kernels: list[UOp] = []
  src = x_ptr
  dst = out_ptr
  for stage in range(6):
    kernels.append(_uop_fft1d_128_stage_axis(src, dst, tw_ptr, axis, inverse, stage))
    src, dst = dst, (tmp_ptr if dst is out_ptr else out_ptr)
  kernels.append(_uop_fft1d_128_stage_last_axis(src, dst, tw_ptr, axis, inverse))
  return kernels


def _fft3d_128_plan(device: str, dtype) -> str:
  # NOTE: kernel_multi is disabled, so plan selection doesn't matter.
  # Return radix4 as default (it was generally faster in benchmarks).
  return "radix4"


def _fft3d_128_kernel_multi(x: Tensor, inverse: bool, plan: str | None = None) -> Tensor:
  # NOTE: kernel_multi has buffer aliasing issues that produce incorrect results.
  # The custom_kernel infrastructure doesn't support the buffer rotation needed for 3D FFT.
  # Disabled until the underlying issue is fixed. Use regular path instead.
  return None


def _fft3d_128_special(x: Tensor, inverse: bool) -> Tensor:
  layout = int(getenv("TINYGRAD_FFT_3D_128_LAYOUT", 0))
  if layout == 1:
    x = x.permute(2, 0, 1, 3).contiguous()
    x = _fft2d_pow2_fast(x, inverse)
    x, inv = _permute_axis_to_last2(x, 0)
    x = _fft1d_pow2_fast(x, 128, inverse)
    x = x if inv is None else x.permute(*inv)
    return x.permute(1, 2, 0, 3)
  if layout == 2:
    x = _fft2d_pow2_fast(x, inverse)
    x, inv = _permute_axis_to_last2(x, 0)
    x = _fft1d_pow2_fast(x, 128, inverse)
    x = x if inv is None else x.permute(*inv)
    return x
  x = x.permute(1, 2, 0, 3).contiguous()
  x = _fft2d_pow2_fast(x, inverse)
  x, inv = _permute_axis_to_last2(x, 0)
  x = _fft1d_pow2_fast(x, 128, inverse)
  x = x if inv is None else x.permute(*inv)
  return x.permute(2, 0, 1, 3)


def _fft3d_impl_4d(x: Tensor, inverse: bool, use_pow2: bool, use_block2d: bool,
                   contig_thr: int, contig_mask: tuple[bool, bool]) -> Tensor:
  prefix = x.ndim - 4
  def _perm_last3(x: Tensor, order: tuple[int, int, int]) -> Tensor:
    if order == (0, 1, 2):
      return x
    key = (x.ndim, order)
    perm = _permute_last3_cache.get(key)
    if perm is None:
      perm = tuple(list(range(prefix)) + [prefix + order[0], prefix + order[1], prefix + order[2], prefix + 3])
      _permute_last3_cache[key] = perm
    return x.permute(*perm)
  if use_block2d:
    x = _fft2d_pow2_fast(x, inverse)
    if contig_mask[0]:
      x = _maybe_contiguous(x, contig_thr)
    x = _perm_last3(x, (1, 2, 0))
    if contig_mask[1]:
      x = _maybe_contiguous(x, contig_thr)
    x = _fft1d_pow2_fast(x, int(x.shape[-2]), inverse)
    x = _perm_last3(x, (2, 0, 1))
    return x
  if use_pow2:
    n0, n1, n2 = int(x.shape[-4]), int(x.shape[-3]), int(x.shape[-2])
    use_cycle = bool(getenv("TINYGRAD_FFT_3D_CYCLE", 0))
    if use_cycle:
      x = _fft1d_pow2_fast(x, n2, inverse)
      x = _perm_last3(x, (1, 2, 0))
      if contig_mask[0]:
        x = _maybe_contiguous(x, contig_thr)
      x = _fft1d_pow2_fast(x, n0, inverse)
      x = _perm_last3(x, (1, 2, 0))
      if contig_mask[1]:
        x = _maybe_contiguous(x, contig_thr)
      x = _fft1d_pow2_fast(x, n1, inverse)
      x = _perm_last3(x, (1, 2, 0))
      return x
    x = _fft1d_pow2_fast(x, n2, inverse)
    x = _perm_last3(x, (0, 2, 1))
    if contig_mask[0]:
      x = _maybe_contiguous(x, contig_thr)
    x = _fft1d_pow2_fast(x, n1, inverse)
    x = _perm_last3(x, (1, 2, 0))
    if contig_mask[1]:
      x = _maybe_contiguous(x, contig_thr)
    x = _fft1d_pow2_fast(x, n0, inverse)
    x = _perm_last3(x, (2, 1, 0))
    return x
  x = _fft1d_impl(x, inverse=inverse)
  x = _perm_last3(x, (0, 2, 1))
  if contig_mask[0]:
    x = _maybe_contiguous(x, contig_thr)
  x = _fft1d_impl(x, inverse=inverse)
  x = _perm_last3(x, (1, 2, 0))
  if contig_mask[1]:
    x = _maybe_contiguous(x, contig_thr)
  x = _fft1d_impl(x, inverse=inverse)
  x = _perm_last3(x, (2, 1, 0))
  return x


def _fft3d_impl_order(x: Tensor, inverse: bool, order: tuple[int, int, int],
                      use_pow2: bool, pow2_axes: tuple[bool, bool, bool], contig_thr: int) -> Tensor:
  if use_pow2:
    for i, ax in enumerate(order):
      x, inv = _permute_axis_to_last2(x, ax)
      x = _fft1d_pow2_fast(x, int(x.shape[-2]), inverse)
      x = x if inv is None else x.permute(*inv)
      x = _maybe_contiguous(x, contig_thr)
    return x
  for ax in order:
    x = _fft_axis(x, ax, inverse)
    x = _maybe_contiguous(x, contig_thr)
  return x


def _fft3d_autotune(x: Tensor, inverse: bool, use_pow2: bool, use_block2d: bool,
                    order: tuple[int, int, int] | None, pow2_axes: tuple[bool, bool, bool],
                    contig_thr: int, contig_mask: tuple[bool, bool]) -> tuple[tuple[int, int, int] | None, int]:
  if x.device != "CPU":
    return order, contig_thr
  if x.ndim == 4:
    thr_list = []
    for v in (0, contig_thr, 4096, 16384):
      if v not in thr_list: thr_list.append(v)
    best_thr, best_time = contig_thr, float("inf")
    for thr in thr_list:
      x0 = _fft3d_impl_4d(x, inverse, use_pow2, use_block2d, thr, contig_mask).realize()
      start = time.perf_counter()
      _ = _fft3d_impl_4d(x0, inverse, use_pow2, use_block2d, thr, contig_mask).realize()
      dt = time.perf_counter() - start
      if dt < best_time:
        best_time, best_thr = dt, thr
    return order, best_thr
  axes = [x.ndim - 4, x.ndim - 3, x.ndim - 2]
  perms = [
    (axes[0], axes[1], axes[2]),
    (axes[0], axes[2], axes[1]),
    (axes[1], axes[0], axes[2]),
    (axes[1], axes[2], axes[0]),
    (axes[2], axes[0], axes[1]),
    (axes[2], axes[1], axes[0]),
  ]
  thr_list = []
  for v in (0, contig_thr, 4096, 16384):
    if v not in thr_list: thr_list.append(v)
  best_order, best_thr, best_time = order, contig_thr, float("inf")
  for ord0 in perms:
    for thr in thr_list:
      x0 = _fft3d_impl_order(x, inverse, ord0, use_pow2, pow2_axes, thr).realize()
      start = time.perf_counter()
      _ = _fft3d_impl_order(x0, inverse, ord0, use_pow2, pow2_axes, thr).realize()
      dt = time.perf_counter() - start
      if dt < best_time:
        best_time, best_order, best_thr = dt, ord0, thr
  return best_order, best_thr


def _fft3d_impl(x: Tensor, inverse: bool = False) -> Tensor:
  x = _as_complex(x)
  if x.ndim < 3:
    raise ValueError("fft3d requires at least 3D input")
  if x.ndim == 4 and tuple(int(s) for s in x.shape[-4:-1]) == (128, 128, 128):
    if getenv("TINYGRAD_FFT_3D_128_KERNEL_MULTI", 1):
      result = _fft3d_128_kernel_multi(x, inverse)
      if result is not None:
        return result
      # Fall through to regular path if kernel_multi returns None (radix4 buffer issue)
    if getenv("TINYGRAD_FFT_3D_128_SPECIAL", 0):
      return _fft3d_128_special(x, inverse)
  if x.ndim == 4 and tuple(int(s) for s in x.shape[-4:-1]) == (8, 8, 8):
    return _fft3d_fused_8(x, inverse)
  plan_key = (tuple(x.shape), x.device, x.dtype)
  plan = _fft3d_plan_cache.get(plan_key)
  if plan is None:
    axes = (x.ndim - 4, x.ndim - 3, x.ndim - 2)
    sizes = tuple(int(x.shape[a]) for a in axes)
    pow2_axes = tuple(_is_power_of_two(s) for s in sizes)
    use_pow2 = all(pow2_axes)
    use_block2d = bool(getenv("TINYGRAD_FFT_3D_BLOCK", 0)) and use_pow2
    contig_mask = (True, True)
    if x.ndim == 4 and min(sizes) >= 128:
      contig_mask = (True, False)
    if x.ndim == 4:
      order = None
    else:
      last2 = x.ndim - 2
      others = [a for a in axes if a != last2]
      order = (last2,) + tuple(sorted(others, key=lambda a: x.shape[a]))
    contig_thr = _get_fft3d_contig_threshold(x.device, x.dtype)
    if getenv("TINYGRAD_FFT_3D_AUTOTUNE", 0) and not capturing:
      order, contig_thr = _fft3d_autotune(x, inverse, use_pow2, use_block2d, order, pow2_axes, contig_thr, contig_mask)
    plan = _fft3d_plan_cache.setdefault(plan_key, (use_block2d, use_pow2, order, pow2_axes, contig_thr, contig_mask))
  use_block2d, use_pow2, order, pow2_axes, contig_thr, contig_mask = plan
  if x.ndim == 4:
    return _fft3d_impl_4d(x, inverse, use_pow2, use_block2d, contig_thr, contig_mask)
  if use_block2d:
    x = _fft2d_pow2_fast(x, inverse)
    x = _maybe_contiguous(x, contig_thr)
    x, inv = _permute_axis_to_last2(x, x.ndim - 4)
    x = _fft1d_pow2_fast(x, int(x.shape[-2]), inverse)
    x = x if inv is None else x.permute(*inv)
    return _maybe_contiguous(x, contig_thr)
  return _fft3d_impl_order(x, inverse, order, use_pow2, pow2_axes, contig_thr)


def fft1d(x: Tensor, inverse: bool = False) -> Tensor:
  x = _as_complex(x)
  _ensure_autotuned(x.device, x.dtype)
  n = int(x.shape[-2])
  if n <= 1:
    return x
  if getenv("TINYGRAD_FFT_JIT", 1) and not capturing:
    key = (tuple(x.shape), inverse, x.device, x.dtype, "1d")
    plan = _fft_plan_cache.get(key)
    if plan is None:
      def plan_fn(x_in: Tensor) -> Tensor:
        return _fft1d_impl(x_in, inverse).realize()
      plan = _fft_plan_cache.setdefault(key, TinyJit(plan_fn))
    try:
      return plan(x)
    except JitError:
      _fft_plan_cache.pop(key, None)
  return _fft1d_impl(x, inverse)


def ifft1d(x: Tensor) -> Tensor:
  return fft1d(x, inverse=True)


def fft2d(x: Tensor, inverse: bool = False) -> Tensor:
  x = _as_complex(x)
  _ensure_autotuned(x.device, x.dtype)
  if x.ndim < 2:
    raise ValueError("fft2d requires at least 2D input")
  if getenv("TINYGRAD_FFT_JIT", 1) and not capturing:
    key = (tuple(x.shape), inverse, x.device, x.dtype, "2d")
    plan = _fft_plan_cache.get(key)
    if plan is None:
      def plan_fn(x_in: Tensor) -> Tensor:
        return _fft2d_impl(x_in, inverse).realize()
      plan = _fft_plan_cache.setdefault(key, TinyJit(plan_fn))
    try:
      return plan(x)
    except JitError:
      _fft_plan_cache.pop(key, None)
  return _fft2d_impl(x, inverse=inverse)


def ifft2d(x: Tensor) -> Tensor:
  return fft2d(x, inverse=True)


def fft3d(x: Tensor, inverse: bool = False) -> Tensor:
  x = _as_complex(x)
  _ensure_autotuned(x.device, x.dtype)
  if x.ndim < 3:
    raise ValueError("fft3d requires at least 3D input")
  if getenv("TINYGRAD_FFT_3D_MULTI_JIT", 0) and not capturing:
    axes = [x.ndim - 4, x.ndim - 3, x.ndim - 2]
    sizes = {a: x.shape[a] for a in axes}
    order = tuple(ax for _, ax in sorted((sizes[a], a) for a in axes))
    for ax in order:
      x = _fft_axis_fft1d(x, ax, inverse)
    return x
  if x.device == "CPU" and getenv("TINYGRAD_FFT_NUMPY_3D", 0):
    import numpy as np
    data = x.numpy()
    c = data[..., 0] + 1j * data[..., 1]
    out = np.fft.ifftn(c) if inverse else np.fft.fftn(c)
    return Tensor.stack([Tensor(out.real, device=x.device), Tensor(out.imag, device=x.device)], dim=-1)
  if getenv("TINYGRAD_FFT_JIT", 1) and not capturing:
    key = (tuple(x.shape), inverse, x.device, x.dtype, "3d")
    plan = _fft_plan_cache.get(key)
    if plan is None:
      def plan_fn(x_in: Tensor) -> Tensor:
        return _fft3d_impl(x_in, inverse).realize()
      plan = _fft_plan_cache.setdefault(key, TinyJit(plan_fn))
    try:
      return plan(x)
    except JitError:
      _fft_plan_cache.pop(key, None)
  return _fft3d_impl(x, inverse=inverse)


def ifft3d(x: Tensor) -> Tensor:
  return fft3d(x, inverse=True)


class FFTPlan:
  def __init__(self, input_shape: tuple[int, ...], complex_shape: tuple[int, ...],
               inverse: bool, device: str, dtype, kind: str):
    if kind not in ("1d", "2d", "3d"):
      raise ValueError("FFTPlan kind must be '1d', '2d', or '3d'")
    self.shape = input_shape
    self.complex_shape = complex_shape
    self.inverse = inverse
    self.device = device
    self.dtype = dtype
    self.kind = kind
    def plan_fn(x_in: Tensor) -> Tensor:
      if self.kind == "1d":
        return _fft1d_impl(x_in, self.inverse).realize()
      if self.kind == "2d":
        return _fft2d_impl(x_in, self.inverse).realize()
      return _fft3d_impl(x_in, self.inverse).realize()
    self._jit = TinyJit(plan_fn)

  def __call__(self, x: Tensor) -> Tensor:
    in_shape = tuple(x.shape)
    if in_shape != self.shape and in_shape != self.complex_shape:
      raise ValueError(f"FFTPlan shape mismatch: expected {self.shape}, got {in_shape}")
    if x.device != self.device:
      raise ValueError(f"FFTPlan device mismatch: expected {self.device}, got {x.device}")
    if x.dtype != self.dtype:
      raise ValueError(f"FFTPlan dtype mismatch: expected {self.dtype}, got {x.dtype}")
    return self._jit(_as_complex(x))


def fft_plan(x: Tensor, inverse: bool = False, kind: str = "1d") -> FFTPlan:
  input_shape = tuple(x.shape)
  x = _as_complex(x)
  complex_shape = tuple(x.shape)
  _ensure_autotuned(x.device, x.dtype)
  key = (complex_shape, inverse, x.device, x.dtype, kind)
  plan = _fft_plan_obj_cache.get(key)
  if plan is None:
    plan = _fft_plan_obj_cache.setdefault(
      key, FFTPlan(input_shape, complex_shape, inverse, x.device, x.dtype, kind)
    )
  return plan


def rfft1d(x: Tensor) -> Tensor:
  if x.ndim < 1:
    raise ValueError("rfft1d requires at least 1D input")
  if x.shape[-1] == 2:
    raise ValueError("rfft1d expects real input without complex last dim")
  return _rfft1d_impl(x)


def irfft1d(x: Tensor, n: int | None = None) -> Tensor:
  if x.ndim < 1:
    raise ValueError("irfft1d requires at least 1D input")
  x = _as_complex(x)
  m = int(x.shape[-2])
  n_out = (m - 1) * 2 if n is None else n
  if n_out < 2:
    return x[..., :1, 0]
  if m < 2:
    return x[..., :1, 0]
  tail = _complex_conj(x[..., 1:m-1, :].flip(-2))
  full = Tensor.cat(x, tail, dim=-2)
  out = ifft1d(full)
  return out[..., :n_out, 0]


def rfft2d(x: Tensor) -> Tensor:
  if x.ndim < 2:
    raise ValueError("rfft2d requires at least 2D input")
  return _rfft2d_impl(x)


def irfft2d(x: Tensor, n: tuple[int, int] | None = None) -> Tensor:
  if x.ndim < 2:
    raise ValueError("irfft2d requires at least 2D input")
  x = _as_complex(x)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    x = x.permute(*range(x.ndim - 3), x.ndim - 2, x.ndim - 3, x.ndim - 1)
  x = ifft1d(x)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    x = x.permute(*range(x.ndim - 3), x.ndim - 2, x.ndim - 3, x.ndim - 1)
  n1 = n[1] if n is not None else None
  out = irfft1d(x, n=n1)
  if n is not None:
    if out.shape[-2:] == (n[0], n1):
      return out
    return out[..., :n[0], :n1]
  return out


def autotune_fft_thresholds(device: str = "CPU", dtype=dtypes.float32,
                            sizes: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
                            thresholds: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
                            iters: int = 2) -> int:
  _fft_autotune_active.add((device, dtype))
  best_thr = thresholds[0]
  best_time = float("inf")
  for thr in thresholds:
    _fft_threshold_cache[(device, dtype)] = thr
    total = 0.0
    for n in sizes:
      x = Tensor.randn(n, 2, device=device, dtype=dtype)
      out = _fft1d_impl(x, inverse=False)
      out.realize()
      start = time.perf_counter()
      for _ in range(iters):
        out = _fft1d_impl(x, inverse=False)
        out.realize()
      total += time.perf_counter() - start
    if total < best_time:
      best_time = total
      best_thr = thr
  _fft_threshold_cache[(device, dtype)] = best_thr
  _fft_autotune_active.discard((device, dtype))
  return best_thr


def autotune_split_radix_thresholds(device: str = "CPU", dtype=dtypes.float32,
                                    sizes: tuple[int, ...] = (8, 16, 32, 64),
                                    thresholds: tuple[int, ...] = (0, 16, 32, 64),
                                    iters: int = 1) -> int:
  _fft_autotune_active.add((device, dtype))
  best_thr = thresholds[0]
  best_time = float("inf")
  for thr in thresholds:
    _fft_split_radix_threshold_cache[(device, dtype)] = thr
    total = 0.0
    for n in sizes:
      x = Tensor.randn(n, 2, device=device, dtype=dtype)
      out = _fft1d_impl(x, inverse=False)
      out.realize()
      start = time.perf_counter()
      for _ in range(iters):
        out = _fft1d_impl(x, inverse=False)
        out.realize()
      total += time.perf_counter() - start
    if total < best_time:
      best_time = total
      best_thr = thr
  _fft_split_radix_threshold_cache[(device, dtype)] = best_thr
  _fft_autotune_active.discard((device, dtype))
  return best_thr
