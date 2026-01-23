import os, time, math
from tinygrad.tensor import Tensor
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
_rev8_cache: dict[str, Tensor] = {}
_fft_split_radix_threshold_cache: dict[tuple[str, object], int] = {}
_digit_reverse_cache: dict[tuple[int, tuple[int, ...]], list[int]] = {}
_digit_reverse_tensor_cache: dict[tuple[int, tuple[int, ...], str], Tensor] = {}


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


def _fft_pow2_iterative_radix8(x: Tensor, n: int, inverse: bool) -> Tensor:
  threshold = _get_radix8_threshold(x.device, x.dtype)
  if n >= 8 and n >= threshold:
    bits = int(math.log2(n))
    radices = [8] + [2] * (bits - 3)
    idx = _digit_reverse_tensor(n, radices, x.device).reshape((1,) * (x.ndim - 2) + (n, 1)).expand(*x.shape)
    x = x.gather(-2, idx)
    prefix = x.shape[:-2]
    x = x.reshape(*prefix, -1, 8, 2)
    x = _fft_pow2_base8(x, inverse)
    x = x.reshape(*prefix, n, 2)
    m = 16
  else:
    radices = [2] * int(math.log2(n))
    idx = _digit_reverse_tensor(n, radices, x.device).reshape((1,) * (x.ndim - 2) + (n, 1)).expand(*x.shape)
    x = x.gather(-2, idx)
    m = 2

  sign = 1.0 if inverse else -1.0
  while m <= n:
    half = m // 2
    prefix = x.shape[:-2]
    x = x.reshape(*prefix, -1, m, 2)
    even = x[..., :half, :]
    odd = x[..., half:, :]
    if m == 2:
      x = Tensor.cat(_complex_add(even, odd), _complex_sub(even, odd), dim=-2)
    else:
      tw = _stage_twiddle(m, inverse, x.device, x.dtype)
      tw = tw.reshape((1,) * (x.ndim - 2) + (half, 2))
      t = _complex_mul(odd, tw)
      x = Tensor.cat(_complex_add(even, t), _complex_sub(even, t), dim=-2)
    x = x.reshape(*prefix, n, 2)
    m *= 2

  if inverse:
    scale = 1.0 / n
    x = Tensor.stack([x[..., 0] * scale, x[..., 1] * scale], dim=-1)
  return x


def _fft_mixed_radix_plan(x: Tensor, n: int, inverse: bool) -> Tensor:
  radices = _factor_list(n)
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
    split_thr = _get_split_radix_threshold(x.device, x.dtype)
    if split_thr and n <= split_thr:
      out = _fft_pow2_split_radix(x, n, inverse)
      if inverse:
        scale = 1.0 / n
        out = Tensor.stack([out[..., 0] * scale, out[..., 1] * scale], dim=-1)
      return out
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
  x = _fft1d_impl(x, inverse=inverse)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    x = x.permute(*range(x.ndim - 3), x.ndim - 2, x.ndim - 3, x.ndim - 1)
  x = _fft1d_impl(x, inverse=inverse)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    x = x.permute(*range(x.ndim - 3), x.ndim - 2, x.ndim - 3, x.ndim - 1)
  return x


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


class FFTPlan:
  def __init__(self, input_shape: tuple[int, ...], complex_shape: tuple[int, ...],
               inverse: bool, device: str, dtype, kind: str):
    if kind not in ("1d", "2d"):
      raise ValueError("FFTPlan kind must be '1d' or '2d'")
    self.shape = input_shape
    self.complex_shape = complex_shape
    self.inverse = inverse
    self.device = device
    self.dtype = dtype
    self.kind = kind
    def plan_fn(x_in: Tensor) -> Tensor:
      if self.kind == "1d":
        return _fft1d_impl(x_in, self.inverse).realize()
      return _fft2d_impl(x_in, self.inverse).realize()
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
  n = int(x.shape[-1])
  out = fft1d(_as_complex(x))
  return out[..., : n // 2 + 1, :]


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
  x = rfft1d(x)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    x = x.permute(*range(x.ndim - 3), x.ndim - 2, x.ndim - 3, x.ndim - 1)
  x = fft1d(x)
  if x.ndim == 3:
    x = x.permute(1, 0, 2)
  else:
    x = x.permute(*range(x.ndim - 3), x.ndim - 2, x.ndim - 3, x.ndim - 1)
  return x


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
  import numpy as np
  _fft_autotune_active.add((device, dtype))
  best_thr = thresholds[0]
  best_time = float("inf")
  for thr in thresholds:
    _fft_threshold_cache[(device, dtype)] = thr
    total = 0.0
    for n in sizes:
      x = Tensor(np.random.randn(n, 2).astype(np.float32), device=device, dtype=dtype)
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
  import numpy as np
  _fft_autotune_active.add((device, dtype))
  best_thr = thresholds[0]
  best_time = float("inf")
  for thr in thresholds:
    _fft_split_radix_threshold_cache[(device, dtype)] = thr
    total = 0.0
    for n in sizes:
      x = Tensor(np.random.randn(n, 2).astype(np.float32), device=device, dtype=dtype)
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
