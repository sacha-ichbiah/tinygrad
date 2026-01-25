"""Debug script for unroll issue."""
import sys
from pathlib import Path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tinygrad import Tensor
from tinygrad.physics import compile_symplectic_program, SymplecticPolicy
import numpy as np

def H(q, p):
    return 0.5 * (p * p + q * q).sum()

q = Tensor([1.0])
p = Tensor([0.0])

policy = SymplecticPolicy(accuracy='fast', scan=True)
prog = compile_symplectic_program('canonical', H=H, policy=policy, sample_state=(q, p))

# Check what unroll is chosen
steps = 1000
shape = q.shape
device = q.device

should_scan = policy.should_scan(steps, shape, device)
print(f'should_scan: {should_scan}')

if should_scan:
    unroll = policy.choose_unroll(steps, shape, device)
    print(f'chosen unroll: {unroll}')

print(f'\nPolicy settings:')
print(f'  accuracy: {policy.accuracy}')
print(f'  scan: {policy.scan}')
print(f'  max_unroll: {policy.max_unroll}')
print(f'  min_unroll: {policy.min_unroll}')

# Test with different unroll values
print('\nTesting 1000 steps with different unroll values:')
for unroll in [None, 2, 4, 8, 10]:
    q = Tensor([1.0])
    p = Tensor([0.0])
    prog = compile_symplectic_program('canonical', H=H, policy=policy, sample_state=(q, p))

    (q1, p1), h1 = prog.evolve((q.clone(), p.clone()), dt=0.01, steps=1000,
                                record_every=1000, unroll=unroll)
    q1.realize()
    result = q1.numpy()[0]
    exact = np.cos(10.0)
    error = abs(result - exact)
    print(f'unroll={str(unroll):>4}: q={result:10.6f}, error={error:.2e}')

print(f'\nExpected: cos(10.0) = {np.cos(10.0):.6f}')
