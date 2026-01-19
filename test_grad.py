
from tinygrad.tensor import Tensor
import numpy as np

def test_grad_connected():
    try:
        x = Tensor([2.0], requires_grad=True)
        # y = x^3
        y = x * x * x
        
        # We want dy/dx = 3x^2 = 12
        # And we want to use that in a further calculation
        # z = (dy/dx) * 2 = 24
        # And then check dz/dx ??? 
        # Actually, let's just see if we can get dy/dx as a tensor
        
        # Does Tensor have a grad function?
        # Or is there a global grad?
        
        # Attempt 1: backward? But backward returns None, populates .grad
        y.backward()
        g = x.grad
        print(f"Gradient via backward: {g.numpy()[0]}")
        
        # Is g connected?
        # Let's try to differentiate g w.r.t x? (Second derivative)
        # z = g * x # 3x^2 * x = 3x^3
        
        if g.requires_grad:
             print("Gradient has requires_grad=True. Good!")
        else:
             print("Gradient has requires_grad=False. Not connected.")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_grad_connected()
