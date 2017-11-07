# Description

This is test with comparison of exact solution with numerical one.

# Exact solution

```
Ez = c^2 * x^2 * t^2
Hy = c^2 * eps0 * x^2 * t^2

Jz = 2 * eps0 * x * t * (x - t)
Mz = 2 * x * t * (eps0 * mu0 * x - t)
```

# Numerical solution

Case of `dt = 0.5 * dx / c`. Numerical solution should match exact solution (with floating point accuracy).
