# Description

This is test with comparison of exact solution with numerical one.

# Exact solution

```
Ez = c^2 * x^2 * t
Hy = c^2 * eps0 * x^2 * t

Jz = eps0 * (- 2 * x * t +x^2)
Mz = - 2 * x * t + x^2 * eps0 * mu0
```

# Numerical solution

Case of `dt = 0.5 * dx / c`. Numerical solution should match exact solution (with floating point accuracy).
