# Description

This is test with comparison of exact solution with numerical one (should have second order accuracy).

# Exact solution

```
Ez = x^2 * t
Hy = eps0 * x^2 * t
```

# Numerical solution

Case of `dt = 0.5 * dx / c`. Numerical solution should match exact solution (with floating point accuracy).
