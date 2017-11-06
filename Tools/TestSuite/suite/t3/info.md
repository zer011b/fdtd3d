# Description

This is test with comparison of exact solution with numerical one.

# Exact solution

```
Ex = c^2 * t * (y^2 + z^2)
Ey = c^2 * t * (x^2 + z^2)
Ez = c^2 * t * (x^2 + y^2)

Hx = c^2 * eps0 * t * (y^2 + z^2)
Hy = c^2 * eps0 * t * (x^2 + z^2)
Hz = c^2 * eps0 * t * (x^2 + y^2)

Jx = c^2 * eps0 * (2 * t * (z - y) + y^2 + z^2)
Jy = c^2 * eps0 * (2 * t * (x - z) + x^2 + z^2)
Jz = c^2 * eps0 * (2 * t * (y - x) + x^2 + y^2)

Mx = c^2 * (2 * t * (y - z) + eps0 * mu0 * (y^2 + z^2))
My = c^2 * (2 * t * (z - x) + eps0 * mu0 * (x^2 + z^2))
Mz = c^2 * (2 * t * (x - y) + eps0 * mu0 * (x^2 + y^2))
```

# Numerical solution

Case of `dt = 0.5 * dx / c`. Numerical solution should match exact solution (with floating point accuracy).
