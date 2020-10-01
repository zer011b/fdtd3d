# Description

This is test with comparison of exact solution with numerical one for TEy mode.

# Equations

```
dEz/dt = 1/eps0 * (dHy/dx - Jz)
dHy/dt = 1/mu0 * (dEz/dx - My)
```

# Exact solution

```
Ez = c^2 * x^2 * t
Hy = c^2 * eps0 * x^2 * t

Jz = -c^2 * eps0 * (- 2 * x * t +x^2)
My = -c^2 * (- 2 * x * t + x^2 * eps0 * mu0)
```

# Numerical solution

Case of `dt = 0.5 * dx / c`, i.e. Courant number set to 0.5. Numerical solution should match exact solution (with floating point accuracy).
