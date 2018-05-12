# Description

This is test with comparison of exact solution with numerical one.

# Equations

```
dEx/dt = 1/eps0 * (dHz/dy - dHy/dz + Jx)
dEy/dt = 1/eps0 * (dHx/dz - dHz/dx + Jy)
dEz/dt = 1/eps0 * (dHy/dx - dHx/dy + Jz)

dHx/dt = 1/mu0 * (dEy/dz - dEz/dy + Mx)
dHy/dt = 1/mu0 * (dEz/dx - dEx/dz + My)
dHz/dt = 1/mu0 * (dEx/dy - dEy/dx + Mz)
```

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

Case of `dt = 0.5 * dx / c`, i.e. Courant number set to 0.5. Numerical solution should match exact solution (with floating point accuracy).
