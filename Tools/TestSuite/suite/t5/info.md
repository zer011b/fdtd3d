# Description

This is test with comparison of exact solution with numerical one (second order of accuracy).
Normed mode of Eps, Mu and C is tested.

# Equations

```
dEz/dt = dHy/dx
dHy/dt = dEz/dx

eps = 1/eps0
mu = 1/mu0
C = 1.0
```

# Exact solution

```
Ez = sin (t - x)
Hy = - sin (t - x)
```

# Numerical solution

Case of `dt = 0.5 * dx`.
