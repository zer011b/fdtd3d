# Description

This is test when exact solution should be obtained from numerical scheme.

**In this test 2D mode is used and auxiliary grids for incident wave are tested.**

# Equation

```
d^2 U(x,t) / dt^2 = с^2 * d^2 U(x,t) / dx^2
```

# Exact solution

```
U(x,t) = i * exp (-i * (wt - kx))
```

# Numerical solution

"Magic time step" case of `dt = dx / c` allows to obtain exact solution from numerical relations.
