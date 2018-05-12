# Description

This is test when exact solution should be obtained from numerical scheme.

**In this test additional grids for incident wave are tested.**

# Exact solution

```
U(x,t) = i * exp (-i * (wt - kx))
```

# Numerical solution

"Magic time step" case of `dt = dx / c` allows to obtain exact solution from numerical relations.
