#### Error and Completeness Utilities

"""
    η(m) = Martin2016_complete(m, A, m50, ρ)

Completeness model of [Martin2016a](@citet) implemented as their Equation 7:

```math
\\eta(m) = \\frac{A}{1 + \\text{exp} \\left( \\frac{m - m_{50}}{\\rho} \\right)}
```

`m` is the magnitude of interest, `A` is the maximum completeness, `m50` is the magnitude at which the data are 50% complete, and `ρ` is an effective slope modifier.
"""
Martin2016_complete(m, A, m50, ρ) = A / (1 + exp((m - m50) / ρ))

"""
    exp_photerr(m, a, b, c, d)

Exponential model for photometric errors of the form

```math
\\sigma(m) = a^{b \\times \\left( m-c \\right)} + d
```

Reported values for some HST data were `a=1.05, b=10.0, c=32.0, d=0.01`. 
"""
exp_photerr(m, a, b, c, d) = a^(b * (m - c)) + d