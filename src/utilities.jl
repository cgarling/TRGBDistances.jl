
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

"""
    _build_histogram(mags, m_min, m_max, bin_width)
Build a histogram of the magnitudes `mags` with bins defined by `m_min`, `m_max`, and `bin_width`.
Returns a tuple of the form `(counts, bin_centers)`.
"""
function _build_histogram(mags, m_min, m_max, bin_width)
    n_bins = max(3, ceil(Int, (m_max - m_min) / bin_width))
    bin_centers = [m_min + (i - 0.5) * bin_width for i in 1:n_bins]
    counts = zeros(Int, n_bins)
    for m in mags
        idx = floor(Int, (m - m_min) / bin_width) + 1
        if 1 <= idx <= n_bins
            counts[idx] += 1
        end
    end
    return counts, bin_centers
end