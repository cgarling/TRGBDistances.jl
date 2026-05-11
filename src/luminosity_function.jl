"""
`LuminosityFunction{T}` is the abstract supertype for luminosity functions that can be used
to model the TRGB. Subtypes should implement [`ψ`](@ref).
"""
abstract type LuminosityFunction{T} end
(l::LuminosityFunction)(m) = ψ(l, m)
Base.Broadcast.broadcastable(t::LuminosityFunction) = Ref(t)

"""
    ψ(model::LuminosityFunction, m)

Returns the luminosity function of the `model` at magnitude `m`.
"""
function ψ end

"""
    is_valid(model::LuminosityFunction)
Check whether the parameters of `model` are valid (e.g., non-negative slopes).
By default, returns `true` for all models. Subtypes can override this to enforce parameter
constraints.
"""
is_valid(model::LuminosityFunction) = true

"""
    BrokenPowerLaw(m_trgb, a, b, c) <: LuminosityFunction

Standard broken power-law luminosity function used to model the TRGB luminosity function.
For a magnitude `m`:

- If `m >= m_trgb` (RGB side, fainter): `ψ = 10^(a*(m - m_trgb) + b)`
- If `m < m_trgb` (AGB side, brighter): `ψ = 10^(c*(m - m_trgb))`

Parameters:
- `m_trgb`: TRGB apparent magnitude
- `a`: Power-law slope of the RGB (faint-side slope)
- `b`: Log₁₀ of the amplitude discontinuity at the TRGB (the "bump")
- `c`: Power-law slope of the AGB (bright-side slope)

For initial optimization, eyeballing an approximate `m_trgb` on the CMD is usually sufficient.
`a = 0.3`, `b = 0.2`, `c = 0.1` may serve as first guesses.

# Examples
```jldoctest
julia> m = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1);

julia> m(24.0) ≈ 10^(0.3*0 + 0.2) # at TRGB: 10^(0.3*0 + 0.2) = 10^0.2
true

julia> m(25.0) > m(24.0)  # fainter: higher ψ (rising RGB LF)
true

julia> m(23.0) < 1.0      # brighter: lower ψ (AGB side)
true
```
"""
struct BrokenPowerLaw{T} <: LuminosityFunction{T}
    m_trgb::T
    a::T
    b::T
    c::T
end
BrokenPowerLaw(m_trgb, a, b, c) = BrokenPowerLaw(promote(m_trgb, a, b, c)...)
is_valid(model::BrokenPowerLaw) = reduce(&, (model.a > 0, model.b > 0, model.c > 0))

function ψ(b::BrokenPowerLaw, m)
    Δm = m - b.m_trgb
    return m >= b.m_trgb ? exp10(b.a * Δm + b.b) : exp10(b.c * Δm)
end

"""
    ϕ(m, model, err_func, complete_func, bias_func; int_width=1.0)

Evaluate the convolved luminosity function at observed magnitude `m` (Eq. 4, [Makarov2006](@citet)).

The result is:

```
ϕ(m) = ∫ ψ(x) · N(m | x + bias(x), σ(x)) · f(x) dx
```

where the integral runs over true magnitudes `x` in `[m - int_width, m + int_width]`,
`bias(x)` is the photometric bias (`<measured> - <input>`), `σ(x)` is the photometric
error, and `f(x)` is the completeness.

The `int_width` parameter (default `1.0` mag) controls the half-width of the integration
window. It should be set to at least several times the maximum photometric error to avoid
truncation.
"""
function ϕ(m, model::LuminosityFunction, err_func, complete_func, bias_func; int_width=1.0)
    return quadgk(x -> _ϕ_integrand(x, m, model, err_func, complete_func, bias_func),
                  m - int_width, m + int_width)[1]
end

@inline function _ϕ_integrand(x, m, model::LuminosityFunction, err_func, complete_func, bias_func)
    σ = err_func(x)
    μ = x + bias_func(x)
    z = (m - μ) / σ
    gauss = exp(-z^2 / 2) / σ / sqrt2π  # σ * √(2π)
    return ψ(model, x) * gauss * complete_func(x)
end

_g(m, μ, σ) = exp(-0.5 * ((m - μ) / σ)^2) / (σ * sqrt2π)
"""
    ϕ_norm(limits, model, err_func, complete_func, bias_func; int_width=1.0)

Compute the normalization constant for the likelihood ([Makarov2006](@citet) Eq. 7):

```
Z = ∫_{m_min}^{m_max} ϕ(m) dm
  = ∫_{m_min}^{m_max} ∫_{m-w}^{m+w} ψ(x)·N(m | x+bias(x), σ(x))·f(x) dx dm
```

The integration domain in `x` is the union of all inner windows:
`[m_min - int_width, m_max + int_width]`.  For each `x`, the contribution to the outer
`m`-integral comes only from `m` values in `[x - int_width, x + int_width]` intersected
with `[m_min, m_max]`.

We exploit this locality: for a fixed true magnitude `x`, the inner m-integral of the
Gaussian is known analytically (it is the difference of two CDF values).
This collapses the 2D integral to a single 1D integral over `x`:

```
Z = ∫_x ψ(x)·f(x)·[Φ(m_max | x+bias,σ) - Φ(m_min | x+bias,σ)] dx
```

where Φ is the normal CDF. This is the **optimized** form.
"""
function ϕ_norm(limits, model::LuminosityFunction, err_func, complete_func, bias_func; int_width=1.0)
    m_min, m_max = limits
    x_lo = m_min - int_width
    x_hi = m_max + int_width
    return quadgk(x -> _ϕ_norm_integrand(x, m_min, m_max, model, err_func, complete_func, bias_func),
                  x_lo, x_hi)[1]
end

# Standard normal CDF using Distributions.jl (already a dependency)
# @inline _Φ(z) = cdf(_STANDARD_NORMAL, z)
# const _STANDARD_NORMAL = Normal(0.0, 1.0)

@inline function _ϕ_norm_integrand(x, m_min, m_max, model::LuminosityFunction, err_func, complete_func, bias_func)
    σ = err_func(x)
    μ = x + bias_func(x)
    # Fraction of the Gaussian kernel that falls inside [m_min, m_max]
    # Original implementation
    # inv_σ = inv(σ)
    # Φ_hi = _Φ((m_max - μ) * inv_σ)
    # Φ_lo = _Φ((m_min - μ) * inv_σ)
    # return ψ(model, x) * complete_func(x) * (Φ_hi - Φ_lo)
    # New implementation with erf
    s = inv(σ) * invsqrt2
    Φ_hi = erf((m_max - μ) * s)
    Φ_lo = erf((m_min - μ) * s)
    return ψ(model, x) * complete_func(x) * (Φ_hi - Φ_lo) / 2
end

"""
    loglikelihood(model, mags, err_func, complete_func, bias_func; int_width=1.0)

Compute the log-likelihood of `mags` under `model` ([Makarov2006](@citet) Eq. 7):

```
log L = Σᵢ log ϕ(mᵢ) - N · log Z
```

where `Z = ∫ ϕ(m) dm` is the normalization integral over the observed magnitude range.
"""
function loglikelihood(model::LuminosityFunction, mags, err_func, complete_func, bias_func; int_width=1.0)
    if !is_valid(model)
        return -Inf
    end
    limits = extrema(mags)
    Z = ϕ_norm(limits, model, err_func, complete_func, bias_func; int_width)
    logZ = log(max(Z, eps()))
    N = length(mags)
    ll = zero(logZ)
    for m in mags
        ϕ_val = ϕ(m, model, err_func, complete_func, bias_func; int_width)
        ll += log(max(ϕ_val, eps()))
    end
    result = ll - N * logZ
    return isnan(result) ? -Inf : result
end
