"""
`LuminosityFunction{T}` is the abstract supertype for luminosity functions that can be used
to model the TRGB. Subtypes should implement the following functions:
 - [`ψ`](@ref TRGBDistances.ψ) returns the value of the luminosity function at a given magnitude.
 - [`is_valid`](@ref TRGBDistances.is_valid) returns `true` if the model parameters are valid (e.g., positive slopes).
 - [`ψ_breakpoints`](@ref TRGBDistances.ψ_breakpoints) returns a tuple of magnitudes at which `ψ` has discontinuities (e.g., jumps). This is used to inform the quadrature routines to avoid straddling these points.
 - [`loglikelihood`](@ref TRGBDistances.loglikelihood) computes the log-likelihood of the observed data given the model parameters and the photometric error, completeness, and bias functions.
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
    ψ_breakpoints(model::LuminosityFunction)

Return a tuple of magnitudes at which `ψ` has a discontinuity (e.g. jumps).
These are passed as interior breakpoints to `quadgk` when computing `ϕ` and
`ϕ_norm`, so that the adaptive integrator does not straddle a jump.

The default returns an empty tuple (no discontinuities). Subtypes should
override this when `ψ` is piecewise-smooth.
"""
ψ_breakpoints(::LuminosityFunction) = ()

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
ψ_breakpoints(model::BrokenPowerLaw) = (model.m_trgb,)

# ---------------------------------------------------------------------------
# Fixed 20-point Gauss-Legendre quadrature (ForwardDiff-compatible)
# Nodes and weights on [-1, 1] with sum(weights) = 2.
# Computed once at Float64 precision via QuadGK.gauss(Float64, 20).
# ---------------------------------------------------------------------------
const _GL20_NODES = (
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513258,
    -0.8391169718222188, -0.7463319064601508, -0.636053680726515,
    -0.5108670019508271, -0.37370608871541955, -0.2277858511416451,
    -0.07652652113349732, 0.07652652113349732, 0.2277858511416451,
    0.37370608871541955, 0.5108670019508271, 0.636053680726515,
    0.7463319064601508, 0.8391169718222188, 0.9122344282513258,
    0.9639719272779138, 0.9931285991850949,
)
const _GL20_WEIGHTS = (
    0.01761400713915218, 0.04060142980038692, 0.06267204833410919,
    0.08327674157670477, 0.1019301198172404, 0.11819453196151851,
    0.1316886384491766, 0.14209610931838218, 0.14917298647260377,
    0.15275338713072564, 0.15275338713072564, 0.14917298647260377,
    0.14209610931838218, 0.1316886384491766, 0.11819453196151851,
    0.1019301198172404, 0.08327674157670477, 0.06267204833410919,
    0.04060142980038692, 0.01761400713915218,
)

"""
    _gl20_integrate(f, a, b)

Integrate `f` over `[a, b]` using a fixed 20-point Gauss-Legendre rule.
Unlike `quadgk`, this uses pre-computed `Float64` nodes and weights, so it is
compatible with ForwardDiff automatic differentiation.
"""
@inline function _gl20_integrate(f, a, b)
    mid = (a + b) / 2
    halflen = (b - a) / 2
    result = zero(typeof(f(mid)))  # correct type for AD
    for (t, w) in zip(_GL20_NODES, _GL20_WEIGHTS)
        result += w * f(muladd(halflen, t, mid))
    end
    return halflen * result
end

"""
    _gl20_integrate_segments(f, pts...)

Integrate `f` piecewise over the breakpoint-separated intervals defined by `pts`
using the fixed 20-point GL rule on each sub-interval.  `pts` must contain at
least two values (the endpoints); any additional values are interior breakpoints.
"""
function _gl20_integrate_segments(f, pts...)
    s = length(pts)
    result = _gl20_integrate(f, pts[1], pts[2])
    for i in 2:(s-1)
        result += _gl20_integrate(f, pts[i], pts[i+1])
    end
    return result
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
function ϕ(m, model::LuminosityFunction, err_func, complete_func, bias_func; int_width=1.0, quad=:adaptive)
    lo = m - int_width
    hi = m + int_width
    # Insert any discontinuity breakpoints that fall strictly inside [lo, hi]
    # so that the integrator treats each smooth sub-interval separately.
    bps = [x for x in ψ_breakpoints(model) if lo < x < hi]
    if quad === :fixed
        return _ϕ_fixed(m, lo, hi, bps, model, err_func, complete_func, bias_func)
    else
        return quadgk(x -> _ϕ_integrand(x, m, model, err_func, complete_func, bias_func),
                      lo, bps..., hi)[1]
    end
end

# Fixed GL quadrature for ϕ.
# The integrand is a narrow Gaussian in x centered near m with width σ(m).
# A single 20-point rule over [m-int_width, m+int_width] misses the peak when σ
# is small compared to int_width.  We therefore cluster quadrature points around
# the peak by splitting [lo, hi] into three sub-regions:
#   1. [lo, peak_lo]   — left tail, essentially zero; 20 GL points
#   2. [peak_lo, peak_hi] — core ±k·σ window; 32 uniform sub-intervals × 20 GL pts
#   3. [peak_hi, hi]   — right tail; 20 GL points
# σ and m are Float64 constants even during ForwardDiff differentiation (they come
# from the observed data and err_func, not from the model parameters), so this
# segmentation is always computable as plain Float64 arithmetic.
function _ϕ_fixed(m::Real, lo::Real, hi::Real, bps, model, err_func, complete_func, bias_func)
    σ_peak = err_func(Float64(m))
    k = 5  # ±5σ covers >99.9999% of a Gaussian
    peak_lo = Float64(m) - k * σ_peak
    peak_hi = Float64(m) + k * σ_peak
    peak_lo = max(peak_lo, lo)
    peak_hi = min(peak_hi, hi)

    f = x -> _ϕ_integrand(x, m, model, err_func, complete_func, bias_func)
    # Number of inner sub-intervals: make each ~σ/2 wide
    n_inner = max(4, round(Int, (peak_hi - peak_lo) / (σ_peak / 2)))

    # Collect all breakpoints that fall in each region and integrate
    result = zero(ψ(model, Float64(m)))  # correct Dual/Float type

    # Left tail [lo, peak_lo]
    if peak_lo > lo
        left_bps = [x for x in bps if lo < x < peak_lo]
        result += _gl20_integrate_segments(f, lo, left_bps..., peak_lo)
    end

    # Core region: split uniformly, inserting breakpoints as sub-interval boundaries
    core_bps = sort([x for x in bps if peak_lo <= x <= peak_hi])
    core_pts = [peak_lo; core_bps; peak_hi]
    for seg_idx in 1:(length(core_pts) - 1)
        seg_lo = core_pts[seg_idx]
        seg_hi = core_pts[seg_idx + 1]
        seg_n = max(2, round(Int, n_inner * (seg_hi - seg_lo) / (peak_hi - peak_lo)))
        seg_w = (seg_hi - seg_lo) / seg_n
        for i in 1:seg_n
            result += _gl20_integrate(f, seg_lo + (i-1)*seg_w, seg_lo + i*seg_w)
        end
    end

    # Right tail [peak_hi, hi]
    if peak_hi < hi
        right_bps = [x for x in bps if peak_hi < x < hi]
        result += _gl20_integrate_segments(f, peak_hi, right_bps..., hi)
    end

    return result
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
function ϕ_norm(limits, model::LuminosityFunction, err_func, complete_func, bias_func; int_width=1.0, quad=:adaptive)
    m_min, m_max = limits
    x_lo = m_min - int_width
    x_hi = m_max + int_width
    # Insert any discontinuity breakpoints that fall strictly inside [x_lo, x_hi]
    bps = [x for x in ψ_breakpoints(model) if x_lo < x < x_hi]
    if quad === :fixed
        return _ϕ_norm_fixed(x_lo, x_hi, bps, m_min, m_max, model, err_func, complete_func, bias_func)
    else
        return quadgk(x -> _ϕ_norm_integrand(x, m_min, m_max, model, err_func, complete_func, bias_func),
                      x_lo, bps..., x_hi)[1]
    end
end

# Fixed GL quadrature for ϕ_norm.
# The ψ integrand is a smooth power law within each breakpoint-separated piece.
# A single 20-point GL rule over a multi-magnitude range is insufficiently accurate
# because the power-law integrand varies rapidly.  We split each piece into
# sub-intervals of at most 0.5 mag to ensure ~1e-8 relative accuracy.
const _GL_NORM_SEG_WIDTH = 0.5  # max sub-interval width in magnitudes

function _ϕ_norm_fixed(x_lo, x_hi, bps, m_min, m_max, model, err_func, complete_func, bias_func)
    f = x -> _ϕ_norm_integrand(x, m_min, m_max, model, err_func, complete_func, bias_func)
    pts = [x_lo; bps; x_hi]
    result = zero(ψ(model, (x_lo + x_hi) / 2))
    for i in 1:(length(pts) - 1)
        seg_lo = pts[i]
        seg_hi = pts[i + 1]
        n_segs = max(1, ceil(Int, (seg_hi - seg_lo) / _GL_NORM_SEG_WIDTH))
        seg_w = (seg_hi - seg_lo) / n_segs
        for j in 1:n_segs
            result += _gl20_integrate(f, seg_lo + (j-1)*seg_w, seg_lo + j*seg_w)
        end
    end
    return result
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
function loglikelihood(model::LuminosityFunction, mags, err_func, complete_func, bias_func; int_width=1.0, quad=:adaptive)
    if !is_valid(model)
        return -Inf
    end
    limits = extrema(mags)
    Z = ϕ_norm(limits, model, err_func, complete_func, bias_func; int_width, quad)
    logZ = log(max(Z, eps()))
    N = length(mags)
    ll = zero(logZ)
    for m in mags
        ϕ_val = ϕ(m, model, err_func, complete_func, bias_func; int_width, quad)
        ll += log(max(ϕ_val, eps()))
    end
    result = ll - N * logZ
    return isnan(result) ? -Inf : result
end
