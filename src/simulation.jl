"""
    cdf_bpl(model::BrokenPowerLaw, m)

Evaluate the (unnormalized then normalized) cumulative distribution function of the
`BrokenPowerLaw` luminosity function at magnitude `m`.

For a `BrokenPowerLaw` with parameters `(m_trgb, a, b, c)`:

- For `m < m_trgb` (AGB side): `CDF(m) = ∫_{-∞}^{m} 10^(c*(x-m_trgb)) dx = 10^(c*(m-m_trgb)) / (c * log(10))`
- For `m >= m_trgb` (RGB side): `CDF(m) = CDF(m_trgb) + ∫_{m_trgb}^{m} 10^(a*(x-m_trgb)+b) dx`

Returns the *normalized* CDF value in [0, 1].
"""
function cdf_bpl(model::BrokenPowerLaw, m)
    return _cdf_unnorm(model, m) / _cdf_unnorm_total(model)
end

# Integral of ψ from -∞ to m
function _cdf_unnorm(model::BrokenPowerLaw, m)
    (; m_trgb, a, b, c) = model
    if m < m_trgb
        # AGB side: ∫_{-∞}^{m} 10^(c*(x-m_trgb)) dx  (converges only if c > 0)
        # does not converge for c ≤ 0 (handle gracefully)
        c ≤ 0 ? zero(m) : exp10(c * (m - m_trgb)) / (c * logten)
    else
        # AGB contribution (up to m_trgb from -∞)
        agb_part = c > 0 ? inv(c * logten) : zero(m)
        # RGB contribution (m_trgb to m)
        if a ≤ 0
            rgb_part = zero(m)  # degenerate; handle gracefully
        else
            rgb_part = exp10(b) * (exp10(a * (m - m_trgb)) - 1) / (a * logten)
        end
        return agb_part + rgb_part
    end
end

# Total integral of ψ from -∞ to +∞ (converges only if a < 0 on the RGB)
# For typical usage where a > 0 (rising RGB), we integrate only over a finite range
# We instead use an upper limit of m_trgb + upper_limit for normalization.
function _cdf_unnorm_total(model::BrokenPowerLaw; upper_limit=10.0)
    return _cdf_unnorm(model, model.m_trgb + upper_limit)
end

"""
    quantile(model::BrokenPowerLaw, p; upper_limit=10.0)

Compute the inverse CDF (quantile function) for magnitude `p ∈ [0, 1]`.
Uses the analytic inverse CDF derived from the broken power-law form.
The `upper_limit` parameter controls the effective upper bound for
normalization and sampling such that the maximum magnitude considered for
normalization and sampling is `m_trgb + upper_limit`).
"""
function quantile(model::BrokenPowerLaw, p; upper_limit=10.0)
    (; m_trgb, a, b, c) = model

    total = _cdf_unnorm_total(model; upper_limit)
    target = p * total

    # IMPORTANT: use SAME CDF function for boundary
    agb_total = _cdf_unnorm(model, m_trgb)

    if target <= agb_total
        # AGB inversion (relative to model definition)
        if c ≤ 0
            return m_trgb
        end

        return m_trgb + log10(target * c * logten) / c
    else
        # RGB inversion
        if a ≤ 0
            return m_trgb
        end

        rhs = target - agb_total
        inner = 1 + rhs * a * logten / exp10(b)

        # safety clamp (prevents NaNs if numerical edge)
        inner = max(inner, 0.0)

        return m_trgb + log10(inner) / a
    end
end
# function quantile(model::BrokenPowerLaw, p; upper_limit=10.0)
#     (; m_trgb, a, b, c) = model
#     total = _cdf_unnorm_total(model; upper_limit)
#     p_unnorm = p * total  # target unnormalized CDF value

#     # Contribution from AGB side
#     agb_total = c > 0 ? inv(c * logten) : zero(p)

#     if p_unnorm <= agb_total
#         # On AGB side: p_unnorm = 10^(c*(m - m_trgb)) / (c * logten)
#         # ⟹ m = m_trgb + log10(p_unnorm * c * logten) / c
#         return m_trgb + log10(p_unnorm * c * logten) / c
#     else
#         # On RGB side: p_unnorm = agb_total + 10^b * (10^(a*(m-m_trgb)) - 1) / (a * logten)
#         # ⟹ 10^(a*(m-m_trgb)) = 1 + (p_unnorm - agb_total) * a * logten / 10^b
#         # ⟹ m = m_trgb + log10(1 + (p_unnorm - agb_total) * a * logten / 10^b) / a
#         if a ≤ 0
#             return m_trgb  # degenerate case
#         end
#         inner = 1 + (p_unnorm - agb_total) * a * logten / exp10(b)
#         return m_trgb + log10(max(inner, 0.0)) / a
#     end
# end

# -----------------------------------------------------------------------
# Random sampling via inverse-CDF
# -----------------------------------------------------------------------

"""
    Random.rand(rng::AbstractRNG, model::BrokenPowerLaw; upper_limit=10.0)
    Random.rand(rng::AbstractRNG, model::BrokenPowerLaw, dims::Dims; upper_limit=10.0)
    Random.rand(rng::AbstractRNG, model::BrokenPowerLaw, dims...; upper_limit=10.0)

Draw magnitudes from the `BrokenPowerLaw` luminosity function using inverse-CDF
sampling. The `upper_limit` parameter controls the effective upper bound for
normalization and sampling such that the maximum magnitude considered for
normalization and sampling is `m_trgb + upper_limit`).
"""
function Random.rand(rng::Random.AbstractRNG, model::BrokenPowerLaw; upper_limit=10.0)
    return quantile(model, Random.rand(rng); upper_limit)
end

function Random.rand(rng::Random.AbstractRNG, model::BrokenPowerLaw, dims::Dims; upper_limit=10.0)
    return [quantile(model, rand(rng); upper_limit) for _ in CartesianIndices(dims)] |> x -> reshape(x, dims)
end

function Random.rand(rng::Random.AbstractRNG, model::BrokenPowerLaw, dims::Integer...; upper_limit=10.0)
    return rand(rng, model, Tuple(dims); upper_limit)
end

# -----------------------------------------------------------------------
# Observational realization
# -----------------------------------------------------------------------

"""
    observe([rng,] model::LuminosityFunction, n::Integer; err_func, complete_func, bias_func, upper_limit)

Generate a synthetic catalog of `n` stars drawn from `model`, applying photometric
incompleteness, bias, and scatter. Sampled stars will have magnitudes no larger than `m_trgb + upper_limit`.

Stars are:
1. Drawn from the luminosity function (depends on implemented `rand` methods for `model`).
2. Rejected with probability `1 - complete_func(m)` (incompleteness).
3. Shifted by `bias_func(m)` and scattered with standard deviation `err_func(m)`.

Returns a `Vector{Float64}` of observed magnitudes.

Requires a defined method for `rand(rng, model; upper_limit)` that samples from the luminosity function.

# Example
```jldoctest
julia> err(m)  = 0.05 + 0.1 * (m - 24.0)^2;

julia> compl(m) = m < 26.0 ? 1.0 : 0.0;

julia> bias(m) = 0.0;

julia> mags = observe(BrokenPowerLaw(24.0, 0.3, 0.2, 0.1), 1000; 
                      err_func=err, complete_func=compl, bias_func=bias, upper_limit=10.0);

julia> mags isa Vector{Float64}
true
```
"""
function observe(rng::Random.AbstractRNG, model::LuminosityFunction, n::Integer;
                 err_func, complete_func, bias_func, kws...)
    observed = Float64[]
    sizehint!(observed, n)
    while length(observed) < n
        m_true = rand(rng, model; kws...)
        # Acceptance-rejection for completeness
        if Random.rand(rng) > complete_func(m_true)
            continue
        end
        # Apply bias and scatter
        m_obs = m_true + bias_func(m_true) + err_func(m_true) * Random.randn(rng)
        push!(observed, m_obs)
    end
    return observed
end

function observe(model::LuminosityFunction, n::Integer; kwargs...)
    return observe(Random.default_rng(), model, n; kwargs...)
end
