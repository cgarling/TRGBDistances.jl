module TRGBDistances

using Distributions: pdf, logpdf, Normal
using QuadGK: quadgk
using Interpolations: linear_interpolation

include("calibrations.jl")

abstract type LuminosityFunction{T} end
(l::LuminosityFunction)(m) = ψ(l, m)

struct BrokenPowerLaw{T} <: LuminosityFunction{T}
    m_trgb::T
    a::T
    b::T
    c::T
end
BrokenPowerLaw(m_trgb, a, b, c) = BrokenPowerLaw(promote(m_trgb, a, b, c)...)
ψ(b::BrokenPowerLaw, m) = m >= b.m_trgb ? exp10(b.a * (m - b.m_trgb) + b.b) : exp10(b.c * (m - b.m_trgb))

# Computes Equation 4 of Makarov 2006
function ϕ(m, b::LuminosityFunction, err_func, complete_func, bias_func)
    # σ = err_func(m)
    # # Defining bias_func(m) = <measured> - <input>, so \bar{m}(m^\prime) = m^\prime + bias_func(m)
    # mp = m + bias_func(m)
    # return quadgk(x -> ψ(b, m) * pdf(Normal(mp, σ), m) * complete_func(m), mp - 10σ, mp + 10σ)[1]
    # Defining bias_func(m) = <measured> - <input>, so \bar{m}(m^\prime) = m^\prime + bias_func(m)
    return quadgk(x -> ψ(b, x) * pdf(Normal(x + bias_func(x), err_func(x)), m) * complete_func(x), m - 1, m + 1)[1]
    # return quadgk(x -> ψ(b, x) * pdf(Normal(m, err_func(x)), x + bias_func(x)) * complete_func(x), m - 1, m + 1)[1]
end

struct LuminosityFunctionData{T,A,B,C,D}
    model::T # Can be a LuminosityFunction, or an alternative constructor (for fixed values)
    mags::A
    err_func::B
    complete_func::C
    bias_func::D
end

# Negative log likelihood
function (fit::LuminosityFunctionData)(θ)
    # θ = exp.(θ)
    # println(θ)
    if mapreduce(x -> sign(x) == -1, |, θ)
        return Inf
    end
    model = fit.model(θ...)
    return -sum(log(ϕ(m, model, fit.err_func, fit.complete_func, fit.bias_func)) for m in fit.mags) + length(fit.mags) * log(quadgk(x -> ϕ(x, model, fit.err_func, fit.complete_func, fit.bias_func), extrema(fit.mags)...)[1])
end


function filter_mags(colors, mags, ridge_colors, ridge_mags, func)
    pidx = sortperm(ridge_mags)
    ridge_colors, ridge_mags = ridge_colors[pidx], ridge_mags[pidx]
    color_itp = linear_interpolation(ridge_mags, ridge_colors; extrapolation_bc=Inf)
    idx = Int64[]
    for i in eachindex(colors, mags)
        mag = mags[i]
        width = func(mag)
        icolor = color_itp(mag)
        if (icolor - width <= colors[i] <= icolor + width) && isfinite(icolor)
            push!(idx, i)
        end
    end
    return idx
end


end # module
