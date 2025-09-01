module TRGBDistances

using Distributions: pdf, logpdf, Normal
using QuadGK: quadgk
using Trapz: trapz
using HCubature: hcubature
using Interpolations: linear_interpolation

include("calibrations.jl")

abstract type LuminosityFunction{T} end
(l::LuminosityFunction)(m) = ψ(l, m)
Base.Broadcast.broadcastable(t::LuminosityFunction) = Ref(t)


struct BrokenPowerLaw{T} <: LuminosityFunction{T}
    m_trgb::T
    a::T
    b::T
    c::T
end
BrokenPowerLaw(m_trgb, a, b, c) = BrokenPowerLaw(promote(m_trgb, a, b, c)...)
ψ(b::BrokenPowerLaw, m) = m >= b.m_trgb ? exp10(b.a * (m - b.m_trgb) + b.b) : exp10(b.c * (m - b.m_trgb))

# Computing Equation 4 of Makarov 2006
@inline function ϕ_integrand(x, m, b::LuminosityFunction, err_func, complete_func, bias_func)
    return ψ(b, x) * pdf(Normal(x + bias_func(x), err_func(x)), m) * complete_func(x)
end
function ϕ(m, b::LuminosityFunction, err_func, complete_func, bias_func)
    # # Defining bias_func(m) = <measured> - <input>, so \bar{m}(m^\prime) = m^\prime + bias_func(m)
    # x = range(m-1, m+1; length=1000)
    # return trapz(x, ϕ_integrand.(x, m, b, Ref(err_func), Ref(complete_func), Ref(bias_func)))
    return quadgk(x -> ϕ_integrand(x, m, b, err_func, complete_func, bias_func), m - 1, m + 1)[1]
end
# Double integral that appears in Makarov 2006 Equation 7; transform variables to rectangular integration region
function transformed_ϕ_integrand(uv, b::LuminosityFunction, err_func, complete_func, bias_func)
    u, v = uv
    x = v + 2u - 1 # 2v + u - 1
    return 2 * ϕ_integrand(x, v, b, err_func, complete_func, bias_func)
end
function ϕ2(limits, b::LuminosityFunction, err_func, complete_func, bias_func)
    return quadgk(x -> ϕ(x, b, err_func, complete_func, bias_func), limits...)[1]
    # return hcubature(uv -> transformed_ϕ_integrand(uv, b, err_func, complete_func, bias_func), [0.0, limits[1]], [1.0, limits[2]])[1]
    # x = 0.0:0.001:1.0
    # y = range(limits...; length=1000)
    # return trapz((x,y), [transformed_ϕ_integrand([i,j], b, err_func, complete_func, bias_func) for i=x,j=y])
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
    # return -sum(log(ϕ(m, model, fit.err_func, fit.complete_func, fit.bias_func)) for m in fit.mags) + length(fit.mags) * log(quadgk(x -> ϕ(x, model, fit.err_func, fit.complete_func, fit.bias_func), extrema(fit.mags)...)[1])
    # return quadgk(x -> ϕ(x, model, fit.err_func, fit.complete_func, fit.bias_func), extrema(fit.mags)...)[1]
    # return ϕ2(extrema(fit.mags), model, fit.err_func, fit.complete_func, fit.bias_func)
    return -sum(log(ϕ(m, model, fit.err_func, fit.complete_func, fit.bias_func)) for m in fit.mags) + length(fit.mags) * log(ϕ2(extrema(fit.mags), model, fit.err_func, fit.complete_func, fit.bias_func))
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
