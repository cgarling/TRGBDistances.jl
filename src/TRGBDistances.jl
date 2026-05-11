module TRGBDistances

import ADTypes
using Distributions: logpdf, cdf, Normal
using QuadGK: quadgk
using Interpolations: linear_interpolation
using SpecialFunctions: erf
using IrrationalConstants: sqrt2π, invsqrt2π, invsqrt2, logten
import Random

include("calibrations.jl")
include("utilities.jl")
include("luminosity_function.jl")
include("fitting.jl")
include("sampling.jl")
include("simulation.jl")

export LuminosityFunction, ψ
export BrokenPowerLaw
export ϕ, ϕ_norm, loglikelihood
export TRGBFitResult, fit, OptimJL
export AbstractOptimizerBackend, AbstractSamplerBackend
export TRGBChain, sample, KissMCMCJL, DynamicHMCJL
export build_gradient, build_hessian
export logprior
export observe, cdf_bpl
export filter_mags

"""
    filter_mags(colors, mags, ridge_colors, ridge_mags, func)

Returns indices into `colors` and `mags` of stars that fall within range of a ridgeline
defined by `ridge_colors` and `ridge_mags`. The selection region is defined by the callable
`func(m)` for magnitude `m` which must return a width. A star is accepted if its color falls
between `± func(m)` of the ridgeline.
"""
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
