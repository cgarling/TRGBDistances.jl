module TRGBDistances

import ADTypes
using Distributions: pdf, logpdf, cdf, Normal
using QuadGK: quadgk
using Interpolations: linear_interpolation
using SpecialFunctions: erf
using IrrationalConstants: sqrt2π, invsqrt2π, invsqrt2, logten
import Random

include("calibrations.jl")
include("utilities.jl")
include("luminosity_functions/luminosity_function.jl")
include("luminosity_functions/fitting.jl")
include("luminosity_functions/sampling.jl")

"""
Abstract type for edge detection methods. Not intended for direct use.
"""
abstract type AbstractEdgeDetector end

"""
    trgb(method, mags)
Estimate the TRGB magnitude from a vector of dereddened apparent magnitudes 
`mags` using the specified edge detection `method`. The method should
be a subtype of `AbstractEdgeDetector`,
e.g. [`Sobel`](@ref), [`GLOESS`](@ref) -- see the docs of these
types for usage examples. Returns a result struct specific to
the method (e.g. [`SobelResult`](@ref), [`GLOESSResult`](@ref)).
"""
function trgb end

include("edge_detection/sobel.jl")
include("edge_detection/gloess.jl")
include("edge_detection/bootstrap.jl")
include("simulation.jl")

export BrokenPowerLaw, fit, sample, observe, filter_mags, Sobel, GLOESS, bootstrap, trgb, Rizzi2007, Fusco2012
export OptimJL, KissMCMCJL, DynamicHMCJL, AdvancedMHJL, AffineInvariantMCMCJL # export backends

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
