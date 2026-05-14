"""
    bootstrap(method, mags; n::Int=1000, rng::AbstractRNG=Random.GLOBAL_RNG)
Perform bootstrap resampling of the unreddened magnitudes `mags` to
estimate the uncertainty of the TRGB magnitude. The `method` should be a
subtype of `AbstractEdgeDetector` (e.g. [`Sobel`](@ref), [`GLOESS`](@ref)).
Returns a vector of TRGB estimates from `n` bootstrap samples.
"""
function bootstrap(method, mags; n::Int=1000, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    trgb_estimates = Vector{Float64}(undef, n)
    sample = similar(mags) # pre-allocate sample vector
    for i in eachindex(trgb_estimates)
        Random.rand!(rng, sample, mags)  # Sample with replacement
        result = trgb(method, sample)
        trgb_estimates[i] = result.m_trgb
    end
    return trgb_estimates
end
