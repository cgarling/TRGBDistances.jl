# ---------------------------------------------------------------------------
# Sobel edge-detection filter for TRGB magnitude estimation.
#
# Primary references:
#   Lee, Freedman & Madore 1993 [Lee1993] — original Sobel TRGB method.
#   Sakai, Madore & Freedman 1996 [Sakai1996] — Gaussian pre-smoothing + Sobel.
# ---------------------------------------------------------------------------

"""
    SobelResult

Result returned by [`sobel_trgb`](@ref).

Fields:
- `m_trgb`: Estimated TRGB apparent magnitude (bin center of peak edge signal).
- `edge_signal`: Sobel edge signal at each bin center (length = number of bins).
- `bin_centers`: Bin centers of the magnitude histogram.
- `counts`: Raw integer star counts per bin.
"""
struct SobelResult
    m_trgb::Float64
    edge_signal::Vector{Float64}
    bin_centers::Vector{Float64}
    counts::Vector{Int}
end

Base.show(io::IO, r::SobelResult) =
    print(io, "SobelResult(m_trgb=$(r.m_trgb))")

"""
    sobel_trgb(mags; bin_width=0.1, response=nothing, magnitude_range=nothing)

Estimate the TRGB apparent magnitude from a vector of observed stellar magnitudes
using a Sobel edge-detection filter applied to the stellar luminosity function
[Lee1993](@cite).

The stars are binned into a histogram of width `bin_width` (in magnitudes).
If `response` is a [`Distributions.UnivariateDistribution`](https://juliastats.org/Distributions.jl/stable/univariate/),
the histogram is first convolved with the distribution's probability density
function before the Sobel kernel is applied.  This generalised pre-smoothing,
described in [Sakai1996](@cite), acts as a matched filter for a TRGB edge
smeared by photometric errors; a `Normal(0, σ)` with `σ` equal to the typical
photometric error near the TRGB is a natural choice.  With `response=nothing`
(the default) a pure Sobel kernel ``[-1, 0, +1]`` is applied to the raw
histogram counts, which corresponds to a Delta-function edge response and is
appropriate when photometric errors are small compared to `bin_width`.

The TRGB is identified as the bin with the *largest* edge signal — the point
where star counts rise most sharply when going from bright to faint (i.e. the
discontinuous step at the TRGB in the RGB luminosity function).

!!! note "Edge detection vs. likelihood fitting"
    The Sobel method is fast and non-parametric: it makes no assumptions about
    the shape of the luminosity function.  It does not produce a formal
    uncertainty estimate. Uncertainties may be estimated by bootstrapping.

# Arguments
- `mags`: Vector of dereddened apparent magnitudes.

# Keyword Arguments
- `bin_width`: Magnitude bin width in magnitudes (default `0.1`).
- `response`: Optional `Distributions.UnivariateDistribution` for
  pre-smoothing the histogram before applying the Sobel kernel.  `nothing`
  (default) skips pre-smoothing.
- `magnitude_range`: Optional `(m_min, m_max)` tuple to restrict the
  magnitude range used for binning.  Defaults to the full data range padded
  by one `bin_width` on each side.

# Returns
A [`SobelResult`](@ref TRGBDistances.SobelResult).

# Examples
```jldoctest
julia> using TRGBDistances

julia> using StableRNGs: StableRNG

julia> model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1);

julia> mags = observe(StableRNG(1), model, 500; err_func=m->0.05, complete_func=m->1.0, bias_func=m->0.0, upper_limit=2.0);

julia> result = sobel_trgb(mags; bin_width=0.1, magnitude_range=(23.5, 24.5));

julia> result isa TRGBDistances.SobelResult
true

julia> abs(result.m_trgb - 24.0) < 0.1 # Close to true value
true
```
"""
function sobel_trgb(mags; bin_width=0.1, response=nothing, magnitude_range=nothing)
    m_min, m_max = if magnitude_range === nothing
        Float64(minimum(mags) - bin_width / 2), Float64(maximum(mags) + bin_width / 2)
    else
        Float64(magnitude_range[1]), Float64(magnitude_range[2])
    end

    # Build bin edges and centres
    n_bins = max(3, ceil(Int, (m_max - m_min) / bin_width))
    bin_centers = [m_min + (i - 0.5) * bin_width for i in 1:n_bins]

    # Populate histogram
    counts = zeros(Int, n_bins)
    for m in mags
        idx = floor(Int, (m - m_min) / bin_width) + 1
        if 1 <= idx <= n_bins
            counts[idx] += 1
        end
    end

    # Optional pre-smoothing with response distribution
    data = if response !== nothing
        _convolve_histogram(counts, bin_centers, response)
    else
        counts
    end

    # Apply Sobel kernel [-1, 0, +1] — edge signal at each interior bin
    edge = zeros(n_bins)
    for i in 2:(n_bins - 1)
        edge[i] = data[i + 1] - data[i - 1]
    end

    # TRGB = bin with largest positive edge signal
    # (the RGB luminosity function rises steeply toward fainter magnitudes, so
    #  the transition from AGB to RGB produces the largest positive gradient)
    m_trgb_idx = argmax(@view edge[2:end-1]) + 1   # +1: offset for excluded first bin
    m_trgb = bin_centers[m_trgb_idx]

    return SobelResult(m_trgb, edge, bin_centers, counts)
end

"""
    _convolve_histogram(counts, bin_centers, d) -> Vector{Float64}

Compute the kernel-smoothed histogram

```
out[i] = Σ_j  counts[j] · pdf(d,  bin_centers[i] − bin_centers[j])
```

This is a discrete convolution of `counts` with the probability density
function of `d` evaluated on the (uniform) bin-centre grid.  The result
is *not* renormalized, so the overall scale is preserved.
"""
function _convolve_histogram(counts::AbstractVector,
                             bin_centers::AbstractVector,
                             d)
    T = promote_type(eltype(counts), eltype(bin_centers), eltype(pdf(d, zero(eltype(bin_centers)))))
    n = length(bin_centers)
    out = zeros(T, n)
    for i in 1:n
        for j in 1:n
            out[i] += counts[j] * pdf(d, bin_centers[i] - bin_centers[j])
        end
    end
    return out
end
