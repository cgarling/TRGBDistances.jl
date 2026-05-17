# ---------------------------------------------------------------------------
# GLOESS (Gaussian-windowed Locally-Weighted Scatterplot Smoothing) smoothing
# combined with Sobel edge detection for TRGB magnitude estimation.
#
# Primary references:
#   Persson et al. 2004 [Persson2004] — definition of the GLOESS algorithm.
#   Madore & Freedman 2009 [Hatt2017] — GLOESS applied to TRGB detection.
# ---------------------------------------------------------------------------

"""
    GLOESSResult

Result returned by [`trgb(GLOESS(...), mags)`](@ref TRGBDistances.trgb).

Fields:
- `m_trgb`: Estimated TRGB apparent magnitude (bin center of peak edge signal
  after GLOESS smoothing).
- `edge_signal`: Sobel edge signal applied to the GLOESS-smoothed histogram.
- `smoothed_counts`: GLOESS-smoothed counts at each bin center.
- `bin_centers`: Bin centers of the magnitude histogram.
- `counts`: Raw integer star counts per bin.
"""
struct GLOESSResult
    m_trgb::Float64
    edge_signal::Vector{Float64}
    smoothed_counts::Vector{Float64}
    bin_centers::Vector{Float64}
    counts::Vector{Int}
end

Base.show(io::IO, r::GLOESSResult) =
    print(io, "GLOESSResult(m_trgb=$(r.m_trgb))")

"""
    GLOESS(; bandwidth=0.2, bin_width=0.1, magnitude_range=nothing)

GLOESS estimator for the TRGB apparent magnitude given a vector
of observed stellar magnitudes `mags`. Uses GLOESS smoothing
[Persson2004](@cite) of the magnitude histogram followed by
Sobel edge detection [Lee1993](@cite), as applied to TRGB
measurement in [Hatt2017](@cite). To compute result,
use [`trgb(GLOESS(; ...), mags)`](@ref TRGBDistances.trgb),
returning a [`GLOESSResult`](@ref TRGBDistances.GLOESSResult).

The algorithm proceeds in three steps:

1. **Bin** the magnitudes into a histogram with bin width `bin_width`.
2. **Smooth** the histogram using the [`gloess_smooth`](@ref TRGBDistances.gloess_smooth) kernel with the
   specified `bandwidth`.  The Gaussian kernel suppresses Poisson noise in the
   luminosity function while preserving the sharp edge at the TRGB.
3. **Detect** the edge: apply the Sobel kernel ``[-1, 0, +1]`` to the smoothed
   histogram.  The TRGB is at the bin with the largest (most positive) edge
   signal, where star counts rise most sharply going from bright to faint.

!!! note "Bandwidth selection"
    The `bandwidth` controls the degree of smoothing.  Too small a bandwidth
    leaves Poisson noise visible; too large a bandwidth washes out the edge.
    A value in the range ``0.1``–``0.3`` mag is typical for well-sampled data.
    An initial guess of ``\\sigma_{\\text{phot}} \\lesssim h \\lesssim 5\\,
    \\sigma_{\\text{phot}}`` (where ``\\sigma_{\\text{phot}}`` is the typical
    photometric error near the TRGB) is a good starting point.

!!! note "Edge detection vs. likelihood fitting"
    Like the plain Sobel filter, GLOESS+Sobel is fast and non-parametric but
    does not produce a formal uncertainty.  Use [`fit`](@ref) for MAP/Bayesian
    inference with uncertainty estimation. Uncertainties may be estimated by bootstrapping.

# Keyword Arguments
- `bandwidth`: GLOESS Gaussian kernel bandwidth in magnitudes (default `0.2`).
- `bin_width`: Magnitude bin width in magnitudes (default `0.1`).
- `magnitude_range`: Optional `(m_min, m_max)` tuple to restrict the
  magnitude range.  Defaults to the full data range padded by one `bin_width`.

# Examples
```jldoctest
julia> using TRGBDistances

julia> using StableRNGs: StableRNG

julia> model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1);

julia> mags = observe(StableRNG(1), model, 2500; err_func=m->0.05, complete_func=m->1.0, bias_func=m->0.0, upper_limit=2.0);

julia> result = trgb(GLOESS(bandwidth=0.2, bin_width=0.1, magnitude_range=(23.5, 24.8)), mags);

julia> result isa TRGBDistances.GLOESSResult
true

julia> abs(result.m_trgb - 24.0) < 0.5
true
```
"""
Base.@kwdef struct GLOESS{T} <: AbstractEdgeDetector
    bandwidth::T = 0.2
    bin_width::T = 0.1
    magnitude_range::Union{Nothing, Tuple{T,T}} = nothing

    function GLOESS(bandwidth, bin_width, magnitude_range::Nothing)
        T_type = promote_type(typeof(bandwidth), typeof(bin_width))
        new{T_type}(convert(T_type, bandwidth), convert(T_type, bin_width), nothing)
    end

    function GLOESS(bandwidth, bin_width, magnitude_range::Tuple)
        T_type = promote_type(typeof(bandwidth), typeof(bin_width), typeof(magnitude_range[1]), typeof(magnitude_range[2]))
        new{T_type}(convert(T_type, bandwidth),
                     convert(T_type, bin_width),
                     (convert(T_type, magnitude_range[1]), convert(T_type, magnitude_range[2])))
    end
end

Base.show(io::IO, s::GLOESS) =
    print(io, "GLOESS(bandwidth=$(s.bandwidth), bin_width=$(s.bin_width), magnitude_range=$(s.magnitude_range))")

"""
    gloess_smooth(counts, bin_centers; bandwidth=0.2) -> Vector{Float64}

Apply GLOESS (Gaussian-windowed Locally-Weighted Scatterplot Smoothing)
smoothing [Persson2004](@cite) to a binned luminosity function.

For each output point ``m_i``, the smoothed value is the Nadaraya–Watson
kernel estimate

```math
\\hat{n}(m_i) = \\frac{\\sum_j n_j \\, G(m_i - m_j;\\, h)}{\\sum_j G(m_i - m_j;\\, h)},
```

where ``G(\\Delta m; h) = \\exp\\!\\bigl(-\\Delta m^2 / (2h^2)\\bigr)`` is a
Gaussian kernel with bandwidth ``h`` and ``n_j`` are the bin counts.

# Arguments
- `counts`: Star counts per bin (integer or float vector).
- `bin_centers`: Bin-center magnitudes (same length as `counts`).

# Keyword Arguments
- `bandwidth`: Gaussian kernel bandwidth in magnitudes (default `0.2`).

# Returns
A `Vector{Float64}` of smoothed counts.

# Examples
```jldoctest
julia> using TRGBDistances: gloess_smooth

julia> bc = collect(22.0:0.1:26.0);

julia> n  = [i > 20 ? 10.0 : 1.0 for i in eachindex(bc)];  # step at index 20

julia> sm = gloess_smooth(n, bc; bandwidth=0.2);

julia> length(sm) == length(bc)
true

julia> sm[1] < sm[end]   # smoothed LF still rises toward faint end
true
```
"""
function gloess_smooth(counts::AbstractVector, bin_centers::AbstractVector; bandwidth=0.2)
    n = length(bin_centers)
    smoothed = zeros(Float64, n)
    h2 = 2 * bandwidth^2
    # Truncate the kernel at 6σ to avoid unnecessary computation in the tails
    trunc = max(1, ceil(Int, 6 * bandwidth / (bin_centers[2] - bin_centers[1])))
    for i in 1:n
        wsum = 0.0
        nsum = 0.0
        for j in max(1, i-trunc):min(n, i+trunc)
            w = exp(-(bin_centers[i] - bin_centers[j])^2 / h2)
            wsum += w
            nsum += w * counts[j]
        end
        smoothed[i] = wsum > 0 ? nsum / wsum : 0.0
    end
    return smoothed
end

@inline function trgb(op::GLOESS, mags)
    return gloess_trgb(mags; bandwidth=op.bandwidth, bin_width=op.bin_width, magnitude_range=op.magnitude_range)
end

# Implements the GLOESS + Sobel edge detection algorithm. Called by `trgb(::GLOESS, mags)`.
function gloess_trgb(mags; bandwidth=0.2, bin_width=0.1, magnitude_range=nothing)
    m_min, m_max = if magnitude_range === nothing
        Float64(minimum(mags) - bin_width / 2), Float64(maximum(mags) + bin_width / 2)
    else
        Float64(magnitude_range[1]), Float64(magnitude_range[2])
    end

    counts, bin_centers = _build_histogram(mags, m_min, m_max, bin_width)
    n_bins = length(bin_centers)

    # GLOESS smoothing
    smoothed = gloess_smooth(counts, bin_centers; bandwidth)

    # Apply Sobel kernel [-1, 0, +1] to smoothed histogram
    edge = zeros(n_bins)
    for i in 2:(n_bins - 1)
        edge[i] = smoothed[i + 1] - smoothed[i - 1]
    end

    # TRGB = bin with largest positive edge signal
    m_trgb_idx = argmax(@view edge[2:end-1]) + 1
    m_trgb = bin_centers[m_trgb_idx]

    return GLOESSResult(m_trgb, edge, smoothed, bin_centers, counts)
end
