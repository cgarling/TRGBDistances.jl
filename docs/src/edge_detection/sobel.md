```@meta
CurrentModule = TRGBDistances
```

# Sobel Edge-Detection Filter

## Algorithm

The Sobel filter locates the TRGB as the sharpest edge in the stellar
luminosity function (LF) histogram [Lee1993](@cite).  The algorithm is:

1. **Bin** the dereddened apparent magnitudes into a histogram with bin width
   ``\Delta m``.
2. *(Optional)* **Pre-smooth** the histogram by convolving it with a
   user-supplied response distribution ``R(m)`` (see [Response function](@ref)).
3. **Apply** the discrete Sobel kernel ``[-1, 0, +1]``:

   ```math
   S(m_i) = \hat{n}(m_{i+1}) - \hat{n}(m_{i-1}),
   ```

   where ``\hat{n}`` is either the raw count (step 1) or the pre-smoothed
   count (step 2), and ``m_i`` are the bin centers.
4. **Identify** the TRGB as the bin with the *largest* ``S(m_i)``.  The RGB
   luminosity function rises steeply toward fainter magnitudes (increasing
   ``m``), so the edge at the TRGB produces the largest positive gradient.

## Response function

By default (`response=nothing`) a pure Sobel kernel is applied to the raw
histogram.  This is optimal when photometric errors are negligible compared to
the bin width. This is equivalent to a Delta function response.

When photometric errors are non-negligible, the sharp edge at the TRGB is
smeared over a scale comparable to the error ``\sigma_{\rm phot}``.  A
*matched filter* can improve the detection: convolve the histogram with the
expected smearing profile before applying the Sobel operator.  Passing a
`Distributions.UnivariateDistribution` as the `response` keyword argument
performs this convolution:

```math
\hat{n}_{\rm smooth}(m_i) = \sum_j n_j \, p\!\left(R;\, m_i - m_j\right),
```

where ``p(R; \cdot)`` is the probability density function of the response
distribution `R`.

A natural choice is `response = Normal(0, σ)` with ``\sigma`` equal to the
typical photometric error near the TRGB.  This
matches [Sakai1996](@citet), who apply a Gaussian kernel before the edge
filter.

## When to use

- **Use the Sobel filter** when you want a fast, non-parametric TRGB estimate.
- **Add Gaussian pre-smoothing** (`response = Normal(0, σ)`) when
  photometric errors are large relative to the bin width, or when the raw
  histogram is noisy due to small sample size.
- **Prefer LF fitting** ([`fit`](@ref)) when formal uncertainties are needed
  or when the data have a well-characterised completeness function.

## API

```@docs
TRGBDistances.sobel_trgb
TRGBDistances.SobelResult
```

## References

```@bibliography
Pages = ["sobel.md"]
Canonical = false
```
