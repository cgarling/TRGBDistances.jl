```@meta
CurrentModule = TRGBDistances
```

# GLOESS Smoothing

## Algorithm

GLOESS (Gaussian-windowed Locally-Weighted Scatterplot Smoothing) was
introduced by [Persson2004](@cite) to smooth Cepheid light curves and has
since been applied to TRGB detection [Hatt2017](@cite).  For TRGB work the
algorithm proceeds in three steps:

1. **Bin** the magnitudes into a histogram of width ``\Delta m``.
2. **Smooth** the histogram using the Nadaraya–Watson kernel estimator with a
   Gaussian kernel of bandwidth ``h``:

   ```math
   \hat{n}(m_i) =
       \frac{\displaystyle\sum_j n_j \,
             \exp\!\!\left[-\dfrac{(m_i - m_j)^2}{2h^2}\right]}
            {\displaystyle\sum_j
             \exp\!\!\left[-\dfrac{(m_i - m_j)^2}{2h^2}\right]}.
   ```

   The bandwidth ``h`` controls the smoothing scale.  A value of
   ``0.1``–``0.3`` mag is typical for well-sampled RGB sequences.

3. **Apply the Sobel kernel** ``[-1, 0, +1]`` to the smoothed histogram
   [Lee1993](@cite) and identify the TRGB at the bin with the largest edge
   signal (see [Sobel Edge-Detection Filter](@ref) for details).

## Bandwidth selection

The bandwidth ``h`` is the key free parameter:

- Too **small** (``h \lesssim \sigma_{\rm phot}``): Poisson noise in the
  histogram is preserved and may produce a false edge detection.
- Too **large** (``h \gtrsim 0.5`` mag): the genuine TRGB edge is smeared and
  the location estimate biases toward the mid-point of the rise.

A practical starting point is ``h \approx 2\text{–}5 \times \sigma_{\rm phot}``
where ``\sigma_{\rm phot}`` is the typical photometric error near the TRGB.

## When to use

- **Use GLOESS** instead of the plain Sobel filter when the histogram is noisy
  (few stars, large photometric errors), or when you want a smoother edge
  signal to visually inspect.
- **Prefer the plain Sobel filter** when the sample is large and photometric
  errors are small, because GLOESS adds minimal benefit and slightly blurs the
  edge.
- **Prefer LF fitting** ([`fit`](@ref)) when formal uncertainties are required.

## API

```@docs
gloess_trgb
gloess_smooth
GLOESSResult
```

## References

```@bibliography
Pages = ["gloess.md"]
Canonical = false
```
