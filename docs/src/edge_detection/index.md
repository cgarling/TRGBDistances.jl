```@meta
CurrentModule = TRGBDistances
```

# Edge Detection Methods

In addition to parametric luminosity-function (LF) modeling, `TRGBDistances.jl`
provides two classical *edge-detection* algorithms for locating the TRGB in a
stellar magnitude histogram:

| Method | Function | Reference |
|--------|----------|-----------|
| Sobel filter | [`sobel_trgb`](@ref) | [Lee1993](@citet) |
| GLOESS + Sobel | [`gloess_trgb`](@ref) | [Persson2004](@citet); [Madore2009](@citet) |

## Why use edge detection instead of LF fitting?

Edge-detection methods have several attractive properties:

- **Non-parametric.** No assumption is made about the shape of the luminosity
  function; the methods find any sharp feature in the histogram.
- **Fast.** Both methods run in milliseconds on typical datasets.
- **Minimal tuning.** Only a bin width (and optionally a smoothing bandwidth)
  must be specified.

They are therefore well suited for:

- A quick, order-of-magnitude TRGB estimate before running a full LF fit.
- Datasets with non-standard luminosity-function shapes (e.g., contaminated
  by non-RGB stellar populations) where the broken-power-law model is
  unreliable.
- Exploratory analysis or algorithm comparison.

The principal limitation is that these methods do **not** produce a formal
uncertainty estimate. Typically bootstrap resampling is used to derive
uncertainty estimates when using edge-detection methods.
Generally we advise using [`fit`](@ref) (LF MLE/MAP) or [`sample`](@ref)
(posterior sampling) when rigorous uncertainty quantification is needed.

## Contents

```@contents
Pages = ["sobel.md", "gloess.md"]
Depth = 2
```
