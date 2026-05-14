```@meta
CurrentModule = TRGBDistances
```

# Edge Detection Methods

In addition to parametric luminosity-function (LF) modeling, `TRGBDistances.jl`
provides two classical *edge-detection* algorithms for locating the TRGB in a
stellar magnitude histogram:

| Method | Type | Reference |
|--------|----------|-----------|
| Sobel filter | [`Sobel`](@ref) | [Lee1993](@citet) |
| GLOESS + Sobel | [`GLOESS`](@ref) | [Persson2004](@citet); [Hatt2017](@citet) |

We provide the [`trgb`](@ref) method to find TRGB magnitudes after constructing an instance
of one of the above with your desired algorithm options .

```@docs
trgb
```

Uncertainty estimates can be derived for edge detection methods via bootstrap resampling.

```@docs
bootstrap
```

## Why use edge detection instead of LF fitting?

Edge-detection methods have several attractive properties:

- **Non-parametric.** No assumption is made about the shape of the luminosity
  function; the methods find any sharp feature in the histogram.
- **Fast.** Both methods run in microseconds on typical datasets.
- **Minimal tuning.** Only a bin width (and optionally a smoothing bandwidth)
  must be specified.

They are therefore well suited for:

- A quick, order-of-magnitude TRGB estimate before running a full LF fit.
- Datasets with non-standard luminosity-function shapes (e.g., contaminated
  by non-RGB stellar populations) where the broken-power-law model is
  unreliable.
- Exploratory analysis or algorithm comparison.

The principal limitation is that these methods do not produce a formal
uncertainty estimate. Bootstrap resampling can be used to estimate
uncertainties when using edge-detection methods (see [`bootstrap`](@ref)).
Generally we advise using [`fit`](@ref) (LF MLE/MAP) or [`sample`](@ref)
(posterior sampling) when rigorous uncertainty quantification is needed.

## Contents

```@contents
Pages = ["sobel.md", "gloess.md"]
Depth = 2
```
