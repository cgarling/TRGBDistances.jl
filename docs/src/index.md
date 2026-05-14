```@meta
CurrentModule = TRGBDistances
```

# TRGBDistances

Documentation for [TRGBDistances](https://github.com/cgarling/TRGBDistances.jl).

`TRGBDistances.jl` provides tools for measuring distances using the Tip of the Red Giant
Branch (TRGB) standard candle. For measuring TRGB magnitudes, the package implements:

- **Luminosity function modeling** following Makarov et al. (2006): broken power-law
  ``\psi(m)`` convolved with photometric errors, completeness, and bias.
- **Maximum likelihood and MAP estimation** via a flexible backend API (Optim.jl).
- **Posterior sampling** via multiple MCMC backends: KissMCMC.jl (affine-invariant
  ensemble), AdvancedMH.jl (random-walk Metropolis-Hastings), AffineInvariantMCMC.jl
  (affine-invariant ensemble), and DynamicHMC.jl (Hamiltonian Monte Carlo).
- **Edge-detection methods**: Sobel filter ([Lee1993](@citet)) and GLOESS+Sobel
  ([Persson2004](@citet); [Hatt2017](@citet)) for fast, non-parametric TRGB estimates.
  Uncertainty estimation via bootstrap resampling is supported.
- **Synthetic catalog generation** via analytic inverse-CDF sampling.
- **Photometric filter selection** utilities.
- **Calibrations** of the TRGB absolute magnitude from the literature [here](@ref calibrations).

## Utilities

```@docs
TRGBDistances.Martin2016_complete
TRGBDistances.exp_photerr
TRGBDistances.observe
TRGBDistances.filter_mags
```

## Contents

```@contents
Pages = [
    "getting_started.md",
    "lf_modeling/theory.md",
    "lf_modeling/implementation.md",
    "edge_detection/index.md",
    "edge_detection/sobel.md",
    "edge_detection/gloess.md",
    "calibrations.md",
]
Depth = 2
```
