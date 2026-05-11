```@meta
CurrentModule = TRGBDistances
```

# TRGBDistances

Documentation for [TRGBDistances](https://github.com/cgarling/TRGBDistances.jl).

`TRGBDistances.jl` provides tools for measuring distances using the Tip of the Red Giant
Branch (TRGB) standard candle. The package implements:

- **Luminosity function modeling** following Makarov et al. (2006): broken power-law
  ``\psi(m)`` convolved with photometric errors, completeness, and bias.
- **Maximum likelihood and MAP estimation** via a flexible backend API (Optim.jl).
- **Posterior sampling** via MCMC (KissMCMC.jl backend).
- **Synthetic catalog generation** via analytic inverse-CDF sampling.
- **Photometric filter selection** utilities.
- **TRGB calibrations** ([Rizzi2007](@citet), Fusco+2012).

## Contents

```@contents
Pages = [
    "getting_started.md",
    "lf_modeling/theory.md",
    "dev/makarov_audit.md",
]
Depth = 2
```

## API Reference

```@index
```

```@autodocs
Modules = [TRGBDistances]
```
