# TRGBDistances

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cgarling.github.io/TRGBDistances.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cgarling.github.io/TRGBDistances.jl/dev/)
[![Build Status](https://github.com/cgarling/TRGBDistances.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cgarling/TRGBDistances.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cgarling/TRGBDistances.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cgarling/TRGBDistances.jl)

`TRGBDistances.jl` provides tools for measuring extragalactic distances using
the **Tip of the Red Giant Branch** (TRGB) standard candle.

## Features

- **Parametric luminosity-function fitting** following
  [Makarov et al. (2006)](https://doi.org/10.1086/508925): a broken power-law
  luminosity function convolved with photometric errors, completeness, and
  photometric bias, fitted by maximum likelihood (MLE) or maximum a posteriori
  (MAP) estimation.
- **Flexible optimization backends** (via package extensions):
  - [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) — NelderMead
    (default), BFGS, Newton, NewtonTrustRegion, with optional ForwardDiff
    automatic differentiation.
- **Posterior sampling backends** (via package extensions):
  - [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl) — affine-invariant
    ensemble (emcee-style).
  - [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) — random-walk
    Metropolis-Hastings.
  - [AffineInvariantMCMC.jl](https://github.com/madsjulia/AffineInvariantMCMC.jl)
    — affine-invariant ensemble (Goodman & Weare 2010).
  - [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) — Hamiltonian
    Monte Carlo (note: slower than the other backends for this use case; see
    documentation).
- **Edge-detection methods** — fast, non-parametric TRGB estimates:
  - **Sobel filter** ([Lee et al. 1993](https://doi.org/10.1086/173327)) —
    applies a discrete edge-detection kernel to the magnitude histogram, with
    optional Gaussian pre-smoothing via a user-supplied response distribution.
  - **GLOESS + Sobel** ([Persson et al. 2004](https://doi.org/10.1086/424934);
    [Hatt et al. 2017](https://doi.org/10.3847/1538-4357/aa7f73)) —
    applies Gaussian locally-weighted smoothing before edge detection,
    improving robustness for noisy histograms.
- **Synthetic catalog generation** using an analytic inverse-CDF sampler.
- **Photometric calibrations** for converting measured TRGB magnitudes
  into distances(Rizzi et al. 2007, Fusco et al. 2012).
- **Color-cut utilities** for selecting RGB stars from a CMD.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/cgarling/TRGBDistances.jl")
```

## Quick start

```julia
using TRGBDistances
using Optim # loads OptimJL extension for luminosity function fitting
using StableRNGs: StableRNG

# 1. Photometric calibration functions (in reality, taken from artificial star tests)
err_func(m) = TRGBDistances.exp_photerr(m, 1.05, 10.0, 32.0, 0.01)
complete_func(m) = TRGBDistances.Martin2016_complete(m, 1.0, 26.0, 0.3)
bias_func(m) = 0.0

# 2. Generate (or load) dereddened magnitudes
model_true = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
mags = observe(StableRNG(1), model_true, 500;
               err_func, complete_func, bias_func, upper_limit=2.0)

# 3. Quick non-parametric estimate with GLOESS+Sobel
#    constraining the solution to be between 23.5 and 24.5 mag
result_gloess = gloess_trgb(mags; bandwidth=0.2, bin_width=0.1, magnitude_range=(23.5, 24.5))
println("GLOESS TRGB: ", result_gloess.m_trgb)

# 4. MLE fit with the broken power-law luminosity function
x0 = [24.2, 0.3, 0.2, 0.1]
result = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0)
println("LF fit TRGB: ", result.minimizer[1], " (converged: ", result.converged, ")")

# 5. Posterior sampling with KissMCMC (affine-invariant ensemble)
using KissMCMC, Distributions
prior = (Normal(24.0, 0.3), nothing, nothing, nothing)
chain = TRGBDistances.sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
                             prior, backend=KissMCMCJL(nsamples=5_000, nburnin=500),
                             use_progress_meter=true)
using Statistics
println("Posterior  m_trgb = ", mean(chain.samples[1, :]),
        " ± ", std(chain.samples[1, :]))
```

Output:

```plain
GLOESS TRGB: 23.95
LF fit TRGB: 24.023271888276373 (converged: true)
Posterior  m_trgb = 24.021977441677706 ± 0.07973460699768788
```

## Documentation

Full documentation is available at <https://cgarling.github.io/TRGBDistances.jl/dev/>.
See the [Getting Started](https://cgarling.github.io/TRGBDistances.jl/dev/getting_started/)
page for more detailed examples including MAP estimation, all sampler backends,
and edge-detection comparisons.
