# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/cgarling/TRGBDistances.jl")
```

## Quick Start: Fitting a TRGB Magnitude

```julia
using TRGBDistances
using Optim  # loads the OptimJL backend extension

# 1. Define photometric calibration functions
#    (typically derived from artificial star tests)
err_func(m)      = 0.05 + 0.005 * (m - 25.0)^2   # photometric error vs magnitude
complete_func(m) = m < 26.5 ? 1.0 : 0.0           # completeness function
bias_func(m)     = 0.0                              # photometric bias (measured - input)

# 2. Load your dereddened stellar magnitudes (I-band or equivalent)
# mags = [...]  # Vector{Float64} of apparent magnitudes

# 3. Generate synthetic data for this example
model_true = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
mags = observe(model_true, 500; err_func=err_func, complete_func=complete_func, bias_func=bias_func)

# 4. Set initial guess for parameters [m_trgb, a, b, c]
x0 = [24.2, 0.3, 0.2, 0.1]

# 5. Fit by MLE
result = fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0)

println("TRGB magnitude: ", result.minimizer[1])
println("Converged: ", result.converged)
```

## MAP Estimation with a Prior on m_TRGB

```julia
using Distributions

# Gaussian prior on m_trgb based on a distance estimate from another method
prior = (Normal(24.0, 0.3), nothing, nothing, nothing)

result_map = fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
                 prior=prior)
```

## Posterior Sampling

```julia
using KissMCMC  # loads the KissMCMC backend extension

chain = sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
               prior=prior,
               backend=KissMCMCJL(nsamples=10_000, nburnin=1_000))

println("Acceptance fraction: ", chain.acceptance_fraction)
println("Posterior mean m_trgb: ", mean(chain.samples[1, :]))
```

## Generating Synthetic Catalogs

```julia
using Random

model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
rng = Xoshiro(42)

# Generate 1000 stars with observational effects
synthetic_mags = observe(rng, model, 1000;
                         err_func=err_func,
                         complete_func=complete_func,
                         bias_func=bias_func)
```
