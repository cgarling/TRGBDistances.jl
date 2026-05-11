```@meta
CurrentModule = TRGBDistances
ShareDefaultModule = true
```

# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/cgarling/TRGBDistances.jl")
```

## Quick Start: Fitting a TRGB Magnitude

```@example
using TRGBDistances
using Optim  # loads the OptimJL backend extension
using StableRNGs: StableRNG # For repeatability

rng = StableRNG(1)

# 1. Define photometric calibration functions
# here we use empirical models, but typically
# these would be derived from artificial star tests
err_func(m) = TRGBDistances.exp_photerr(m, 1.05, 10.0, 32.0, 0.01)
complete_func(m) = TRGBDistances.Martin2016_complete(m, 1.0, 26.0, 0.3)
bias_func(m) = 0.0 # photometric bias (measured - input), leave 0

# 2. Load your dereddened stellar magnitudes (I-band or equivalent)
# mags = [...]  # Vector{Float64} of apparent magnitudes

# 3. Generate synthetic data for this example;
# 1000 stars within 2 mags of the TRGB
m_trgb = 24.0
model_true = BrokenPowerLaw(m_trgb, 0.3, 0.2, 0.1)
mags = observe(rng, model_true, 1000;
               err_func, complete_func, bias_func, upper_limit=2.0)

# 4. Set initial guess for parameters [m_trgb, a, b, c]
x0 = [m_trgb + 0.2, 0.3, 0.2, 0.1]

# 5. Fit by MLE
result = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0)

println("TRGB magnitude: ", result.minimizer[1])
println("Converged: ", result.converged)
```

Now we'll plot the histogram of the data with the true TRGB and best-fit solution:

```@example
using CairoMakie

hist_bins = range(m_trgb - 4, m_trgb + 2; length=30)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Magnitude", ylabel = "Count", 
    title = "TRGB Fit", limits = (extrema(hist_bins)..., nothing, nothing))

# Histogram of magnitudes
hist!(ax, mags; color = :blue, label = "Observed Magnitudes", bins = hist_bins)

# True TRGB
vlines!(ax, [model_true.m_trgb]; color = :red, linestyle = :solid,
    linewidth = 2, label = "True TRGB")

# Fitted TRGB
vlines!(ax, [result.minimizer[1]]; color = :black, linestyle = :dash,
    linewidth = 2, label = "Fitted TRGB")

axislegend(ax, position = :lt)

fig
```

## MAP Estimation with a Prior on m_TRGB

```@example
using Distributions

# Gaussian prior on m_trgb based on a distance estimate from another method
prior = (Normal(24.0, 0.3), nothing, nothing, nothing)

result_map = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
                               prior=prior)
```

## Fitting with Different Optim Methods

```@example
using Optim
using ADTypes
using ForwardDiff

# Optimizing with BFGS; quasi-Newton, first order
result_bfgs = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, backend=OptimJL(method=Optim.BFGS(), ad=ADTypes.AutoForwardDiff()))

# Optimizing with Newton; second order
result_newton = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, backend=OptimJL(method=Optim.Newton(), ad=ADTypes.AutoForwardDiff()))

# Optimizing with NewtonTrustRegion; second order
result_ntr = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, backend=OptimJL(method=Optim.NewtonTrustRegion(), ad=ADTypes.AutoForwardDiff()))

println("BFGS TRGB magnitude: ", result_bfgs.minimizer[1])
println("Newton TRGB magnitude: ", result_newton.minimizer[1])
println("NewtonTrustRegion TRGB magnitude: ", result_ntr.minimizer[1])
```

### Choosing an Optimization Method
On well-constrained problems, BFGS, Newton, and NewtonTrustRegion should reach the same answer.
Gradients and Hessians are obtained via automatic differentiation; the above examples use
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) by specifying the
`ad=ADTypes.AutoForwardDiff()` keyword argument.
The second-order methods can often be faster than the first-order methods; we typically recommend
`Optim.NewtonTrustRegion`.

## Posterior Sampling

```@example
using KissMCMC  # loads the KissMCMC backend extension

chain = TRGBDistances.sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, use_progress_meter=false,
    backend=KissMCMCJL(nsamples=1_000, nburnin=100))

println("Acceptance fraction: ", chain.acceptance_fraction)
println("Posterior mean and std (m_trgb): ", mean(chain.samples[1, :]), " ± ", std(chain.samples[1, :]))
```

## Posterior Sampling with DynamicHMC

[DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) uses a Hamiltonian Monte Carlo algorithm and is slower
than basic MCMC methods (such as are implemented by KissMCMC.jl) for a fixed number of samples.
For [`BrokenPowerLaw`](@ref) with four free parameters, using this method is often overkill --
basic MCMC algorithms can achieve high acceptance ratios for this low dimensional space. In general
HMC may become more efficient for higher dimensional problems.
For faster sampling, consider using KissMCMC.

```julia
using DynamicHMC

chain = TRGBDistances.sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, rng=rng,
    backend=DynamicHMCJL(nsamples=200, n_warmup=100, ad=AutoForwardDiff()))
```
