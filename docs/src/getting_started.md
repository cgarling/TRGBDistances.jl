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
[KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl) provides a multi-threaded implementation
of the Goodman–Weare affine-invariant ensemble sampler (the same algorithm as `emcee` in
Python). It is simple, lightweight, and performant -- for low dimensional TRGB fits
(there are only four free parameters in the broken power law luminosity function model)
this algorithm is able to achieve good sampling efficiency.

```@example
using KissMCMC  # loads the KissMCMC backend extension

chain = TRGBDistances.sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, use_progress_meter=false,
    backend=KissMCMCJL(nsamples=2_000, nburnin=200))

println("Acceptance fraction: ", chain.acceptance_fraction)
println("Posterior mean and std (m_trgb): ", mean(chain.samples[1, :]), " ± ", std(chain.samples[1, :]))
```

## Posterior Sampling with AdvancedMH.jl

[AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) provides a random-walk
Metropolis-Hastings sampler.  The `proposal_scale` keyword controls the standard deviation
of the isotropic Gaussian proposal; it should be set to roughly the marginal posterior
standard deviation of the parameters (a few percent of a magnitude is a good first guess
for `m_trgb`).

```@example
using AdvancedMH

chain_amh = TRGBDistances.sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, rng=StableRNG(2),
    backend=AdvancedMHJL(nsamples=1_000, nburnin=100, proposal_scale=0.01))

println("Acceptance fraction: ", round(chain_amh.acceptance_fraction; digits=3))
println("Posterior mean and std (m_trgb): ",
    round(mean(chain_amh.samples[1, :]); digits=4), " ± ",
    round(std(chain_amh.samples[1, :]); digits=4))
```

## Posterior Sampling with AffineInvariantMCMC.jl

[AffineInvariantMCMC.jl](https://github.com/madsjulia/AffineInvariantMCMC.jl) implements
the Goodman–Weare affine-invariant ensemble sampler (the same algorithm as `emcee` in
Python).  Like KissMCMC's `emcee` backend, it maintains a population of walkers that
collectively explore the posterior; it performs well on correlated, anisotropic posteriors
without requiring any tuning of a proposal covariance. This package does not support threading
while KissMCMC does, making it slower when multiple threads are available.

```@example
using AffineInvariantMCMC

chain_ai = TRGBDistances.sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, rng=StableRNG(3),
    backend=AffineInvariantMCMCJL(nsamples=1_000, nburnin=100, nwalkers=4))

println("Acceptance fraction: ", round(chain_ai.acceptance_fraction; digits=3))
println("Posterior mean and std (m_trgb): ",
    round(mean(chain_ai.samples[1, :]); digits=4), " ± ",
    round(std(chain_ai.samples[1, :]); digits=4))
```

## Posterior Sampling with DynamicHMC

[DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) uses a Hamiltonian Monte Carlo
algorithm and is slower than basic MCMC methods (such as are implemented by KissMCMC.jl)
for a fixed number of samples.  As currently implemented, each likelihood evaluation
involves a numerical convolution integral, and HMC requires many gradient evaluations
per sample; for [`BrokenPowerLaw`](@ref) with only four free parameters, basic MCMC
algorithms such as KissMCMC or AdvancedMH are likely to be more efficient.  HMC may
become comparatively more efficient for higher-dimensional problems.

```julia
using DynamicHMC

chain = TRGBDistances.sample(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    prior=prior, rng=rng,
    backend=DynamicHMCJL(nsamples=200, n_warmup=100, ad=AutoForwardDiff()))
```

## Edge Detection with the Sobel Filter

Edge-detection methods provide a fast, non-parametric TRGB estimate that does not require
specifying a luminosity-function model. However, these methods can be "tricked" as we will
see below. The [Sobel filter](@ref "Sobel Edge-Detection Filter")
[Lee1993](@cite) detects the sharpest step in the magnitude histogram.

```@example
# Pure Sobel (no pre-smoothing) — fast, appropriate when errors ≪ bin_width
result_sobel = sobel_trgb(mags; bin_width=0.1)
println("Sobel TRGB: ", round(result_sobel.m_trgb; digits=3))
```

You will note that this is not the correct answer; the Sobel filter has found the
the sharpest step in the histogram, but in this random sample, that step is *not* the TRGB.

```@example
fig2 = Figure()
ax2  = Axis(fig2[1, 1], xlabel = "Magnitude", ylabel = "Count",
    title = "Edge Detection Comparison",
    limits = (extrema(hist_bins)..., nothing, nothing))

hist!(ax2, mags; color = :blue, label = "Observed", bins = hist_bins)
vlines!(ax2, [model_true.m_trgb];      color = :red,    linestyle = :solid,  linewidth = 2, label = "True TRGB")
vlines!(ax2, [result.minimizer[1]];    color = :black,  linestyle = :dash, linewidth = 2, label = "LF fit")
vlines!(ax2, [result_sobel.m_trgb];    color = :green,  linestyle = :dashdot,   linewidth = 2, label = "Sobel")

axislegend(ax2, position = :lt)
fig2
```

To prevent such misidentifications, `sobel_trgb` takes a `magnitude_range` keyword argument
that can be used to restrict the range of magnitudes it searches, ensuring it finds the right
feature:

```@example
# Pure Sobel (no pre-smoothing) — fast, appropriate when errors ≪ bin_width
result_sobel = sobel_trgb(mags; bin_width=0.1, magnitude_range=(23.5,24.5))
println("Sobel TRGB: ", round(result_sobel.m_trgb; digits=3))
```

Passing a `response` distribution convolves the histogram with the distribution PDF before
applying the Sobel kernel, acting as a matched filter for photometric-error smearing
[Sakai1996](@cite):

```@example
using Distributions

# Pre-smooth with a Gaussian matching the typical photometric error near the TRGB
result_sobel_smooth = sobel_trgb(mags; bin_width=0.1, magnitude_range=(23.5,24.5),
                                 response=Normal(0, 0.05))
println("Sobel (Gaussian pre-smooth) TRGB: ", round(result_sobel_smooth.m_trgb; digits=3))
```

## Edge Detection with GLOESS

The [GLOESS](@ref "GLOESS Smoothing") algorithm [Persson2004](@cite) applies
Gaussian-windowed locally-weighted smoothing to the histogram before Sobel edge detection
[Madore2009](@cite).  It is more robust than the plain Sobel filter when the histogram
is noisy.

```@example
result_gloess = gloess_trgb(mags; bandwidth=0.2, bin_width=0.1, magnitude_range=(23.5,24.5))
println("GLOESS TRGB: ", round(result_gloess.m_trgb; digits=3))
```

We can overlay all three estimates on the histogram:

```@example
fig2 = Figure()
ax2  = Axis(fig2[1, 1], xlabel = "Magnitude", ylabel = "Count",
    title = "Edge Detection Comparison",
    limits = (extrema(hist_bins)..., nothing, nothing))

hist!(ax2, mags; color = :blue, label = "Observed", bins = hist_bins)
vlines!(ax2, [model_true.m_trgb]; color = :red, linestyle = :solid, linewidth = 2, label = "True TRGB")
vlines!(ax2, [result.minimizer[1]]; color = :black, linestyle = :dash, linewidth = 2, label = "LF fit")
vlines!(ax2, [result_sobel.m_trgb]; color = :green, linestyle = :dashdot, linewidth = 2, label = "Sobel")
vlines!(ax2, [result_gloess.m_trgb]; color = :purple, linestyle = :dot, linewidth = 2, label = "GLOESS")

axislegend(ax2, position = :lt)
fig2
```

## References
This page cites the following references:

```@bibliography
Pages = ["getting_started.md"]
Canonical = false
```