"""
Benchmarks for TRGBDistances.jl luminosity function evaluation.

Compares the old nested-quadrature approach (ϕ₂ calling ϕ calling quadgk)
with the optimized analytic CDF-based normalization (ϕ_norm).
"""

using BenchmarkTools
using TRGBDistances
using TRGBDistances: exp_photerr, Martin2016_complete
using QuadGK: quadgk
using Random: Xoshiro

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

# model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
# err_func(m)      = 0.05 + 0.005 * (m - 24.0)^2
# complete_func(m) = m < 26.5 ? 1.0 : 0.0
# bias_func(m)     = 0.0

model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
err_func      = x -> exp_photerr(x, 1.05, 10.0, 32.0, 0.01)
complete_func = x -> Martin2016_complete(x, 1.0, 26.0, 0.3)
bias_func     = m -> 0.0

# Synthetic catalog
rng = Xoshiro(42)
mags = TRGBDistances.observe(rng, model, 200;
                              err_func=err_func,
                              complete_func=complete_func,
                              bias_func=bias_func)

limits = extrema(mags)

# -----------------------------------------------------------------------
# Benchmark 1: Single ϕ evaluation
# -----------------------------------------------------------------------
println("=== ϕ (single star convolution integral) ===")
b_phi = @benchmark ϕ(24.0, $model, $err_func, $complete_func, $bias_func)
display(b_phi)

# -----------------------------------------------------------------------
# Benchmark 2: Normalization — old nested quadrature (ϕ₂ equivalent)
# The old approach was:  quadgk(m -> ϕ(m, ...), limits...)
# i.e., outer 1D integral over m, each step evaluating inner 1D integral.
# -----------------------------------------------------------------------
println("\n=== Old normalization: nested quadrature (quadgk over ϕ) ===")
old_norm(lims) = quadgk(m -> ϕ(m, model, err_func, complete_func, bias_func), lims...)[1]
b_old_norm = @benchmark old_norm($limits)
display(b_old_norm)

# -----------------------------------------------------------------------
# Benchmark 3: Normalization — new analytic CDF reduction (ϕ_norm)
# Single 1D integral over true magnitude x, using analytic Gaussian CDF.
# -----------------------------------------------------------------------
println("\n=== New normalization: analytic CDF reduction (ϕ_norm) ===")
b_new_norm = @benchmark ϕ_norm($limits, $model, $err_func, $complete_func, $bias_func)
display(b_new_norm)

# -----------------------------------------------------------------------
# Benchmark 4: Full negative log-likelihood evaluation
# -----------------------------------------------------------------------
println("\n=== Full loglikelihood (N=$(length(mags)) stars) ===")
b_ll = @benchmark loglikelihood($model, $mags, $err_func, $complete_func, $bias_func)
display(b_ll)

# -----------------------------------------------------------------------
# Speedup summary
# -----------------------------------------------------------------------
println("\n=== Speedup (old normalization / new normalization) ===")
println("Median speedup: $(round(median(b_old_norm.times) / median(b_new_norm.times); digits=1))×")
