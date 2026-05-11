"""
    AbstractOptimizerBackend

Abstract supertype for all optimizer backends used with [`fit`](@ref).
Concrete subtypes should implement `_fit(backend, model_factory, mags, err_func, complete_func, bias_func, x0, prior; kwargs...)`.
"""
abstract type AbstractOptimizerBackend end

"""
    TRGBFitResult

Result returned by [`fit`](@ref).

Fields:
- `minimizer`: Parameter vector at the optimum (MLE or MAP).
- `minimum`: Objective value (negative log-likelihood or negative log-posterior) at the optimum.
- `converged`: Whether the optimizer reported convergence.
- `backend`: The backend used.
- `result`: The raw result from the backend (backend-specific).
"""
struct TRGBFitResult{T,R,B}
    minimizer::T
    minimum::Float64
    converged::Bool
    backend::B
    result::R
end

Base.show(io::IO, r::TRGBFitResult) =
    print(io, "TRGBFitResult(minimizer=$(r.minimizer), minimum=$(r.minimum), converged=$(r.converged))")

# -----------------------------------------------------------------------
# Prior interface
# -----------------------------------------------------------------------

"""
    logprior(prior, θ)

Evaluate the log-prior of parameter vector `θ` under `prior`.
By default (when `prior=nothing`) returns `0.0` (flat/improper prior).
"""
logprior(::Nothing, θ) = 0.0

"""
    logprior(prior::Function, θ)

Evaluate a user-supplied log-prior function.
"""
logprior(prior::Function, θ) = prior(θ)

"""
    logprior(prior::Tuple, θ)

Evaluate a tuple of per-parameter priors (from Distributions.jl or any object with `logpdf`).
Pass `nothing` to skip a parameter's prior.

# Example
For [`BrokenPowerLaw`](@ref), a tuple like `(Normal(24.0, 0.5), nothing, nothing, nothing)`
applies a Gaussian prior on `m_trgb` and flat priors on the other parameters.

```julia
using Distributions: Normal
prior = (Normal(24.0, 0.5), nothing, nothing, nothing)  # prior only on m_trgb
```
"""
function logprior(prior::Tuple, θ)
    lp = 0.0
    for (p, x) in zip(prior, θ)
        if p !== nothing
            lp += logpdf(p, x)
        end
    end
    return lp
end

# -----------------------------------------------------------------------
# Objective builder (negative log-posterior = NLL + negative log-prior)
# -----------------------------------------------------------------------

function _build_objective(model_factory, mags, err_func, complete_func, bias_func, prior, int_width; quad=:adaptive)
    function objective(θ)
        model = model_factory(θ...)
        ll = loglikelihood(model, mags, err_func, complete_func, bias_func; int_width, quad)
        lp = logprior(prior, θ)
        return -(ll + lp)  # minimize negative log-posterior
    end
    return objective
end

# -----------------------------------------------------------------------
# Default backend (pure Julia NelderMead-like grid search placeholder)
# -----------------------------------------------------------------------

"""
    OptimJL(; method=nothing, options=nothing, autodiff=nothing)

Backend for optimization using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).
Requires the `Optim` package to be loaded (via the `TRGBDistancesOptimExt` extension).

`method` defaults to `Optim.NelderMead()`.

`autodiff` can be set to an `ADTypes.jl` autodiff type (e.g. `ADTypes.AutoForwardDiff()`)
to enable automatic differentiation for gradient-based methods. When both `Optim` and
`ForwardDiff` are loaded, the `TRGBDistancesOptimForwardDiffExt` extension automatically
uses `AutoForwardDiff` for first-order methods when `autodiff` is `nothing`.

# Example
```julia
using Optim
result = fit(BrokenPowerLaw, mags, err, compl, bias, x0; backend=OptimJL())
```
"""
struct OptimJL{M,O,A} <: AbstractOptimizerBackend
    method::M
    options::O
    autodiff::A
end
OptimJL(; method=nothing, options=nothing, autodiff=nothing) = OptimJL(method, options, autodiff)

# -----------------------------------------------------------------------
# fit() — public API
# -----------------------------------------------------------------------

"""
    fit(model_factory, mags, err_func, complete_func, bias_func, x0;
        backend=OptimJL(), prior=nothing, int_width=1.0, kwargs...)

Fit a [`LuminosityFunction`](@ref) model to observed magnitudes using maximum likelihood
estimation (MLE) or maximum a posteriori (MAP) estimation.

# Arguments

- `model_factory`: A callable that accepts a parameter vector and returns a [`LuminosityFunction`](@ref).
  Typically one of the concrete types like [`BrokenPowerLaw`](@ref).
- `mags`: Vector of observed (dereddened) apparent magnitudes.
- `err_func`: `err_func(m)` returns the photometric error at magnitude `m`.
- `complete_func`: `complete_func(m)` returns the photometric completeness at `m` (in [0,1]).
- `bias_func`: `bias_func(m)` returns the photometric bias at `m` (`<measured> - <input>`).
- `x0`: Initial parameter vector.

# Keyword Arguments

- `backend`: An [`AbstractOptimizerBackend`](@ref) instance (default: `OptimJL()`).
- `prior`: Optional prior. Can be `nothing` (flat), a `Function`, or a `Tuple` of per-parameter
  distributions. See [`logprior`](@ref).
- `int_width`: Half-width of the integration window for the convolution integral (default `1.0` mag).

# Returns

A [`TRGBFitResult`](@ref).

# Examples

```jldoctest
julia> using TRGBDistances: exp_photerr, Martin2016_complete, fit, observe, BrokenPowerLaw, TRGBFitResult, OptimJL

julia> using Optim

julia> using StableRNGs: StableRNG

julia> err_func(x) = exp_photerr(x, 1.05, 10.0, 32.0, 0.01);

julia> complete_func(x) = Martin2016_complete(x, 1.0, 26.0, 0.3);

julia> bias_func(m) = 0.0;

julia> model_true = BrokenPowerLaw(22.0, 0.3, 0.4, 0.1);

julia> x0 = [22.2, 0.4, 0.3, 0.1];  # start near truth

julia> mags = observe(StableRNG(42), model_true, 500; err_func, complete_func, bias_func, upper_limit=1.0); # 500 stars within 1 mag of the TRGB

julia> result = fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0); # Uses OptimJL(method=Optim.NelderMead()) by default

julia> result isa TRGBFitResult
true

julia> result = fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0; backend=OptimJL(method=Optim.BFGS())); # Use BFGS with autodifferentiation

julia> result isa TRGBFitResult
true
```
"""
function fit(model_factory, mags, err_func, complete_func, bias_func, x0;
             backend=OptimJL(), prior=nothing, int_width=1.0, quad=:adaptive, kwargs...)
    objective = _build_objective(model_factory, mags, err_func, complete_func, bias_func, prior, int_width; quad)
    return _fit(backend, model_factory, mags, err_func, complete_func, bias_func, x0, prior, int_width, quad, objective; kwargs...)
end

# Generic fallback for unloaded backends
function _fit(backend::AbstractOptimizerBackend, model_factory, mags, err_func, complete_func, bias_func, x0, prior, int_width, quad, objective; kwargs...)
    error("Backend $(typeof(backend)) requires loading the corresponding package. " *
          "For OptimJL, use `using Optim`.")
end
