module TRGBDistancesOptimForwardDiffExt

# This extension is loaded when BOTH Optim and ForwardDiff are present.
# It defines a more-specific _fit override for OptimJL{M,O,Nothing} that
# automatically uses ForwardDiff-based gradients for first-order (gradient-
# based) Optim methods.
#
# ForwardDiff cannot differentiate through quadgk (adaptive Gauss-Kronrod)
# because it tries to compute the quadrature nodes at the Dual number type.
# To work around this, this extension rebuilds the objective with quad=:fixed,
# which uses pre-computed Float64 Gauss-Legendre nodes that are compatible with
# ForwardDiff.

using TRGBDistances: TRGBDistances, OptimJL, TRGBFitResult, _build_objective
import TRGBDistances: _fit
using Optim: Optim
using ForwardDiff: ForwardDiff

# More specific than the base extension's _fit(::OptimJL, ...) because the
# third type parameter is constrained to Nothing.  Julia therefore always
# dispatches here (instead of the base extension) for OptimJL{M,O,Nothing}
# when this extension is loaded.
function _fit(backend::OptimJL{M,O,Nothing}, model_factory, mags, err_func, complete_func,
              bias_func, x0, prior, int_width, quad, objective; kwargs...) where {M,O}
    method  = isnothing(backend.method)  ? Optim.NelderMead() : backend.method
    options = isnothing(backend.options) ? Optim.Options()    : backend.options
    # For first-order (gradient-based) methods, rebuild the objective with
    # quad=:fixed so that ForwardDiff can differentiate through the integrals.
    result = if method isa Optim.FirstOrderOptimizer
        obj_fixed = _build_objective(model_factory, mags, err_func, complete_func,
                                     bias_func, prior, int_width; quad=:fixed)
        g!(G, x) = ForwardDiff.gradient!(G, obj_fixed, x)
        od = Optim.OnceDifferentiable(obj_fixed, g!, float.(x0))
        Optim.optimize(od, float.(x0), method, options; kwargs...)
    else
        Optim.optimize(objective, float.(x0), method, options; kwargs...)
    end
    return TRGBFitResult(
        Optim.minimizer(result),
        Optim.minimum(result),
        Optim.converged(result),
        backend,
        result,
    )
end

end # module

