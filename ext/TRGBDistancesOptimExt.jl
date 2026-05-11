module TRGBDistancesOptimExt

using TRGBDistances: TRGBDistances, OptimJL, TRGBFitResult, AbstractOptimizerBackend
import TRGBDistances: _fit
using Optim: Optim

# Handles all OptimJL calls: passes `autodiff` to Optim.optimize when it is set.
# When both Optim and ForwardDiff are loaded, the TRGBDistancesOptimForwardDiffExt
# defines a more-specific method for OptimJL{M,O,Nothing} that auto-injects
# ForwardDiff for first-order methods; that more-specific method takes priority.
function _fit(backend::OptimJL, model_factory, mags, err_func, complete_func, bias_func,
              x0, prior, int_width, quad, objective; kwargs...)
    method  = isnothing(backend.method)  ? Optim.NelderMead() : backend.method
    options = isnothing(backend.options) ? Optim.Options()    : backend.options
    ad      = backend.autodiff
    result  = if isnothing(ad)
        Optim.optimize(objective, float.(x0), method, options; kwargs...)
    else
        Optim.optimize(objective, float.(x0), method, options; kwargs..., autodiff=ad)
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
