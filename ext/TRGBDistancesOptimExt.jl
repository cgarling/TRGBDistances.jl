module TRGBDistancesOptimExt

using TRGBDistances: TRGBDistances, OptimJL, TRGBFitResult, AbstractOptimizerBackend
import TRGBDistances: _fit
using Optim: Optim

function _fit(backend::OptimJL, objective, x0; kwargs...)
    method = isnothing(backend.method) ? Optim.NelderMead() : backend.method
    options = isnothing(backend.options) ? Optim.Options() : backend.options
    result = Optim.optimize(objective, float.(x0), method, options; kwargs...)
    return TRGBFitResult(
        Optim.minimizer(result),
        Optim.minimum(result),
        Optim.converged(result),
        backend,
        result,
    )
end

end # module
