module TRGBDistancesOptimExt

# Loaded when Optim is available.
# Handles all OptimJL dispatch:
#   - Derivative-free methods (NelderMead, etc.): no AD needed.
#   - First-order methods (BFGS, etc.): calls build_gradient(backend.ad, f).
#   - Second-order methods (Newton, etc.): calls build_gradient + build_hessian.
# build_gradient / build_hessian are defined in the AD
# extensions, which must also be loaded for gradient-based methods.

using TRGBDistances: TRGBDistances, OptimJL, TRGBFitResult, build_gradient, build_hessian
import TRGBDistances: _fit
using Optim: Optim

function _fit(backend::OptimJL, objective, x0; kwargs...)
    method  = isnothing(backend.method)  ? Optim.NelderMead() : backend.method
    options = isnothing(backend.options) ? Optim.Options()    : backend.options
    ad      = backend.ad
    x0f     = float.(x0)

    result = if !isnothing(ad) && method isa Optim.SecondOrderOptimizer
        _, g! = build_gradient(ad, objective)
        _, h! = build_hessian(ad, objective)
        td = Optim.TwiceDifferentiable(objective, g!, h!, x0f)
        Optim.optimize(td, x0f, method, options; kwargs...)
    elseif !isnothing(ad) && method isa Optim.FirstOrderOptimizer
        _, g! = build_gradient(ad, objective)
        od = Optim.OnceDifferentiable(objective, g!, x0f)
        Optim.optimize(od, x0f, method, options; kwargs...)
    else
        Optim.optimize(objective, x0f, method, options; kwargs...)
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

