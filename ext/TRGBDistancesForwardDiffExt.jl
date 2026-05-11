module TRGBDistancesForwardDiffExt

# Loaded when ForwardDiff is available.
# Provides build_gradient and build_hessian for ForwardDiffAD().
# These are called by optimizer and sampler extensions that need derivative
# information without coupling directly to ForwardDiff.

import ADTypes
import ForwardDiff
using TRGBDistances: TRGBDistances, ForwardDiffAD
import TRGBDistances: build_gradient, build_hessian

"""
    build_gradient(::ADTypes.AutoForwardDiff, f) -> (g, g!)

Return gradient callables for `f` using ForwardDiff (forward-mode AD).
- `g(x)` allocates and returns the gradient vector.
- `g!(G, x)` writes the gradient in-place to `G`.
"""
function build_gradient(::ADTypes.AutoForwardDiff, f)
    g(x) = ForwardDiff.gradient(f, x)
    g!(G, x) = ForwardDiff.gradient!(G, f, x)
    return g, g!
end

"""
    build_hessian(::ADTypes.AutoForwardDiff, f) -> (h, h!)

Return Hessian callables for `f` using ForwardDiff (forward-mode AD).
- `h(x)` allocates and returns the Hessian matrix.
- `h!(H, x)` writes the Hessian in-place to `H`.
"""
function build_hessian(::ADTypes.AutoForwardDiff, f)
    h(x) = ForwardDiff.hessian(f, x)
    h!(H, x) = copyto!(H, ForwardDiff.hessian(f, x))
    return h, h!
end

end # module
