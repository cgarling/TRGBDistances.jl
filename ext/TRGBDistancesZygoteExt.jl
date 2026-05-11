module TRGBDistancesZygoteExt

# Loaded when Zygote is available.
# Provides build_gradient and build_hessian for Zygote

import ADTypes
using TRGBDistances: TRGBDistances
import TRGBDistances: build_gradient, build_hessian
using Zygote: Zygote

"""
    build_gradient(::ADTypes.AutoZygote, f) -> (g, g!)

Return gradient callables for `f` using Zygote (reverse-mode AD).
- `g(x)` allocates and returns the gradient vector.
- `g!(G, x)` writes the gradient in-place to `G`.
"""
function build_gradient(::ADTypes.AutoZygote, f)
    g(x) = only(Zygote.gradient(f, x))
    g!(G, x) = copyto!(G, only(Zygote.gradient(f, x)))
    return g, g!
end

"""
    build_hessian(::ADTypes.AutoZygote, f) -> (h, h!)

Return Hessian callables for `f` using Zygote.
- `h(x)` allocates and returns the Hessian matrix.
- `h!(H, x)` writes the Hessian in-place to `H`.
"""
function build_hessian(::ADTypes.AutoZygote, f)
    h(x) = Zygote.hessian(f, x)
    h!(H, x) = copyto!(H, Zygote.hessian(f, x))
    return h, h!
end

end # module
