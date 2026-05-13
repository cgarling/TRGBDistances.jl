# Implementation: TRGB Luminosity Function Modeling

As outlined in the [theory](@ref lf_theory) document, the broken power law model
is most commonly used to model the transition from the RGB to the AGB.

```@docs
TRGBDistances.BrokenPowerLaw
```

This is a concrete subtype of [`TRGBDistances.LuminosityFunction`](@ref),
which defines the interface that all luminosity functions must conform
to in order to work with our API.

```@docs
TRGBDistances.LuminosityFunction
TRGBDistances.ψ
TRGBDistances.is_valid
TRGBDistances.ψ_breakpoints
TRGBDistances.loglikelihood
```

```@docs
TRGBDistances.fit
TRGBDistances.TRGBResult
TRGBDistances.sample
TRGBDistances.TRGBChain
```

## Backends