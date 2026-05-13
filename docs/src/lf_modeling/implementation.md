# Implementation: TRGB Luminosity Function Modeling

See the [getting started](@ref getting_started) page for practical examples.
This page provides API documentation.

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

MLE/MAP estimates can be obtained with [`fit`](@ref TRGBDistances.fit)

```@docs
TRGBDistances.fit
TRGBDistances.logprior
TRGBDistances.TRGBFitResult
```

And posterior samples can be drawn with [`sample`](@ref TRGBDistances.sample)

```@docs
TRGBDistances.sample
TRGBDistances.TRGBChain
```

The fitting and sampling methods support different backends through package extensions.
The available backends are described below.

## Fitting Backends

```@docs
TRGBDistances.AbstractOptimizerBackend
TRGBDistances.OptimJL
```

## Sampling Backends

```@docs
TRGBDistances.AbstractSamplerBackend
TRGBDistances.KissMCMCJL
TRGBDistances.DynamicHMCJL
TRGBDistances.AdvancedMHJL
TRGBDistances.AffineInvariantMCMCJL
```