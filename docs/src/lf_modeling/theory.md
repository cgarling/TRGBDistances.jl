# Theory: TRGB Luminosity Function Modeling

## Background

The Tip of the Red Giant Branch (TRGB) is a standard candle used to measure extragalactic
distances. At the TRGB, low-mass stars undergo the helium flash, creating a sharp feature in
the stellar luminosity function that is nearly universal in old, metal-poor populations.

In the I band (or HST F814W, at ~814 nm), the TRGB has an absolute magnitude of approximately M_I ≈ −4.0
([Rizzi2007](@citet), with a color correction), making it detectable to distances of several Mpc with
HST or JWST.

## The Broken Power-Law Luminosity Function

Following [Makarov2006(@citet)], the theoretical luminosity function is parameterized as:

```math
\psi(m) = \begin{cases}
10^{a(m - m_\mathrm{TRGB}) + b} & m \geq m_\mathrm{TRGB} \quad \text{(RGB)} \\
10^{c(m - m_\mathrm{TRGB})} & m < m_\mathrm{TRGB} \quad \text{(AGB)}
\end{cases}
```

where:
- ``m_\mathrm{TRGB}`` — apparent TRGB magnitude (the free parameter of primary interest)
- ``a`` — slope of the RGB luminosity function (faint side; typically ``a > 0`` since more stars are fainter)
- ``b`` — log₁₀ of the amplitude discontinuity at the TRGB (the "TRGB jump")
- ``c`` — slope of the AGB luminosity function (bright side)

The discontinuity at ``m = m_\mathrm{TRGB}`` (a factor of ``10^b`` jump in the luminosity function density) is the observable feature used to locate the TRGB.

## Observed Luminosity Function

Real photometric data are affected by:

1. **Photometric errors**: The measured magnitude ``m`` differs from the true magnitude ``m'`` by some function ``e(m|m')``. 
2. **Photometric bias**: Systematic offsets ``\bar{m}(m') = m' + \mathrm{bias}(m')`` (where bias ``= \langle m_\mathrm{meas}\rangle - m_\mathrm{input}``), typically measured via artificial star tests.
3. **Completeness**: Not all stars are detected. The detection fraction ``f(m')`` decreases near the detection limit.

The observed luminosity function ``\phi(m)`` is the convolution of ``\psi`` with the effects of the measurement process
(Equation 4 of [Makarov2006](@citet)),

```math
\phi(m) = \int_{-\infty}^{\infty} \psi(m') \cdot e(m|m') \cdot f(m') \, dm'
```

Assuming a Gaussian distribution for the photometric errors with mean ``m' + \mathrm{bias}(m')`` and standard deviation ``\sigma(m')``, we have

```math
\phi(m) = \int_{-\infty}^{\infty} \psi(m') \cdot \mathcal{N}\!\left(m \,\Big|\, m' + \mathrm{bias}(m'),\, \sigma(m')\right) \cdot f(m') \, dm'
```

In practice, the integration is truncated to a window ``m \pm w`` (default ``w = 1`` mag) which is sufficient so long as ``\sigma \ll w``.

## Maximum Likelihood Estimation

Following Equation 7 of [Makarov2006](@citet) the log-likelihood is:

```math
\ln \mathcal{L} = \sum_{i=1}^N \ln \phi(m_i) - N \ln \int_{m_\mathrm{min}}^{m_\mathrm{max}} \phi(m) \, dm
```

where ``\sum_{i=1}^N`` indexes over the magnitudes of observed stars and 
the normalization integral ``Z = \int_{m_\mathrm{min}}^{m_\mathrm{max}} \phi(m) \, dm``
is computed over the range of observed magnitudes.

### Optimized Normalization

Substituting the definition of ``\phi(m)`` into ``Z`` and exchanging the order of integration:

```math
Z = \int_{m_\mathrm{min}}^{m_\mathrm{max}} \int \psi(m') \cdot \mathcal{N}(m | \bar{m}(m'), \sigma(m')) \cdot f(m') \, dm' \, dm
```

The inner integral over ``m`` (at fixed ``m'``) is the Gaussian CDF difference:

```math
Z = \int_{x_\mathrm{lo}}^{x_\mathrm{hi}} \psi(x) \cdot f(x) \cdot \left[\Phi\!\left(\frac{m_\mathrm{max} - \bar{m}(x)}{\sigma(x)}\right) - \Phi\!\left(\frac{m_\mathrm{min} - \bar{m}(x)}{\sigma(x)}\right)\right] dx
```

This collapses the 2D normalization integral to a single 1D quadrature over true magnitude ``x``,
significantly reducing computational cost compared to the naive nested quadrature approach.


## References
This page cites the following references:

```@bibliography
Pages = ["theory.md"]
Canonical = false
```