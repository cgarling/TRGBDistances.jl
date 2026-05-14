
"""
    Rizzi2007(color, obssym::Symbol, magsym::Symbol, colorsym::Symbol)

Color-dependent TRGB zeropoint magnitude from [Rizzi2007](@citet).

Supported filter combinations are:

| `obssym` | `magsym` | `colorsym` | Equation |
|---|---|---|---|
| `:JC` | `:I` | `:VI` | `-4.05 + 0.217 * (color - 1.60)` |
| `:ACS` | `:F814W` | `:F606W` | `-4.06 + 0.200 * (color - 1.23)` |
| `:ACS` | `:F814W` | `:F555W` | `-4.06 + 0.150 * (color - 1.74)` |
| `:WFPC2` | `:F814W` | `:F555W` | `-4.01 + 0.180 * (color - 1.58)` |
| `:WFPC2` | `:F814W` | `:F606W` | `-4.01 + 0.150 * (color - 1.12)` |

# Returns
- `Float64`: The TRGB zeropoint magnitude in the requested band.

# Examples
```jldoctest
julia> TRGBDistances.Rizzi2007(1.5, :ACS, :F814W, :F606W) ≈ -4.006
true
```
"""
function Rizzi2007(color, obssym::Symbol, magsym::Symbol, colorsym::Symbol)
    if obssym == :ACS && magsym == :F814W && colorsym == :F606W
        return -4.06 + 0.2 * (color - 1.23)
    elseif obssym == :ACS && magsym == :F814W && colorsym == :F555W
        return -4.06 + 0.15 * (color - 1.74)
    elseif obssym == :WFPC2 && magsym == :F814W && colorsym == :F555W
        return -4.01 + 0.18 * (color - 1.58)
    elseif obssym == :WFPC2 && magsym == :F814W && colorsym == :F606W
        return -4.01 + 0.15 * (color - 1.12)
    elseif obssym == :JC && magsym == :I && colorsym == :VI
        return -4.05 + 0.217 * (color - 1.6)
    end
    throw(ArgumentError("Rizzi2007 unsupported arguments"))
end

"""
    Fusco2012(color)

Color-dependent TRGB zeropoint magnitude from [Fusco2012](@citet),
``M_{F814W} = -3.63 + 0.08 * (F475W - F814W)^2 - 0.4 * (F475W - F814W)``.

```jldoctest
julia> TRGBDistances.Fusco2012(1.5) ≈ -4.05
true
```
"""
Fusco2012(color) = -3.63 + 0.08 * color^2 - 0.4 * color

"""
    Jang2017(color, obssym::Symbol, magsym::Symbol, colorsym::Symbol)

Color-dependent TRGB zeropoint magnitudes from [Jang2017](@citet).
Supported filter combinations are:

| `obssym` | `magsym` | `colorsym` | Equation |
|---|---|---|---|
| `:JC` | `:I` | `:V` | `-4.016 + 0.091*(color - 1.5)^2 - 0.007*(color - 1.5)` |
| `:ACS` | `:F814W` | `:F606W` | `-4.015 + 0.159*(color - 1.1)^2 - 0.047*(color - 1.1)` |
| `:ACS` | `:F814W` | `:F555W` | `-4.011 + 0.116*(color - 1.6)^2 - 0.043*(color - 1.6)` |
| `:WFC3` | `:F814W` | `:F606W` | `-4.024 + 0.150*(color - 1.1)^2 - 0.050*(color - 1.1)` |
| `:WFC3` | `:F814W` | `:F555W` | `-4.019 + 0.113*(color - 1.6)^2 - 0.048*(color - 1.6)` |
| `:WFPC2` | `:F814W` | `:F555W` | `-4.007 + 0.123*(color - 1.5)^2 - 0.051*(color - 1.5)` |
| `:WFPC2` | `:F814W` | `:F606W` | `-3.984 + 0.179*(color - 1.0)^2 - 0.034*(color - 1.0)` |

```jldoctest
julia> TRGBDistances.Jang2017(1.5, :ACS, :F814W, :F606W) ≈ -4.00836
true
```
"""
function Jang2017(color::Real, obssym::Symbol, magsym::Symbol, colorsym::Symbol)
    if obssym == :JC && magsym == :I && colorsym == :V
        return -4.016 + 0.091*(color - 1.5)^2 - 0.007*(color - 1.5)
    elseif obssym == :ACS && magsym == :F814W && colorsym == :F606W
        return -4.015 + 0.159*(color - 1.1)^2 - 0.047*(color - 1.1)
    elseif obssym == :ACS && magsym == :F814W && colorsym == :F555W
        return -4.011 + 0.116*(color - 1.6)^2 - 0.043*(color - 1.6)
    elseif obssym == :WFC3 && magsym == :F814W && colorsym == :F606W
        return -4.024 + 0.150*(color - 1.1)^2 - 0.050*(color - 1.1)
    elseif obssym == :WFC3 && magsym == :F814W && colorsym == :F555W
        return -4.019 + 0.113*(color - 1.6)^2 - 0.048*(color - 1.6)
    elseif obssym == :WFPC2 && magsym == :F814W && colorsym == :F555W
        return -4.007 + 0.123*(color - 1.5)^2 - 0.051*(color - 1.5)
    elseif obssym == :WFPC2 && magsym == :F814W && colorsym == :F606W
        return -3.984 + 0.179*(color - 1.0)^2 - 0.034*(color - 1.0)
    else
        throw(ArgumentError("Jang2017 unsupported arguments"))
    end
end

"""
    Freedman2019()

TRGB zeropoint from [Freedman2019](@citet) of -4.049 ± 0.022 (stat) ± 0.039 (sys) in I and HST/F814W.

```jldoctest
julia> TRGBDistances.Freedman2019() ≈ -4.049
true
```
"""
Freedman2019() = -4.049

"""
    Freedman2020(color::Union{Real, Nothing}=nothing,
                 magsym::Symbol=:I)

TRGB zeropoint calibration from [Freedman2020](@citet).

Supported parameter combinations are:

| `magsym` | Assumed `color` convention | Equation |
|---|---|---|
| `:I` | none | `-4.047` |
| `:F814W` | none | `-4.054` |
| `:J` | `J - K` | `-5.14 - 0.85 * (color - 1)` |
| `:H` | `J - K` | `-5.95 - 1.62 * (color - 1)` |
| `:K` | `J - K` | `-6.14 - 1.85 * (color - 1)` |
| `:V` | `V - I` | `-2.25 + (color - 1.8)` |

# Arguments
- `color`: Optional color term used for color-dependent calibrations.
  For `:J`, `:H`, and `:K`, the assumed color is `J - K`.
  For `:V`, the assumed color is `V - I`.
- `magsym`: Magnitude/filter band in which the TRGB zeropoint is returned.

# Returns
- `Float64`: The TRGB zeropoint magnitude in the requested band.

# Examples
```jldoctest
julia> TRGBDistances.Freedman2020(nothing, :I) ≈ -4.047
true

julia> TRGBDistances.Freedman2020(1.2, :J) ≈ -5.31
true
```
"""
function Freedman2020(color::Union{Real, Nothing}=nothing, magsym::Symbol=:I)
    if magsym == :I
        return -4.047
    elseif magsym == :F814W
        return -4.054
    elseif !isnothing(color) && magsym == :J # color = J-K
        return -5.14 - 0.85 * (color - 1)
    elseif !isnothing(color) && magsym == :H # color = J-K
        return -5.95 - 1.62 * (color - 1) 
    elseif !isnothing(color) && magsym == :K # color = J-K
        return -6.14 - 1.85 * (color - 1)
    elseif !isnothing(color) && magsym == :V # color = V-I
        return -2.25 + (color - 1.8)
    end
    throw(ArgumentError("Freedman2020 calibration unsupported arguments"))
end

"""
    Yuan2019(obssym::Symbol=:ACS, magsym::Symbol=:F814W)

TRGB zeropoint from [Yuan2019](@citet), which is -3.97 ± 0.046 in F814W and I.

```jldoctest
julia> TRGBDistances.Yuan2019() ≈ -3.97
true
```
"""
Yuan2019() = -3.97

"""
    Cerny2021(color::Union{Real, Nothing}=nothing,
               magsym::Symbol=:I,
               colorsym::Union{Symbol, Nothing}=nothing)

TRGB zeropoint calibration from [Cerny2021](@citet).

Supported parameter combinations are:

| `magsym` | `colorsym` | Equation |
|---|---|---|
| `:I` | `nothing` | `-4.056` |
| `:F814W` | `nothing` | `-4.056` |
| `:V` | `nothing` | `-2.26` |
| `:J` | `nothing` | `-5.16` |
| `:H` | `nothing` | `-6.03` |
| `:K` | `nothing` | `-6.16` |
| `:I` | `:VI` | `-4.06` |
| `:F814W` | `:VI` | `-4.06` |
| `:V` | `:VI` | `-2.26 + (color - 1.8)` |
| `:J` | `:JK` | `-5.16 - 0.85 * (color - 1)` |
| `:H` | `:JK` | `-6.03 - 1.72 * (color - 1)` |
| `:K` | `:JK` | `-6.16 - 1.85 * (color - 1)` |

# Arguments
- `color`: Optional color term used for color-dependent calibrations.
- `magsym`: Magnitude/filter band in which the TRGB zeropoint is returned.
- `colorsym`: Symbol describing the color combination associated with `color`.

# Returns
- `Float64`: The TRGB zeropoint magnitude in the requested band.

# Examples
```jldoctest
julia> TRGBDistances.Cerny2021() ≈ -4.056
true

julia> TRGBDistances.Cerny2021(1.2, :J, :JK) ≈ -5.33
true
```
"""
function Cerny2021(color::Union{Real, Nothing}=nothing, magsym::Symbol=:I, colorsym::Union{Symbol, Nothing}=nothing)
    if isnothing(color) && (magsym == :I || magsym == :F814W)
        return -4.056
    elseif isnothing(color) && magsym == :V
        return -2.26
    elseif isnothing(color) && magsym == :J
        return -5.16
    elseif isnothing(color) && magsym == :H
        return -6.03
    elseif isnothing(color) && magsym == :K
        return -6.16
    elseif !isnothing(color) && (magsym == :I || magsym == :F814W) && colorsym == :VI
        return -4.06
    elseif !isnothing(color) && magsym == :V && colorsym == :VI
        return -2.26 + (color - 1.8) # assuming color is V-I
    elseif !isnothing(color) && magsym == :J && colorsym == :JK
        return -5.16 - 0.85 * (color - 1) # color is J-K
    elseif !isnothing(color) && magsym == :H && colorsym == :JK
        return -6.03 - 1.72 * (color - 1) # color is J-K
    elseif !isnothing(color) && magsym == :K && colorsym == :JK
        return -6.16 - 1.85 * (color - 1) # color is J-K
    end
    throw(ArgumentError("Cerny2021 calibration unsupported arguments"))
end

"""
    Hoyt2018(color::Real, magsym::Symbol=:K, colorsym::Symbol=:JK)

TRGB zeropoint from [Hoyt2018](@citet). Supported parameter combinations are:

| `magsym` | `colorsym` | Assumed `color` | Equation |
|---|---|---|---|
| `:J` | `:JK` | `J - K` | `-5.14 - 0.85 * (color - 1)` |
| `:H` | `:JK` | `J - K` | `-5.94 - 1.62 * (color - 1)` |
| `:K` | `:JK` | `J - K` | `-6.14 - 1.85 * (color - 1)` |
| `:J` | `:JH` | `J - H` | `-5.13 - 1.11 * (color - 0.8)` |
| `:H` | `:JH` | `J - H` | `-5.93 - 2.11 * (color - 0.8)` |
| `:K` | `:JH` | `J - H` | `-6.13 - 2.41 * (color - 0.8)` |

# Arguments
- `color`: Color term used in the calibration relation.
  The interpretation depends on `colorsym`.
- `magsym`: Magnitude/filter band in which the TRGB zeropoint is returned.
- `colorsym`: Symbol specifying the color convention associated with `color`.

# Returns
- `Float64`: The TRGB zeropoint magnitude in the requested band.

```jldoctest
julia> TRGBDistances.Hoyt2018(1.5, :K) ≈ -6.14 - 1.85 * (1.5 - 1.0)
true
```
"""
function Hoyt2018(color::Real, magsym::Symbol=:K, colorsym::Symbol=:JK)
    if magsym == :J && colorsym == :JK
        return -5.14 - 0.85 * (color - 1)
    elseif magsym == :H && colorsym == :JK
        return -5.94 - 1.62 * (color - 1)
    elseif magsym == :K && colorsym == :JK
        return -6.14 - 1.85 * (color - 1)
    elseif magsym == :J && colorsym == :JH
        return -5.13 - 1.11 * (color - 0.8)
    elseif magsym == :H && colorsym == :JH
        return -5.93 - 2.11 * (color - 0.8)
    elseif magsym == :K && colorsym == :JH
        return -6.13 - 2.41 * (color - 0.8)
    end
    throw(ArgumentError("Hoyt2018 unsupported arguments"))
end
