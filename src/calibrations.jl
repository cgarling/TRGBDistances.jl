
"""
    Rizzi2007(color, obssym::Symbol, magsym::Symbol, colorsym::Symbol)
Color-dependent TRGB zeropoint magnitude from Rizzi+2007. 

```jldoctest
julia> TRGBDistances.Rizzi2007(1.5, :ACS, :F814W, :F606W) ≈ -4.006
true
```
"""
function Rizzi2007(color, obssym::Symbol, magsym::Symbol, colorsym::Symbol)
    if obssym == :ACS
        if magsym == :F814W
            if colorsym == :F606W
                return -4.06 + 0.2 * (color - 1.23)
            elseif colorsym == :F555W
                return -4.06 + 0.15 * (color - 1.74)
            else
                throw(ArgumentError("`Rizzi2007` for `obssym = :ACS` supports `colorsym` of `:F606W` or `:F555W`, not $colorsym."))
            end
        else
            throw(ArgumentError("`Rizzi2007` for `obssym = :ACS` only supports `magsym == :F814W`."))
        end
    elseif obssym == :WFPC2

    end
end

"""
    Fusco2012(color)
Returns M_{F814W} given a F475W - F814W color.
"""
function Fusco2012(color)
    return -3.63 + 0.08 * color^2 - 0.4 * color

end