module TRGBDistancesMakieExt

    using TRGBDistances: SobelResult, GLOESSResult # LuminosityFunction, TRGBFitResult
    import Makie as M

    function M.convert_arguments(P::Type{<:M.BarPlot}, r::SobelResult)
        return M.convert_arguments(M.BarPlot, r.bin_centers, r.counts)
    end

    function M.convert_arguments(P::M.PointBased, r::SobelResult) # lines or scatter
        return M.convert_arguments(P, r.bin_centers, r.edge_signal)
    end

end # module