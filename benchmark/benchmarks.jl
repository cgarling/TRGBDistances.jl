import TRGBDistances
using BenchmarkTools
using Distributions: Normal
using LinearAlgebra: BLAS
using Random: Xoshiro
using PrettyTables

function show_benchmarks(results)
    # Collect results
    sorted  = sort(collect(results), by=first)
    names   = [k for (k,_) in sorted]
    trials  = [v for (_,v) in sorted]

    # Pack into matrix
    data = hcat(
        names,
        [BenchmarkTools.prettytime(median(t).time) for t in trials],
        [BenchmarkTools.prettymemory(median(t).memory) for t in trials],
        [median(t).allocs for t in trials]
    )

    # Make pretty table
    pretty_table(data;
        column_labels = ["Benchmark", "Median Time", "Memory", "Allocs"],
        alignment     = [:l, :r, :r, :r]
    )
end

BLAS.set_num_threads(1)

const SUITE = BenchmarkGroup()
SUITE["core"] = BenchmarkGroup()

# Setup data
const model = TRGBDistances.BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
const err_func = x -> TRGBDistances.exp_photerr(x, 1.05, 10.0, 32.0, 0.01)
const complete_func = x -> TRGBDistances.Martin2016_complete(x, 1.0, 26.0, 0.3)
const bias_func = m -> 0.0
const mags = TRGBDistances.observe(Xoshiro(42), model, 200;
                                    err_func=err_func,
                                    complete_func=complete_func,
                                    bias_func=bias_func)
const limits = extrema(mags)

# ϕ — adaptive vs fixed quadrature
SUITE["core"]["phi_adaptive"] = @benchmarkable TRGBDistances.ϕ(24.0, $model, $err_func, $complete_func, $bias_func; quad=:adaptive)
SUITE["core"]["phi_fixed"] = @benchmarkable TRGBDistances.ϕ(24.0, $model, $err_func, $complete_func, $bias_func; quad=:fixed)

# ϕ_norm — adaptive vs fixed quadrature
SUITE["core"]["phi_norm_adaptive"] = @benchmarkable TRGBDistances.ϕ_norm($limits, $model, $err_func, $complete_func, $bias_func; quad=:adaptive)
SUITE["core"]["phi_norm_fixed"] = @benchmarkable TRGBDistances.ϕ_norm($limits, $model, $err_func, $complete_func, $bias_func; quad=:fixed)

# loglikelihood — adaptive vs fixed quadrature
SUITE["core"]["loglikelihood_adaptive"] = @benchmarkable TRGBDistances.loglikelihood($model, $mags, $err_func, $complete_func, $bias_func; quad=:adaptive)
SUITE["core"]["loglikelihood_fixed"] = @benchmarkable TRGBDistances.loglikelihood($model, $mags, $err_func, $complete_func, $bias_func; quad=:fixed)

# old nested quadrature for reference
SUITE["core"]["old_norm"] = @benchmarkable TRGBDistances.quadgk(m -> TRGBDistances.ϕ(m, $model, $err_func, $complete_func, $bias_func), $limits...)[1]


######################################################
# Problem-size-specific benchmarks for edge detection
SUITE["sobel"] = BenchmarkGroup()
SUITE["sobel"]["trgb"] = BenchmarkGroup()
SUITE["gloess"] = BenchmarkGroup()
SUITE["gloess"]["trgb"] = BenchmarkGroup()

const bin_widths = (0.01, 0.05, 0.1)
const magnitude_ranges = ((23.5, 24.5), (23.0, 25.0), (22.5, 25.5))
for (i, width) in enumerate(bin_widths)
    for (j, mag_range) in enumerate(magnitude_ranges)
        SUITE["sobel"]["trgb"]["bin_width=$(width)_mag_range=$(mag_range[2]-mag_range[1])"] = @benchmarkable TRGBDistances.trgb(TRGBDistances.Sobel(bin_width=$width, magnitude_range=$mag_range, response=Normal(0, $width)), $mags)
        SUITE["gloess"]["trgb"]["bandwidth=$(width)_bin_width=$(width)_mag_range=$(mag_range[2]-mag_range[1])"] = @benchmarkable TRGBDistances.trgb(TRGBDistances.GLOESS(bandwidth=$width, bin_width=$width, magnitude_range=$mag_range), $mags)
    end
end

# If not on CI, we'll show a nice table
if get(ENV, "CI", "false") == "false"
    # Run the benchmarks
    results = run(SUITE, verbose=true)

    println("⎯⎯⎯ Core Suite ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
    show_benchmarks(results["core"])
    println("⎯⎯⎯ Sobel Suite ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
    show_benchmarks(results["sobel"]["trgb"])
    println("⎯⎯⎯ GLOESS Suite ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
    show_benchmarks(results["gloess"]["trgb"])
end