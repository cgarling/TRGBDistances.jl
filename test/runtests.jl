using ADTypes: AutoForwardDiff
using TRGBDistances
using TRGBDistances: fit  # disambiguate from StatsAPI/Distributions
using TRGBDistances: exp_photerr, Martin2016_complete
using Test
using Random
using Statistics: mean
using StableRNGs: StableRNG

# Run doctests first
using Documenter: DocMeta, doctest
DocMeta.setdocmeta!(TRGBDistances, :DocTestSetup, :(using TRGBDistances, Optim, ForwardDiff); recursive=true)
doctest(TRGBDistances)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
# const err_func      = m -> 0.05
# const complete_func = m -> 1.0
const err_func      = x -> exp_photerr(x, 1.05, 10.0, 32.0, 0.01)
const complete_func = x -> Martin2016_complete(x, 1.0, 26.0, 0.3)
const bias_func     = m -> 0.0

@testset "TRGBDistances.jl" begin

    # -------------------------------------------------------------------
    @testset "BrokenPowerLaw — definition" begin
        using TRGBDistances: BrokenPowerLaw, ψ
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        # At the TRGB from the bright (AGB) side: 10^(c*0) = 1.0
        @test ψ(model, 24.0 - 1e-12) ≈ 1.0 atol=1e-6
        # At the TRGB from the faint (RGB) side: 10^(a*0 + b) = 10^b
        @test ψ(model, 24.0) ≈ exp10(0.2) atol=1e-10
        # Discontinuity magnitude = 10^b
        @test ψ(model, 24.0) / ψ(model, 24.0 - 1e-12) ≈ exp10(0.2) atol=1e-4
        # RGB side is increasing (fainter = larger ψ)
        @test ψ(model, 25.0) > ψ(model, 24.0)
        # AGB side: brighter stars are rarer (for c > 0, ψ decreases going brighter)
        @test ψ(model, 23.0) < 1.0
        # Callable form matches ψ
        @test model(24.5) == ψ(model, 24.5)
        # Promote constructor
        m = BrokenPowerLaw(24, 0.3, 0.2, 0.1)
        @test m.m_trgb isa Float64
    end

    # -------------------------------------------------------------------
    @testset "ϕ — convolution integral" begin
        using TRGBDistances: BrokenPowerLaw, ϕ
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        # ϕ should be positive everywhere near the TRGB
        for m in 22.0:0.5:26.0
            @test ϕ(m, model, err_func, complete_func, bias_func) >= 0
        end
        # With larger error, ϕ should be smoother (less peaked) than ψ
        # @test ϕ(24.0, model, x -> 0.05, complete_func, bias_func) > ϕ(24.0, model, x -> 0.1, complete_func, bias_func)
        # Completeness of 0 should give ϕ = 0
        @test ϕ(24.0, model, err_func, m -> 0.0, bias_func) ≈ 0.0 atol=1e-10
    end

    # -------------------------------------------------------------------
    @testset "ϕ_norm — normalization" begin
        using TRGBDistances: BrokenPowerLaw, ϕ_norm
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        limits = (22.0, 27.0)
        # Normalization must be positive
        Z = ϕ_norm(limits, model, err_func, complete_func, bias_func)
        @test Z > 0
        # Normalization should agree with brute-force nested quadrature
        using QuadGK: quadgk
        Z_nested = quadgk(m -> ϕ(m, model, err_func, complete_func, bias_func), limits...)[1]
        @test Z ≈ Z_nested rtol=1e-3
    end

    # -------------------------------------------------------------------
    @testset "loglikelihood — formula" begin
        using TRGBDistances: BrokenPowerLaw, loglikelihood
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        mags = [23.5, 24.0, 24.2, 24.5, 25.0]
        ll = loglikelihood(model, mags, err_func, complete_func, bias_func)
        @test isfinite(ll)
        @test ll <= 0  # log-likelihood is not necessarily ≤ 0 in general, but check it's finite

        # Likelihood consistency: evaluated at the same θ, the likelihood should be finite
        # and negative-LL should be less for data generated from the model.
        model_eval = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        ll_val = loglikelihood(model_eval, mags, err_func, complete_func, bias_func)
        @test isfinite(ll_val)
    end

    # -------------------------------------------------------------------
    @testset "Inverse-CDF sampling — BrokenPowerLaw" begin
        using TRGBDistances: BrokenPowerLaw
        using TRGBDistances: TRGBDistances
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        # CDF at m_trgb from the bright side should be less than at m_trgb
        @test TRGBDistances.cdf_bpl(model, 23.5) < TRGBDistances.cdf_bpl(model, 24.0)
        @test TRGBDistances.cdf_bpl(model, 24.0) < TRGBDistances.cdf_bpl(model, 25.0)
        # CDF is in [0,1]
        for m in 20.0:0.5:30.0
            c = TRGBDistances.cdf_bpl(model, m)
            @test 0 <= c <= 1
        end
        # Quantile is inverse of CDF
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]
            m = TRGBDistances.quantile(model, p)
            @test TRGBDistances.cdf_bpl(model, m) ≈ p atol=1e-4
        end
        # rand produces values
        rng = StableRNG(1234)
        samples = rand(rng, model, 100)
        @test length(samples) == 100
        @test all(isfinite, samples)
        # All samples from a model with m_trgb=24 should be >= m_trgb - some offset
        # (AGB side can extend brighter, but not infinitely far for c > 0)
        @test all(s -> s >= 20.0, samples)  # no sample unreasonably bright
    end

    # -------------------------------------------------------------------
    @testset "observe — synthetic catalog generation" begin
        using TRGBDistances: BrokenPowerLaw, observe
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        err(m)   = 0.05 + 0.05 * max(m - 25.0, 0.0)
        compl(m) = m < 26.5 ? 1.0 : 0.0
        bias(m)  = 0.0
        rng = StableRNG(1234)
        mags = observe(rng, model, 200; err_func=err, complete_func=compl, bias_func=bias)
        @test length(mags) == 200
        @test all(isfinite, mags)
        # All observed mags should be below the completeness limit
        @test all(m -> m < 27.0, mags)
    end

    # -------------------------------------------------------------------
    @testset "logprior interface" begin
        using TRGBDistances: logprior
        θ = [24.0, 0.3, 0.2, 0.1]
        # Flat prior returns 0
        @test logprior(nothing, θ) == 0.0
        # Function prior
        @test logprior(θ -> -sum(θ .^ 2), θ) == -sum(θ .^ 2)
        # Tuple prior (only first element)
        using Distributions
        prior = (Normal(24.0, 0.5), nothing, nothing, nothing)
        lp = logprior(prior, θ)
        @test lp ≈ logpdf(Normal(24.0, 0.5), 24.0)
    end

    # -------------------------------------------------------------------
    @testset "fit API — OptimJL backend" begin
        # Only run if Optim is available
        try
            using TRGBDistances: BrokenPowerLaw, fit, OptimJL, observe, TRGBFitResult
            using Optim: Optim
            model_true = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
            rng = StableRNG(1234)
            mags = observe(rng, model_true, 500;
                           err_func=err_func, complete_func=complete_func, bias_func=bias_func)
            x0 = [24.2, 0.3, 0.2, 0.1]  # start very close to truth
            result = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0)
            @test result isa TRGBFitResult
            @test isfinite(result.minimum)
        catch e
            if e isa ErrorException && occursin("requires loading Optim", e.msg)
                @test_broken false  # Optim not loaded
            else
                rethrow(e)
            end
        end
    end

    # -------------------------------------------------------------------
    @testset "filter_mags" begin
        using TRGBDistances: filter_mags
        colors = [1.0, 1.2, 1.5, 2.0, 0.5]
        mags   = [24.0, 24.5, 25.0, 25.5, 26.0]
        ridge_colors = [1.0, 1.1, 1.2, 1.3, 1.4]
        ridge_mags   = [23.5, 24.0, 24.5, 25.0, 25.5]
        width_func(m) = 0.3
        idx = filter_mags(colors, mags, ridge_colors, ridge_mags, width_func)
        @test idx isa Vector{Int}
        # Stars with very different colors should not be included
        @test 4 ∉ idx  # color=2.0 is likely too far from ridge
    end

    # -------------------------------------------------------------------
    @testset "Rizzi2007 calibration" begin
        using TRGBDistances: Rizzi2007
        cal = Rizzi2007(1.5, :ACS, :F814W, :F606W)
        @test cal ≈ -4.006 atol=1e-3
        @test_throws ArgumentError Rizzi2007(1.5, :ACS, :F814W, :F435W)
    end

    # -------------------------------------------------------------------
    @testset "AD backends — build_gradient / build_hessian" begin
        using ForwardDiff: ForwardDiff
        using LinearAlgebra: I
        using TRGBDistances: build_gradient, build_hessian
        f  = x -> sum(x .^ 2)
        x  = [1.0, 2.0, 3.0]

        @testset "ForwardDiff" begin
            g, g! = build_gradient(AutoForwardDiff(), f)
            @test g(x) ≈ 2 .* x
            G = zeros(3)
            g!(G, x)
            @test G ≈ 2 .* x

            h, h! = build_hessian(AutoForwardDiff(), f)
            @test h(x) ≈ 2 * Matrix(I, 3, 3)
            H = zeros(3, 3)
            h!(H, x)
            @test H ≈ 2 * Matrix(I, 3, 3)
        end
    end

    # -------------------------------------------------------------------
    @testset "fit API — first/second-order with ForwardDiff" begin
        using Optim: Optim
        using ForwardDiff: ForwardDiff
        using TRGBDistances: BrokenPowerLaw, fit, OptimJL, observe, TRGBFitResult
        model_true = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        rng  = StableRNG(1234)
        mags = observe(rng, model_true, 500;
                       err_func=err_func, complete_func=complete_func, bias_func=bias_func)
        x0 = [24.2, 0.3, 0.2, 0.1]

        r_bfgs = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
                                   backend=OptimJL(method=Optim.BFGS(), ad=AutoForwardDiff()))
        @test r_bfgs isa TRGBFitResult
        @test r_bfgs.converged
        @test isfinite(r_bfgs.minimum)

        r_newton = TRGBDistances.fit(BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
                                     backend=OptimJL(method=Optim.NewtonTrustRegion(), ad=AutoForwardDiff()))
        @test r_newton isa TRGBFitResult
        @test r_newton.converged
        @test isfinite(r_newton.minimum)
    end

    # -------------------------------------------------------------------
    # This is pretty slow, but leaving it for now (could reduce samples, warmup, etc)
    # @testset "sample API — DynamicHMCJL backend" begin
    #     using DynamicHMC: DynamicHMC
    #     using LogDensityProblems: LogDensityProblems
    #     using ForwardDiff: ForwardDiff
    #     using Distributions
    #     using TRGBDistances: BrokenPowerLaw, observe, TRGBChain
    #     model_true = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
    #     rng  = StableRNG(1234)
    #     mags = observe(rng, model_true, 300;
    #                    err_func=err_func, complete_func=complete_func, bias_func=bias_func)
    #     x0    = [24.0, 0.3, 0.2, 0.1]
    #     prior = (Normal(24.0, 0.5), nothing, nothing, nothing)
    #     chain = TRGBDistances.sample(
    #         BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
    #         backend=DynamicHMCJL(nsamples=200, n_warmup=100, ad=AutoForwardDiff()),
    #         prior=prior, rng=StableRNG(1234),
    #     )
    #     @test chain isa TRGBChain
    #     @test size(chain.samples, 1) == 4    # n_params
    #     @test size(chain.samples, 2) == 200  # nsamples
    #     @test length(chain.logprob) == 200
    #     @test 0 <= chain.acceptance_fraction <= 1
    # end

    # -------------------------------------------------------------------
    @testset "sample API — AdvancedMHJL backend" begin
        using AdvancedMH: AdvancedMH
        using Distributions
        using TRGBDistances: BrokenPowerLaw, observe, TRGBChain
        model_true = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        rng  = StableRNG(42)
        mags = observe(rng, model_true, 300;
                       err_func=err_func, complete_func=complete_func, bias_func=bias_func)
        x0    = [24.0, 0.3, 0.2, 0.1]
        prior = (Normal(24.0, 0.5), nothing, nothing, nothing)
        chain = TRGBDistances.sample(
            BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
            backend=AdvancedMHJL(nsamples=500, nburnin=100, proposal_scale=0.01),
            prior=prior, rng=StableRNG(42),
        )
        @test chain isa TRGBChain
        @test size(chain.samples, 1) == 4     # n_params
        @test size(chain.samples, 2) == 500   # nsamples
        @test length(chain.logprob) == 500
        @test 0 <= chain.acceptance_fraction <= 1
        @test all(isfinite, chain.samples)
    end

    # -------------------------------------------------------------------
    @testset "sample API — AffineInvariantMCMCJL backend" begin
        using AffineInvariantMCMC: AffineInvariantMCMC
        using Distributions
        using TRGBDistances: BrokenPowerLaw, observe, TRGBChain
        model_true = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        rng = StableRNG(7)
        mags= observe(rng, model_true, 300;
                      err_func=err_func, complete_func=complete_func, bias_func=bias_func)
        x0 = [24.0, 0.3, 0.2, 0.1]
        prior = (Normal(24.0, 0.5), nothing, nothing, nothing)
        chain = TRGBDistances.sample(
            BrokenPowerLaw, mags, err_func, complete_func, bias_func, x0;
            backend=AffineInvariantMCMCJL(nsamples=100, nburnin=50, nwalkers=6),
            prior=prior, rng=StableRNG(7))
        @test chain isa TRGBChain
        @test size(chain.samples, 1) == 4       # n_params
        @test size(chain.samples, 2) == 6 * 100  # n_walkers * n_samples
        @test length(chain.logprob) == 6 * 100
        @test 0 <= chain.acceptance_fraction <= 1
        @test all(isfinite, chain.samples)
    end

    # -------------------------------------------------------------------
    @testset "Sobel edge detection" begin
        using TRGBDistances: BrokenPowerLaw, observe, sobel_trgb, SobelResult
        # Generate a synthetic dataset
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        rng   = StableRNG(99)
        mags  = observe(rng, model, 500;
                        err_func, complete_func,
                        bias_func, upper_limit=2.0)

        # Pure Sobel (no pre-smoothing) — structural tests
        result = sobel_trgb(mags; bin_width=0.1, magnitude_range=(23.5, 24.5))
        @test result isa SobelResult
        @test length(result.bin_centers) == length(result.edge_signal)
        @test length(result.bin_centers) == length(result.counts)
        @test all(b -> 23.4 <= b <= 24.6, result.bin_centers)  # bin centers should be within the specified range (with some padding)
        @test sum(result.counts) == count(m -> 23.5 <= m <= 24.5, mags)
        @test result.m_trgb ≈ 24.0 atol=0.1  # Should be close to true value

        # With Gaussian pre-smoothing — structural tests
        using Distributions
        result_smooth = sobel_trgb(mags; bin_width=0.1, response=Normal(0, 0.05), magnitude_range=(23.5, 24.5))
        @test result_smooth isa SobelResult
        @test result.m_trgb ≈ result_smooth.m_trgb atol=0.01  # Should give similar TRGB estimate
    end

    # -------------------------------------------------------------------
    @testset "GLOESS edge detection" begin
        using TRGBDistances: BrokenPowerLaw, observe, gloess_smooth, gloess_trgb, GLOESSResult
        model = BrokenPowerLaw(24.0, 0.3, 0.2, 0.1)
        rng   = StableRNG(123)
        mags  = observe(rng, model, 500;
                        err_func, complete_func,
                        bias_func, upper_limit=2.0)

        # Structural tests
        result = gloess_trgb(mags; bandwidth=0.2, bin_width=0.1, magnitude_range=(23.5, 24.5))
        @test result isa GLOESSResult
        @test length(result.bin_centers) == length(result.edge_signal)
        @test length(result.bin_centers) == length(result.smoothed_counts)
        @test length(result.bin_centers) == length(result.counts)
        @test sum(result.counts) == count(m -> 23.5 <= m <= 24.5, mags)
        @test all(b -> 23.4 <= b <= 24.9, result.bin_centers)  # bin centers should be within the specified range (with some padding)
        @test result.m_trgb ≈ 24.0 atol=0.1  # Should be close to true value

        # gloess_smooth standalone
        bc = collect(22.0:0.1:26.0)
        n  = [i > 20 ? 10.0 : 1.0 for i in eachindex(bc)]
        sm = gloess_smooth(n, bc; bandwidth=0.2)
        @test length(sm) == length(bc)
        @test all(isfinite, sm)
        # Smoothed LF still rises toward faint end
        @test sm[end] > sm[1]
    end

end
