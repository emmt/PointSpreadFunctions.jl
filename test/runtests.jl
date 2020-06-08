module TestPointSpreadFunctions

using PointSpreadFunctions
using Test

@testset "PSF models and fitting" begin
    x = 1:20
    y = 1:27
    x0 = 7.2
    y0 = 9.3
    let psf = GaussianPSF(5.4),
        mdl = psf((x .- x0), (y .- y0)'),
        dat = 3.1*mdl + 0.1*randn(size(mdl)),
        wgt = ones(eltype(dat), size(dat))

        @test getfwhm(psf) === psf[1]
        @test psf[:] === (getfwhm(psf),)
        for (psf1, pos1) in (
            PointSpreadFunctions.fit(GaussianPSF(3), (6,11), dat;
                                     nonnegative=true,
                                     maxeval=200),
            PointSpreadFunctions.fit(GaussianPSF(3), (6,11), wgt, dat;
                                     nonnegative=true,
                                     maxeval=200),
        )
            @test psf1[1] ≈ psf[1] rtol=0.1  atol=0.0
            @test pos1[1] ≈ x0     rtol=0.05 atol=0.0
            @test pos1[2] ≈ y0     rtol=0.05 atol=0.0
        end
    end

    let psf = AiryPSF(1.0)
        @test psf[:] === (1.0, 0.0)
        @test getfwhm(psf) ≈ PointSpreadFunctions.AIRY_FWHM
        @test findzeros(psf, 1:5) ≈ [PointSpreadFunctions.AIRY_FIRST_ZERO,
                                     PointSpreadFunctions.AIRY_SECOND_ZERO,
                                     PointSpreadFunctions.AIRY_THIRD_ZERO,
                                     PointSpreadFunctions.AIRY_FOURTH_ZERO,
                                     PointSpreadFunctions.AIRY_FIFTH_ZERO] atol=0.01
    end

    let psf = AiryPSF(5.4, 0.4),
        mdl = psf((x .- x0), (y .- y0)'),
        dat = 3.1*mdl + 0.1*randn(size(mdl)),
        wgt = ones(eltype(dat), size(dat))

        for (psf1, pos1) in (
            PointSpreadFunctions.fit(AiryPSF(5, 0.3), (6,11), dat;
                                     nonnegative=true, rho=(0.03,1e-8),
                                     maxeval=200),
            PointSpreadFunctions.fit(AiryPSF(5, 0.3), (6,11), wgt, dat;
                                     nonnegative=true, rho=(0.03,1e-8),
                                     maxeval=200),
        )
            @test psf1[1] ≈ psf[1] rtol=0.1  atol=0.0
            @test psf1[2] ≈ psf[2] rtol=0.1  atol=0.0
            @test pos1[1] ≈ x0     rtol=0.05 atol=0.0
            @test pos1[2] ≈ y0     rtol=0.05 atol=0.0
        end
    end

    let psf = CauchyPSF(5.4),
        mdl = psf((x .- x0), (y .- y0)'),
        dat = 3.1*mdl + 0.1*randn(size(mdl)),
        wgt = ones(eltype(dat), size(dat))

        for (psf1, pos1) in (
            PointSpreadFunctions.fit(CauchyPSF(4), (6,11), dat;
                                     nonnegative=true,
                                     rho=(0.03,1e-6),
                                     maxeval=200),
            PointSpreadFunctions.fit(CauchyPSF(4), (6,11), wgt, dat;
                                     nonnegative=true,
                                     rho=(0.03,1e-6),
                                     maxeval=200),
        )
            @test psf1[1] ≈ psf[1] rtol=0.1  atol=0.0
            @test pos1[1] ≈ x0     rtol=0.05 atol=0.0
            @test pos1[2] ≈ y0     rtol=0.05 atol=0.0
        end
    end
end

end # module
