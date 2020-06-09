module TestPointSpreadFunctions

using Test
using PointSpreadFunctions
using PointSpreadFunctions:
    AbstractPSF,
    finite_and_nonnegative,
    finite_and_positive,
    throw_bad_argument,
    to_float
import PointSpreadFunctions:
    check_structure,
    getfwhm,
    isvalid,
    parameters
using PointSpreadFunctions.Fitting:
    objfun

struct MyPSF <: AbstractPSF{3}
    prm::NTuple{3,Float64}
    MyPSF(a::Real, b::Real, c::Real) =
        new(map(x -> to_float(Float64, x), (a,b,c)))
end

parameters(P::MyPSF) = P.prm
getfwhm(P::MyPSF) = P[1]
isvalid(P::MyPSF) = (_check_fwhm(P) && _check_b(P) && _check_c(P))
function check_structure(P::MyPSF)
    _check_fwhm(P) || throw_bad_argument("invalid FWHM (",getfwhm(P),")")
    _check_b(P) || throw_bad_argument("invalid parameter B (",_get_b(P),")")
    _check_c(P) || throw_bad_argument("invalid parameter C (",_get_c(P),")")
end

_get_b(P::MyPSF) = P[2]
_get_c(P::MyPSF) = P[3]

_check_fwhm(P::MyPSF) = finite_and_nonnegative(getfwhm(P))
_check_b(P::MyPSF) = finite_and_positive(_get_b(P))
_check_c(P::MyPSF) = (c = _get_c(P); isfinite(c) && 0 < c < 1)

@testset "PSF models and fitting" begin

    @test finite_and_positive(NaN) == false
    @test finite_and_positive(+Inf) == false
    @test finite_and_positive(-1) == false
    @test finite_and_positive(0) == false
    @test finite_and_positive(π) == true

    @test finite_and_nonnegative(NaN) == false
    @test finite_and_nonnegative(+Inf) == false
    @test finite_and_nonnegative(-1) == false
    @test finite_and_nonnegative(0) == true
    @test finite_and_nonnegative(π) == true

    @test_throws ArgumentError throw_bad_argument("oops!")
    @test_throws ArgumentError throw_bad_argument("oops! (", π, ")")

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

    @test_throws ArgumentError check_structure(MyPSF(-1.0,1,0.3))
    @test_throws ArgumentError check_structure(AiryPSF(-1.0))
    @test_throws ArgumentError check_structure(AiryPSF(1.0, 1.1))
    @test_throws ArgumentError check_structure(GaussianPSF(-1.0))
    @test_throws ArgumentError check_structure(CauchyPSF(Inf))
    @test_throws ArgumentError check_structure(MoffatPSF(0.0,0.5))
    @test_throws ArgumentError check_structure(MoffatPSF(1.0,-0.2))
    @test isvalid(AiryPSF(1.0, 1.1)) == false
    @test isvalid(AiryPSF(1.0, 0.3)) == true
    @test isvalid(GaussianPSF(-1.0)) == false
    @test isvalid(GaussianPSF(1.0)) == true
    @test isvalid(CauchyPSF(NaN)) == false
    @test isvalid(CauchyPSF(1.0)) == true
    @test isvalid(MoffatPSF(0.0,0.5)) == false
    @test isvalid(MoffatPSF(1.0,-0.2)) == false
    @test isvalid(MoffatPSF(1.0,0.4)) == true

    let psf = AiryPSF(1.0)
        @test psf[:] === (1.0, 0.0)
        @test getfwhm(psf) ≈ PointSpreadFunctions.AIRY_FWHM
        @test findzeros(psf, 1:5) ≈ [PointSpreadFunctions.AIRY_FIRST_ZERO,
                                     PointSpreadFunctions.AIRY_SECOND_ZERO,
                                     PointSpreadFunctions.AIRY_THIRD_ZERO,
                                     PointSpreadFunctions.AIRY_FOURTH_ZERO,
                                     PointSpreadFunctions.AIRY_FIFTH_ZERO] atol=0.01
    end
    for psf in (AiryPSF(1.1), AiryPSF(1.2, 0.3),
                CauchyPSF(1.8), GaussianPSF(2.7),
                MoffatPSF(1.4, 0.5))
        # Check peak PSF is 1 at (0,0).
        @test psf(0) == 1
        @test psf(0, 0) == 1

        for (x,y) in ((1.23, -0.12), (-0.7, sqrt(2)))
            @test psf(x, y) ≈ psf(hypot(x,y))
            for T in (Float32, Float64)
                @test psf(T, x, y) ≈ T(psf(x,y))
                @test psf(T, hypot(x,y)) ≈ T(psf(hypot(x,y)))
                @test psf(T, x, y) ≈ psf(T, hypot(x,y))
            end
        end
    end

    let psf = AiryPSF(5.4, 0.33),
        mdl = psf((x .- x0), (y .- y0)'),
        dat = 3.1*mdl + 0.1*randn(size(mdl)),
        wgt = ones(eltype(dat), size(dat))

        for (psf1, pos1) in (
            PointSpreadFunctions.fit(AiryPSF(5, 0.3), (7,9), dat;
                                     nonnegative=true, rho=(0.03,1e-8),
                                     maxeval=200),
            PointSpreadFunctions.fit(AiryPSF(5, 0.3), (7,9), wgt, dat;
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
