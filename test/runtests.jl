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

struct MyPSF2 <: AbstractPSF{0}; end

parameters(P::MyPSF) = P.prm
parameters(P::MyPSF2) = ()
getfwhm(P::MyPSF) = P[1]
getfwhm(P::MyPSF2) = 1.0
# Purposely not extend isvalid and check_structure for MyPSF2 to check
# default implementations.
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

    @test isvalid(MyPSF2()) == true
    @test check_structure(MyPSF2()) == true

    @test_throws ArgumentError throw_bad_argument("oops!")
    @test_throws ArgumentError throw_bad_argument("oops! (", π, ")")

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

    # Tests for any PSF models.
    for psf in (AiryPSF(1.1), AiryPSF(1.2, 0.3),
                CauchyPSF(1.8), GaussianPSF(2.7),
                MoffatPSF(1.4, 0.5))
        # Check peak PSF is 1 at (0,0).
        @test psf(0) == 1
        @test psf(0, 0) == 1

        # Check FWHM.
        let fwhm = getfwhm(psf)
            @test psf(fwhm/2) ≈ 0.5
            @test psf(fwhm/2, 0) ≈ 0.5
            @test psf(0, fwhm/2) ≈ 0.5
        end

        # Check show extension.
        io = IOBuffer()
        show(io, psf)
        str = String(take!(io))
        tmp = repr(typeof(psf))
        # Note: `findlast(char,str)` is not available until Julia 1.3 and
        # is not in `Compat` so we have to do it ourselves.
        i = findlast(==('.'), tmp)
        pfx = (i === nothing ? tmp : tmp[i+1:end])
        @test startswith(str, pfx*"(")
        @test endswith(str, ")")

        for (x,y) in ((1.23, -0.12), (-0.7, sqrt(2)))
            @test psf(x, y) ≈ psf(hypot(x,y))
            for T in (Float32, Float64)
                @test psf(T, x, y) ≈ T(psf(x,y))
                @test psf(T, hypot(x,y)) ≈ T(psf(hypot(x,y)))
                @test psf(T, x, y) ≈ psf(T, hypot(x,y))
            end
        end
    end

    # Specific tests for AiryPSF.
    let psf = AiryPSF(1.0)
        @test psf[:] === (1.0, 0.0)
        @test getfwhm(psf) ≈ PointSpreadFunctions.AIRY_FWHM
        @test findzeros(psf, 1:5) ≈ [PointSpreadFunctions.AIRY_FIRST_ZERO,
                                     PointSpreadFunctions.AIRY_SECOND_ZERO,
                                     PointSpreadFunctions.AIRY_THIRD_ZERO,
                                     PointSpreadFunctions.AIRY_FOURTH_ZERO,
                                     PointSpreadFunctions.AIRY_FIFTH_ZERO] atol=0.01
    end

    # Test fitting.
    x = 1:15
    y = 1:17
    x0 = 6.2
    y0 = 7.3
    α = 3.1
    σ = 0.03
    xinit = 6
    yinit = 7
    roi = (2:last(x)-1, :)
    for (psf, initialpsf) in (
        (GaussianPSF(5.4), GaussianPSF(5)),
        (CauchyPSF(5.4), CauchyPSF(4)),
        (AiryPSF(5.4, 0.33), AiryPSF(5, 0.3)),
        (MoffatPSF(5.2, 1.5), MoffatPSF(5, 1)),
    )

        mdl = psf((x .- x0), (y .- y0)')
        dat = α*mdl + σ*randn(size(mdl))
        wgt = ones(eltype(dat), size(dat))

        if psf isa Union{AiryPSF,GaussianPSF,CauchyPSF,MoffatPSF}
            # Check that PSF has radial symmetry.
            @test psf(hypot.((x .- x0), (y .- y0)')) ≈ mdl
        end

        # Check conversion of arguments in objective function with given
        # alpha.
        @test objfun(     dat, 1, psf, (0, 0)) == objfun(     dat, 1.0, psf, (0.0, 0.0))
        @test objfun(wgt, dat, 1, psf, (0, 0)) == objfun(wgt, dat, 1.0, psf, (0.0, 0.0))

        # Check conversion of arguments in objective function with
        # automatic alpha.
        @test objfun(     dat, psf, (0, 0)) == objfun(     dat, psf, (0.0, 0.0))
        @test objfun(wgt, dat, psf, (0, 0)) == objfun(wgt, dat, psf, (0.0, 0.0))

        # Check that the nonnegative constraints works.
        @test objfun(     -dat, psf, (0, 0), true) == 0
        @test objfun(wgt, -dat, psf, (0, 0), true) == 0

        # Check the versions of the objective function that behave as if
        # the model is zero everywhere.
        @test objfun(     dat) ≈ objfun(     dat, 0, psf, (0, 0))
        @test objfun(wgt, dat) ≈ objfun(wgt, dat, 0, psf, (0, 0))
        @test objfun(     dat, false) ≈ objfun(     dat, true)
        @test objfun(wgt, dat, false) ≈ objfun(wgt, dat, true)

        for (psf1, pos1) in (
            PointSpreadFunctions.fit(initialpsf, (xinit,yinit), dat;
                                     nonnegative=true),
            PointSpreadFunctions.fit(initialpsf, (xinit,yinit), wgt, dat;
                                     nonnegative=true),
            PointSpreadFunctions.fit(initialpsf, (xinit,yinit), dat, roi;
                                     nonnegative=true),
            PointSpreadFunctions.fit(initialpsf, (xinit,yinit), wgt, dat, roi;
                                     nonnegative=true),
        )
            @test pos1[1] ≈ x0     rtol=0.05 atol=0.0
            @test pos1[2] ≈ y0     rtol=0.05 atol=0.0
            for i in 1:length(psf[:])
                @test psf1[i] ≈ psf[i] rtol=0.1  atol=0.0
            end
        end
    end
end

end # module
