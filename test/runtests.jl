module TestPointSpreadFunctions

using PointSpreadFunctions
using Test

@testset "PointSpreadFunctions.jl" begin
    # Write your own tests here.
    x = 1:20
    y = 1:27
    x0 = 7.2
    y0 = 9.3
    psf = GaussianPSF(5.4)
    mdl = psf((x .- x0), (y .- y0)')
    dat = 3.1*mdl + 0.1*randn(size(mdl))
    psf1, pos1 = PointSpreadFunctions.fit(GaussianPSF(3), (6,11), dat;
                                          nonnegative=true,
                                          maxeval=200)
    @test psf1[1] ≈ psf[1] rtol=0.1  atol=0.0
    @test pos1[1] ≈ x0     rtol=0.05 atol=0.0
    @test pos1[2] ≈ y0     rtol=0.05 atol=0.0

    psf = AiryPSF(5.4, 0.4)
    mdl = psf((x .- x0), (y .- y0)')
    dat = 3.1*mdl + 0.1*randn(size(mdl))
    psf2, pos2 = PointSpreadFunctions.fit(AiryPSF(5, 0.3), (6,11), dat;
                                          nonnegative=true, rho=(0.03,1e-8),
                                          maxeval=200)
    @test psf2[1] ≈ psf[1] rtol=0.1  atol=0.0
    @test psf2[2] ≈ psf[2] rtol=0.1  atol=0.0
    @test pos2[1] ≈ x0     rtol=0.05 atol=0.0
    @test pos2[2] ≈ y0     rtol=0.05 atol=0.0

    psf = CauchyPSF(5.4)
    mdl = psf((x .- x0), (y .- y0)')
    dat = 3.1*mdl + 0.1*randn(size(mdl))
    psf3, pos3 = PointSpreadFunctions.fit(CauchyPSF(4), (6,11), dat;
                                          nonnegative=true,
                                          rho=(0.03,1e-6),
                                          maxeval=200)
    @test psf3[1] ≈ psf[1] rtol=0.1  atol=0.0
    @test pos2[1] ≈ x0     rtol=0.05 atol=0.0
    @test pos2[2] ≈ y0     rtol=0.05 atol=0.0
end


end # module
