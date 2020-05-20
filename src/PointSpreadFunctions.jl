#
# PointSpreadFunctions.jl --
#
# Modeling and fitting of Point Spread Functions (PSF) for Julia.
#

module PointSpreadFunctions

export
    AiryPSF,
    CauchyPSF,
    GaussianPSF,
    MoffatPSF,
    getfwhm,
    findzeros,
    findzeros!

import Base: getindex
using Base: @propagate_inbounds

using OptimPack.Brent: fzero

using SpecialFunctions

include("models.jl")
include("fitting.jl")
import .Fitting: fit

end # module
