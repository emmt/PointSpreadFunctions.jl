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
    findzeros

import Base: getindex
using Base: @propagate_inbounds

using OptimPack.Brent: fzero

using SpecialFunctions

function fit end

include("models.jl")
include("fitting.jl")

end # module
