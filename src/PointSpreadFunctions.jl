module PointSpreadFunctions

export
    AiryPSF,
    CauchyPSF,
    GaussianPSF,
    MoffatPSF,
    getfwhm,
    findzeros,
    findzeros!

using OptimPack.Brent: fzero

using SpecialFunctions

const J1 = besselj1
const AIRY_FWHM = 1.02899397
const AIRY_FIRST_ZERO  = 1.22
const AIRY_SECOND_ZERO = 2.23
const AIRY_THIRD_ZERO  = 3.24
const AIRY_FOURTH_ZERO = 4.24
const AIRY_FIFTH_ZERO  = 5.24


"""

Instances of sub-types of `AbstractPSF` are used to store the parameters of
various Point Spread Functions (PSF).  Let `P` be such an instance, then:

    P([T,] r)
    P([T,] x, y)

yield an approximation of the PSF at a distance `r` or at a position `(x,y)`
relative to the position of the point source.

The PSF is normalized such that its peak value (usually at the center) is equal
to one.  The rationale is that normalization by the peak value does not depend
on the dimensionality while the integral of the PSF does depend on the
dimensionality.

See also [`AiryPSF`](@ref), [`CauchyPSF`](@ref), [`GaussianPSF`](@ref),
[`MoffatPSF`](@ref).

"""
abstract type AbstractPSF end

"""
    getfwhm(P)

yields the full width at half maximum (FWHM) of point spread function `P`.
Computing the result may require some calculations, so it is better to save
the result if this value must be used several times.

"""
function getfwhm end

"""
    P = AiryPSF(lim [, eps=0])

defines the point spread function `P` of a circular pupil (AiryPSF PSF).
Argument `lim` is the distance in the focal plane corresponding to the
diffraction limit.  Assuming the distance is given in angular units
(radians), `lim = λ/D` with `λ` the wavelength and `D` the pupil diameter.
Optional argument `eps` is the ratio of the size of the central obscuration
to that of the pupil (necessarily `0 ≤ eps < 1`).  If not specified, `eps =
0` is assumed (unobstructed pupil).  Then:

    P([T,] r)
    P([T,] x, y)

yield the PSF at distance `r` or position `(x,y)` relative to the position of
the source and normalized so that the peak value of the PSF is one.  Optional
argument `T` is the floating-point type of the result.  Arguments `lim`, `r`,
`x` and `y` are in the same units.

See also [`CauchyPSF`](@ref), [`GaussianPSF`](@ref), [`MoffatPSF`](@ref).

"""
struct AiryPSF <: AbstractPSF
    lim::Float64
    eps::Float64
    _a::Float64
    _c::Float64
    _o::Bool
    function AiryPSF(lim::Real, eps::Real = 0.0)
        (isfinite(lim) && lim > 0.0) ||
            throw_bad_argument("bad diffraction limit value")
        (isfinite(eps) && 0.0 ≤ eps < 1.0) ||
            throw_bad_argument("out of range central obscuration ratio")
        obstrucated = (eps > 0.0)
        a = obstrucated ? 2.0/((1.0 - eps)*(1.0 + eps)) : 2.0
        return new(lim, eps, a, pi/lim, obstrucated)
    end
end

@noinline throw_bad_argument(args...) = throw_bad_argument(string(args...))
@noinline throw_bad_argument(mesg::String) = throw(ArgumentError(mesg))

function getfwhm(P::AiryPSF; kwds...)
    l, e, a, c = P.lim, P.eps, P._a, P._c
    v = 1/sqrt(2)
    f(r) = (e > 0.0 ?
            v - _annular_amplitude(a, c, e, r) :
            v - _circular_amplitude(a, c, r))
    return fzero(f, 0.52*l, 0.35*l; kwds...)[1]*2
end

@inline _circular_amplitude(a::T, c::T, r::T) where {T<:AbstractFloat} =
    (r == zero(T) ? one(T) : (u = c*r; (a/u)*J1(u)))

@inline _annular_amplitude(a::T, c::T, e::T, r::T) where {T<:AbstractFloat} =
    (r == zero(T) ? one(T) : (u = c*r; (a/u)*(J1(u) - e*J1(e*u))))

@inline _airy1(a::T, c::T, r::T) where {T<:AbstractFloat} =
    (p = _circular_amplitude(a, c, r); p*p)

@inline _airy1(a::T, c::T, x::T, y::T) where {T<:AbstractFloat} =
    _airy1(a, c, hypot(x, y))

@inline _airy2(a::T, c::T, e::T, r::T) where {T<:AbstractFloat} =
    (p = _annular_amplitude(a, c, e, r); p*p)

@inline _airy2(a::T, c::T, e::T, x::T, y::T) where {T<:AbstractFloat} =
    _airy2(a, c, e, hypot(x, y))

(P::AiryPSF)(::Type{Float64}, r::Float64) =
    (P._o ?
     _airy2(P._a, P._c, P.eps, r) :
     _airy1(P._a, P._c,        r))

(P::AiryPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    (P._o ?
     _airy2(P._a, P._c, P.eps, T(r)) :
     _airy1(P._a, P._c,        T(r)))

(P::AiryPSF)(::Type{Float64}, x::Float64, y::Float64) =
    (P._o ?
     _airy2(P._a, P._c, P.eps, x, y) :
     _airy1(P._a, P._c,        x, y))

(P::AiryPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    (P._o ? _airy2(P._a, P._c, P.eps, T(x), T(y))
     : _airy1(P._a, P._c,        T(x), T(y)))

function (P::AiryPSF)(::Type{T}, r::AbstractArray{T,N}) where {T<:AbstractFloat, N}
    a, c, e = T(P._a), T(P._c), T(P.eps)
    return (P._o ?
            map((r) -> _airy2(a, c, e, r), r) :
            map((r) -> _airy1(a, c,    r), r))
end

function (P::AiryPSF)(::Type{T}, r::AbstractArray) where {T<:AbstractFloat}
    a, c, e = T(P._a), T(P._c), T(P.eps)
    return (P._o ?
            map((r) -> _airy2(a, c, e, T(r)), r) :
            map((r) -> _airy1(a, c,    T(r)), r))
end

function (P::AiryPSF)(::Type{T},
                   x::AbstractArray{T,Nx},
                   y::AbstractArray{T,Ny}) where {T<:AbstractFloat,Nx,Ny}
    a, c, e = T(P._a), T(P._c), T(P.eps)
    return (P._o ?
            broadcast((x, y) -> _airy2(a, c, e, x, y), x, y) :
            broadcast((x, y) -> _airy1(a, c,    x, y), x, y))
end

function (P::AiryPSF)(::Type{T},
                   x::AbstractArray{Tx,Nx},
                   y::AbstractArray{Ty,Ny}) where {T<:AbstractFloat,
                                                   Tx<:Real,Nx,Ty<:Real,Ny}

    a, c, e = T(P._a), T(P._c), T(P.eps)
    return (P._o ?
            broadcast((x, y) -> _airy2(a, c, e, T(x), T(y)), x, y) :
            broadcast((x, y) -> _airy1(a, c,    T(x), T(y)), x, y))
end

"""
    findzeros([T=Float64,] P, n; kwds...)

yield the `n`-th zero of PSF `P`.  If `n` is a vector of integers, a vector
with the corresponding zeros is returned.  Optional argument `T` can be
used to specify the floating-point type of the result.  The keywords of
`OptimPack.Brent.fzero` can be specified.

See also: [`OptimPack.Brent.fzero`](@ref]

"""
findzeros(P::AiryPSF, args...; kwds...) =
    findzeros(Float64, P::AiryPSF, args...; kwds...)

function findzeros(::Type{T},
                   P::AiryPSF,
                   n::AbstractVector{I};
                   kwds...) where {T<:AbstractFloat,I<:Integer}
    result = Array{T}(length(n))
    for i in eachindex(result, n)
        result[i] = findzeros(T, P, n[i])
    end
    return result
end

# FIXME: the initial bracket has to be improved for obstruction ratio around
# 0.5, in fact a method like BRADI should be used
function findzeros(::Type{T}, P::AiryPSF, n::Integer;
                   kwds...) where {T<:AbstractFloat}
    @assert n ≥ 1
    l, e, a, c = T(P.lim), T(P.eps), T(P._a), T(P._c)
    g = (n + T(0.25) - T(0.49)*e)*l  # guess for the n-th zero
    h = T(0.7)*l                     # half-width of search interval
    P._o ?
        fzero(r -> _annular_amplitude(a, c, e, r), g - h, g + h; kwds...)[1] :
        fzero(r -> _circular_amplitude(a, c, r),   g - h, g + h; kwds...)[1]
end

"""
    P = CauchyPSF(fwhm)
    P([T,] r)
    P([T,] x, y)

defines a Cauchy PSF `P` (or Lorentzian) of full width at half maximum
`fwhm` which can be used to compute the PSF at distance `r` or position
`(x,y)` relative to the position of the source and normalized so that the
peak value of the PSF is one.  Optional argument `T` is the floating-point
type of the result.  Arguments `fwhm`, `r`, `x` and `y` are in the same
units.

See also [`AiryPSF`](@ref), [`GaussianPSF`](@ref), [`MoffatPSF`](@ref).

"""
struct CauchyPSF <: AbstractPSF
    fwhm::Float64
    _q::Float64
    function CauchyPSF(fwhm::Real)
        (isfinite(fwhm) && fwhm > 0.0) ||
            throw_bad_argument("bad FWHM value")
        return new(fwhm, (fwhm/2)^2)
    end
end

getfwhm(P::CauchyPSF) = P.fwhm

@inline _cauchy(q::T, r::T) where {T<:AbstractFloat} = q/(r*r + q)

@inline _cauchy(q::T, x::T, y::T) where {T<:AbstractFloat} = q/(x*x + y*y + q)

(P::CauchyPSF)(::Type{Float64}, r::Float64) =
    _cauchy(P._q, r)

(P::CauchyPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    _cauchy(T(P._q), T(r))

(P::CauchyPSF)(::Type{Float64}, x::Float64, y::Float64) =
    _cauchy(P._q, x, y)

(P::CauchyPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    _cauchy(T(P._q), T(x), T(y))

function (P::CauchyPSF)(::Type{T}, r::AbstractArray{T,N}) where {T<:AbstractFloat, N}
    q = T(P._q)
    map((r) -> _cauchy(q, r), r)
end

function (P::CauchyPSF)(::Type{T}, r::AbstractArray) where {T<:AbstractFloat}
    q = T(P._q)
    map((r) -> _cauchy(q, T(r)), r)
end

function (P::CauchyPSF)(::Type{T},
                        x::AbstractArray{T,Nx},
                        y::AbstractArray{T,Ny}) where {T<:AbstractFloat,Nx,Ny}
    q = T(P._q)
    broadcast((x, y) -> _cauchy(q, x, y), x, y)
end

function (P::CauchyPSF)(::Type{T}, x::AbstractArray{Tx,Nx},
                        y::AbstractArray{Ty,Ny}) where {T<:AbstractFloat,
                                                        Tx<:Real,Nx,Ty<:Real,Ny}

    q = T(P._q)
    broadcast((x, y) -> _cauchy(q, T(x), T(y)), x, y)
end


"""
    P = GaussianPSF(fwhm)
    P([T,] r)
    P([T,] x, y)

defines a Gaussian PSF `P` of full width at half maximum `fwhm` which can be
used to compute the PSF at distance `r` or position `(x,y)` relative to the
position of the source and normalized so that the peak value of the PSF is one.
Optional argument `T` is the floating-point type of the result.  Arguments
`fwhm`, `r`, `x` and `y` are in the same units.

See also [`AiryPSF`](@ref), [`CauchyPSF`](@ref), [`MoffatPSF`](@ref).

"""
struct GaussianPSF <: AbstractPSF
    fwhm::Float64
    _q::Float64
    function GaussianPSF(fwhm::Real)
        (isfinite(fwhm) && fwhm > 0.0) ||
            throw_bad_argument("bad FWHM value")
        return new(fwhm, -log(2.0)*(2.0/fwhm)^2)
    end
end

getfwhm(P::GaussianPSF) = P.fwhm

@inline _gauss(q::T, r::T) where {T<:AbstractFloat} = exp(r*r*q)

@inline _gauss(q::T, x::T, y::T) where {T<:AbstractFloat} = exp((x*x + y*y)*q)

(P::GaussianPSF)(::Type{Float64}, r::Float64) =
    _gauss(P._q, r)

(P::GaussianPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    _gauss(T(P._q), T(r))

(P::GaussianPSF)(::Type{Float64}, x::Float64, y::Float64) =
    _gauss(P._q, x, y)

(P::GaussianPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    _gauss(T(P._q), T(x), T(y))

function (P::GaussianPSF)(::Type{T},
                          r::AbstractArray{T,N}) where {T<:AbstractFloat, N}
    q = T(P._q)
    map((r) -> _gauss(q, r), r)
end

function (P::GaussianPSF)(::Type{T}, r::AbstractArray) where {T<:AbstractFloat}
    q = T(P._q)
    map((r) -> _gauss(q, T(r)), r)
end

function (P::GaussianPSF)(::Type{T},
                          x::AbstractArray{T,N},
                          y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    q = T(P._q)
    if size(x) == size(y)
        return broadcast((x, y) -> _gauss(q, x, y), x, y)
    else
        return P(T,x).*P(T,y)
    end
end

function (P::GaussianPSF)(::Type{T}, x::AbstractArray{Tx,Nx},
                          y::AbstractArray{Ty,Ny}) where {T<:AbstractFloat,
                                                          Tx<:Real,Nx,Ty<:Real,Ny}
    q = T(P._q)
    if size(x) == size(y)
        return broadcast((x, y) -> _gauss(q, T(x), T(y)), x, y)
    else
        return P(T,x).*P(T,y)
    end
end


"""
    P = MoffatPSF(fwhm, beta)
    P([T,] r)
    P([T,] x, y)

defines a Moffat PSF `P` of full width at half maximum `fwhm` and exponent
`beta` which can be used to compute the PSF at distance `r` or position `(x,y)`
relative to the position of the source and normalized so that the peak value of
the PSF is one.  Optional argument `T` is the floating-point type of the
result.  Arguments `fwhm`, `r`, `x` and `y` are in the same units.

See also [`AiryPSF`](@ref), [`CauchyPSF`](@ref), [`GaussianPSF`](@ref).

"""
struct MoffatPSF <: AbstractPSF
    fwhm::Float64
    beta::Float64
    _p::Float64
    _q::Float64
    function MoffatPSF(fwhm::Real, beta::Real)
        (isfinite(fwhm) && fwhm > 0.0) ||
            throw_bad_argument("bad FWHM value")
        (isfinite(beta) && beta > 0.0) ||
            throw_bad_argument("bad value for exponent `beta`")
        return new(fwhm, beta, -beta, (2^(1/beta) - 1)*(2/fwhm)^2)
    end
end

getfwhm(P::MoffatPSF) = P.fwhm

@inline _moffat(p::T, q::T, r::T) where {T<:AbstractFloat} =
    (r*r*q + one(T))^p

@inline _moffat(p::T, q::T, x::T, y::T) where {T<:AbstractFloat} =
    ((x*x + y*y)*q + one(T))^p

(P::MoffatPSF)(::Type{Float64}, r::Float64) =
    _moffat(P._q, r)

(P::MoffatPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    _moffat(T(P._q), T(r))

(P::MoffatPSF)(::Type{Float64}, x::Float64, y::Float64) =
    _moffat(P._q, x, y)

(P::MoffatPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    _moffat(T(P._q), T(x), T(y))

function (P::MoffatPSF)(::Type{T},
                        r::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    q = T(P._q)
    map((r) -> _moffat(q, r), r)
end

function (P::MoffatPSF)(::Type{T}, r::AbstractArray) where {T<:AbstractFloat}
    q = T(P._q)
    map((r) -> _moffat(q, T(r)), r)
end

function (P::MoffatPSF)(::Type{T},
                        x::AbstractArray{T,Nx},
                        y::AbstractArray{T,Ny}) where{T<:AbstractFloat,Nx,Ny}
    q = T(P._q)
    broadcast((x, y) -> _moffat(q, x, y), x, y)
end

function (P::MoffatPSF)(::Type{T},
                        x::AbstractArray{Tx,Nx},
                        y::AbstractArray{Ty,Ny}) where {T<:AbstractFloat,
                                                        Tx<:Real,Nx,Ty<:Real,Ny}

    q = T(P._q)
    broadcast((x, y) -> _moffat(q, T(x), T(y)), x, y)
end

# Define methods common to all sub-types of AbstractPSF.
for T in (:AiryPSF, :CauchyPSF, :GaussianPSF, :MoffatPSF)
    @eval begin
        (P::$T)(r::Real) = P(Float64, r)
        (P::$T)(r::AbstractArray) = P(Float64, r)
        (P::$T)(x::Real, y::Real) = P(Float64, x, y)
        (P::$T)(x::AbstractArray, y::AbstractArray) = P(Float64, x, y)
    end
end

end # module
