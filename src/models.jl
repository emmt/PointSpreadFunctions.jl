#
# models.jl --
#
# Models of Point Spread Functions (PSF) for Julia.
#

const J1 = besselj1
const AIRY_FWHM = 1.02899397
const AIRY_FIRST_ZERO  = 1.22
const AIRY_SECOND_ZERO = 2.23
const AIRY_THIRD_ZERO  = 3.24
const AIRY_FOURTH_ZERO = 4.24
const AIRY_FIFTH_ZERO  = 5.24

"""

Instances of sub-types of `PointSpreadFunctions.AbstractPSF{N}` are used to
store the parameters of various *Point Spread Functions* (PSF) models, `N`
is the number of parameters.  Let `P` be such an instance, then:

    P([T,] r)
    P([T,] x, y)

yield an approximation of the PSF at a distance `r` or at a position
`(x,y)` relative to the position of the point source.

Use index notation to retrieve the parameters as an `N`-tuple or a specific
parameters:

    P[:]  # yields all parameters
    P[i]  # yields i-th parameter

The PSF is normalized such that its peak value (usually at the center) is
equal to one.  The rationale is that normalization by the peak value does
not depend on the dimensionality while the integral of the PSF does depend
on the dimensionality.

See also [`AiryPSF`](@ref), [`CauchyPSF`](@ref), [`GaussianPSF`](@ref),
[`MoffatPSF`](@ref).

""" AbstractPSF

abstract type AbstractPSF{N} end

@inline getindex(P::AbstractPSF, ::Colon) = parameters(P)
@inline @propagate_inbounds getindex(P::AbstractPSF, i::Integer) =
    getindex(parameters(P), i)

"""
    getfwhm(P)

yields the full width at half maximum (FWHM) of point spread function `P`.
Computing the result may require some calculations, so it is better to save
the result if this value must be used several times.

""" getfwhm

"""
    check_structure(psf)

throws an exception if the parameters of the point spread function `psf`
are invalid.  Call [`isvalid(psf)`](@ref) to check the validity of the
parameters without throwing exceptions.

"""
check_structure(psf::AbstractPSF) =
    isvalid(psf) || throw_bad_argument("PSF has invalid parameter(s)")

"""
    isvalid(psf) -> bool

yields whether the parameters of the point spread function `psf` are
correct.  The [`check_structure`](@ref) method is similar but throws for
invalid parameters.

"""
isvalid(::AbstractPSF) = true

"""
    P = AiryPSF(lim [, ρ=0])

defines the *Point Spread Function* (PSF) `P` of a circular pupil.
Argument `lim` is the distance in the focal plane corresponding to the
diffraction limit.  Assuming the distance is given in angular units
(radians), `lim = λ/D` with `λ` the wavelength and `D` the pupil diameter
(both expressed in the same units).  Optional argument `ρ` is the ratio of
the size of the central obscuration to that of the pupil (necessarily `0 ≤
ρ < 1`).  If not specified, `ρ = 0` is assumed (unobstructed pupil).  Then:

    P([T,] r)
    P([T,] x, y)

yield the PSF at distance `r` or position `(x,y)` relative to the position of
the source and normalized so that the peak value of the PSF is one.  Optional
argument `T` is the floating-point type of the result.  Arguments `lim`, `r`,
`x` and `y` are in the same units.

The peak-normalized Airy PSF at distance `r` is given by:

    P(r) = ((a/(c*r))*(J1(c*r) - ρ*J1(ρ*c*r)))^2

with `a = 2/(1 - ρ^2)`, `c = π/lim` and `J1` the Bessel function of the
first kind.

See also [`CauchyPSF`](@ref), [`GaussianPSF`](@ref), [`MoffatPSF`](@ref).

""" AiryPSF

struct AiryPSF <: AbstractPSF{2}
    prm::NTuple{2,Float64}
    a::Float64
    c::Float64
    AiryPSF(lim::Real, rho::Real = 0.0) =
        new((lim, rho),
            (rho == 0 ? 2.0 : 2.0/((1.0 - rho)*(1.0 + rho))),
            π/lim)
end

@inline isvalid(P::AiryPSF) = _check_airy_lim(P) && _check_airy_rho(P)

function check_structure(P::AiryPSF)
    _check_airy_lim(P) ||
        throw_bad_argument("bad diffraction limit value")
    _check_airy_rho(P) ||
        throw_bad_argument("out of range central obscuration ratio")
end

@inline _check_airy_lim(P::AiryPSF) = finite_and_positive(_get_lim(P))
@inline _check_airy_rho(P::AiryPSF) = (ρ = _get_rho(P);
                                       isfinite(ρ) && 0.0 ≤ ρ < 1.0)

Base.show(io::IO, P::AiryPSF) =
    print(io, "AiryPSF(", P[1], ",", P[2], ")")

@inline parameters(P::AiryPSF) = getfield(P, :prm)
@inline obstrucated(P::AiryPSF) =  _get_rho(P) == 0

@inline _get_lim(P::AiryPSF) = @inbounds P[1]
@inline _get_rho(P::AiryPSF) = @inbounds P[2]
@inline _get_a(P::AiryPSF) = getfield(P, :a)
@inline _get_c(P::AiryPSF) = getfield(P, :c)

function getfwhm(P::AiryPSF; kwds...)
    l, ρ, a, c =  _get_lim(P), _get_rho(P), _get_a(P), _get_c(P)
    v = 1/sqrt(2)
    f(r) = v - (ρ == 0 ? _circular_amplitude(a, c, r) :
                _annular_amplitude(a, c, ρ, r))
    return fzero(f, 0.52*l, 0.35*l; kwds...)[1]*2
end

@inline _circular_amplitude(a::T, c::T, r::T) where {T<:AbstractFloat} =
    r == zero(T) ? one(T) : (u = c*r; (a/u)*J1(u))

@inline _annular_amplitude(a::T, c::T, ρ::T, r::T) where {T<:AbstractFloat} =
    r == zero(T) ? one(T) : (u = c*r; (ρ == zero(T) ?
                                       (a/u)*J1(u) :
                                       (a/u)*(J1(u) - ρ*J1(ρ*u))))

# No central obscuration.
@inline _airy(a::T, c::T, r::T) where {T<:AbstractFloat} =
    r == zero(T) ? one(T) : (u = c*r; ((a/u)*J1(u))^2)

# With central obscuration.
@inline _airy(a::T, c::T, ρ::T, r::T) where {T<:AbstractFloat} =
    r == zero(T) ? one(T) : (u = c*r; (ρ == zero(T) ?
                                       (a/u)*J1(u) :
                                       (a/u)*(J1(u) - ρ*J1(ρ*u)))^2)

(P::AiryPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    _airy(to_float(T, _get_a(P)),
          to_float(T, _get_c(P)),
          to_float(T, _get_rho(P)),
          to_float(T, r))

(P::AiryPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    P(T, hypot(to_float(T, x), to_float(T, y)))

function (P::AiryPSF)(::Type{T},
                      r::AbstractArray{<:Real}) where {T<:AbstractFloat}
    a = to_float(T, _get_a(P))
    c = to_float(T, _get_c(P))
    ρ = to_float(T, _get_rho(P))
    ρ == zero(ρ) ?
        map(r -> _airy(a, c,    to_float(T, r)), r) :
        map(r -> _airy(a, c, ρ, to_float(T, r)), r)
end

function (P::AiryPSF)(::Type{T},
                      x::AbstractArray{<:Real},
                      y::AbstractArray{<:Real}) where {T<:AbstractFloat}
    a = to_float(T, _get_a(P))
    c = to_float(T, _get_c(P))
    ρ = to_float(T, _get_rho(P))
    ρ == zero(ρ) ?
        broadcast((x, y) -> _airy(a, c,    hypot(to_float(T, x), to_float(T, y))),
                  x, y) :
        broadcast((x, y) -> _airy(a, c, ρ, hypot(to_float(T, x), to_float(T, y))),
                  x, y)
end

"""
    findzeros([T=Float64,] P, n; kwds...)

yield the `n`-th zero of *Point Spread Function* (PSF) `P`.  If `n` is a
vector of integers, a vector with the corresponding zeros is returned.
For example:

    findzeros(P, 1:3)

yields the first 3 zeros of `P`.

Optional argument `T` can be used to specify the floating-point type of the
result.  The keywords of `OptimPack.fzero` can be specified.

"""
findzeros(P::AiryPSF, args...; kwds...) =
    findzeros(Float64, P::AiryPSF, args...; kwds...)

function findzeros(::Type{T},
                   P::AiryPSF,
                   n::AbstractVector{I};
                   kwds...) where {T<:AbstractFloat,I<:Integer}
    result = Array{T}(undef, length(n))
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
    l, ρ, a, c = T(_get_lim(P)), T(_get_rho(P)), T(_get_a(P)), T(_get_c(P))
    g = (n + T(0.25) - T(0.49)*ρ)*l  # guess for the n-th zero
    h = T(0.7)*l                     # half-width of search interval
    obstrucated(P) ?
        fzero(r -> _annular_amplitude(a, c, ρ, r), g - h, g + h; kwds...)[1] :
        fzero(r -> _circular_amplitude(a, c, r),   g - h, g + h; kwds...)[1]
end

"""
    P = CauchyPSF(fwhm)

defines a Cauchy *Point Spread Function* (PSF) `P` (or Lorentzian) of full
width at half maximum `fwhm` which can be used as:

    P([T,] r)
    P([T,] x, y)

to compute the PSF at distance `r` or position `(x,y)` relative to the
position of the source and normalized so that the peak value of the PSF is
one.  Optional argument `T` is the floating-point type of the result.
Arguments `fwhm`, `r`, `x` and `y` are in the same units.

The peak-normalized Cauchy PSF at distance `r` is given by:

    P(r) = q/(q + r^2)

with `q = (fwhm/2)^2`.

See also [`AiryPSF`](@ref), [`GaussianPSF`](@ref), [`MoffatPSF`](@ref).

""" CauchyPSF

struct CauchyPSF <: AbstractPSF{1}
    fwhm::Float64
    q::Float64
    CauchyPSF(fwhm::Real) = new(fwhm, (fwhm/2)^2)
end

@inline isvalid(P::CauchyPSF) = finite_and_positive(getfwhm(P))

check_structure(P::CauchyPSF) =
    finite_and_positive(getfwhm(P)) || throw_bad_argument("bad FWHM value")

Base.show(io::IO, P::CauchyPSF) =
    print(io, "CauchyPSF(", P[1], ")")

@inline getfwhm(P::CauchyPSF) = getfield(P, :fwhm)
@inline parameters(P::CauchyPSF) = (getfwhm(P),)

@inline _get_q(P::CauchyPSF) = getfield(P, :q)

@inline _cauchy(q::T, r::T) where {T<:AbstractFloat} = q/(r*r + q)

@inline _cauchy(q::T, x::T, y::T) where {T<:AbstractFloat} = q/(x*x + y*y + q)

(P::CauchyPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    _cauchy(to_float(T, _get_q(P)),
            to_float(T, r))

(P::CauchyPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    _cauchy(to_float(T, _get_q(P)),
            to_float(T, x),
            to_float(T, y))

function (P::CauchyPSF)(::Type{T},
                        r::AbstractArray{<:Real}) where {T<:AbstractFloat}
    q = to_float(T, _get_q(P))
    map(r -> _cauchy(q, to_float(T, r)), r)
end

function (P::CauchyPSF)(::Type{T},
                        x::AbstractArray{<:Real},
                        y::AbstractArray{<:Real}) where {T<:AbstractFloat}

    q = to_float(T, _get_q(P))
    broadcast((x, y) -> _cauchy(q, to_float(T, x), to_float(T, y)), x, y)
end


"""
    P = GaussianPSF(fwhm)

defines a Gaussian *Point Spread Function* (PSF) `P` of full width at half
maximum `fwhm` which can be used as:

    P([T,] r)
    P([T,] x, y)

to compute the PSF at distance `r` or position `(x,y)` relative to the
position of the source and normalized so that the peak value of the PSF is
one.  Optional argument `T` is the floating-point type of the result.
Arguments `fwhm`, `r`, `x` and `y` are in the same units.

The peak-normalized Gaussian PSF at distance `r` is given by:

    P(r) = exp(q*r^2)

with `q = -4*log(2)/fwhm^2`.

See also [`AiryPSF`](@ref), [`CauchyPSF`](@ref), [`MoffatPSF`](@ref).

""" GaussianPSF

struct GaussianPSF <: AbstractPSF{1}
    fwhm::Float64
    q::Float64
    GaussianPSF(fwhm::Real) = new(fwhm, -log(2.0)*(2.0/fwhm)^2)
end

@inline isvalid(P::GaussianPSF) = finite_and_positive(getfwhm(P))

check_structure(P::GaussianPSF) =
    finite_and_positive(getfwhm(P)) || throw_bad_argument("bad FWHM value")

Base.show(io::IO, P::GaussianPSF) =
    print(io, "GaussianPSF(", P[1], ")")

@inline getfwhm(P::GaussianPSF) = getfield(P, :fwhm)
@inline parameters(P::GaussianPSF) = (getfwhm(P),)

@inline _get_q(P::GaussianPSF) = getfield(P, :q)

@inline _gauss(q::T, r::T) where {T<:AbstractFloat} = exp(r*r*q)

@inline _gauss(q::T, x::T, y::T) where {T<:AbstractFloat} = exp((x*x + y*y)*q)

(P::GaussianPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    _gauss(to_float(T, _get_q(P)),
           to_float(T, r))

(P::GaussianPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    _gauss(to_float(T, _get_q(P)),
           to_float(T, x),
           to_float(T, y))

function (P::GaussianPSF)(::Type{T},
                        r::AbstractArray{<:Real}) where {T<:AbstractFloat}
    q = to_float(T, _get_q(P))
    map(r -> _gauss(q, to_float(T, r)), r)
end

function (P::GaussianPSF)(::Type{T},
                        x::AbstractArray{<:Real},
                        y::AbstractArray{<:Real}) where {T<:AbstractFloat}

    if axes(x) == axes(y)
        q = to_float(T, _get_q(P))
        broadcast((x, y) -> _gauss(q, to_float(T, x), to_float(T, y)), x, y)
    else
        P(T,x).*P(T,y)
    end
end


"""
    P = MoffatPSF(fwhm, beta)

defines a Moffat PSF `P` of full width at half maximum `fwhm` and exponent
`beta` which can be used as:

    P([T,] r)
    P([T,] x, y)

to compute the PSF at distance `r` or position `(x,y)` relative to the
position of the source and normalized so that the peak value of the PSF is
one.  Optional argument `T` is the floating-point type of the result.
Arguments `fwhm`, `r`, `x` and `y` are in the same units.

The peak-normalized Moffat PSF at distance `r` is given by:

    P(r) = (1 + q*r^2)^(-beta)

with `q = (2^(1/beta) - 1)*(2/fwhm)^2`.

See also [`AiryPSF`](@ref), [`CauchyPSF`](@ref), [`GaussianPSF`](@ref).

""" MoffatPSF

struct MoffatPSF <: AbstractPSF{2}
    prm::NTuple{2,Float64} # (fwhm, beta)
    p::Float64 # (2^(1/beta) - 1)*(2/fwhm)^2
    q::Float64 # -beta
    MoffatPSF(fwhm::Real, beta::Real) =
        new((fwhm, beta), -beta, (2.0^(1.0/beta) - 1.0)*(2.0/fwhm)^2)
end

@inline isvalid(P::MoffatPSF) =
    finite_and_positive(_get_fwhm(P)) && finite_and_positive(_get_beta(P))

function check_structure(P::MoffatPSF)
    finite_and_positive(_get_fwhm(P)) ||
        throw_bad_argument("bad FWHM value")
    finite_and_positive(_get_beta(P)) ||
        throw_bad_argument("bad value for exponent `beta`")
end

Base.show(io::IO, P::MoffatPSF) =
    print(io, "MoffatPSF(", P[1], ",", P[2], ")")

@inline getfwhm(P::MoffatPSF) = _get_fwhm(P)
@inline parameters(P::MoffatPSF) =  getfield(P, :prm)

@inline _get_fwhm(P::MoffatPSF) = @inbounds P[1]
@inline _get_beta(P::MoffatPSF) = @inbounds P[2]
@inline _get_p(P::MoffatPSF) = getfield(P, :p)
@inline _get_q(P::MoffatPSF) = getfield(P, :q)

@inline _moffat(p::T, q::T, r::T) where {T<:AbstractFloat} =
    (r*r*q + one(T))^p

@inline _moffat(p::T, q::T, x::T, y::T) where {T<:AbstractFloat} =
    ((x*x + y*y)*q + one(T))^p

(P::MoffatPSF)(::Type{T}, r::Real) where {T<:AbstractFloat} =
    _moffat(to_float(T, _get_p(P)),
            to_float(T, _get_q(P)),
            to_float(T, r))

(P::MoffatPSF)(::Type{T}, x::Real, y::Real) where {T<:AbstractFloat} =
    _moffat(to_float(T, _get_p(P)),
            to_float(T, _get_q(P)),
            to_float(T, x),
            to_float(T, y))

function (P::MoffatPSF)(::Type{T},
                        r::AbstractArray{<:Real}) where {T<:AbstractFloat}
    p = to_float(T, _get_p(P))
    q = to_float(T, _get_q(P))
    map(r -> _moffat(p, q, to_float(T, r)), r)
end

function (P::MoffatPSF)(::Type{T},
                        x::AbstractArray{<:Real},
                        y::AbstractArray{<:Real}) where {T<:AbstractFloat}

    p = to_float(T, _get_p(P))
    q = to_float(T, _get_q(P))
    broadcast((x, y) -> _moffat(p, q, to_float(T, x), to_float(T, y)), x, y)
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


@noinline throw_bad_argument(args...) = throw_bad_argument(string(args...))
@noinline throw_bad_argument(mesg::String) = throw(ArgumentError(mesg))

@inline finite_and_positive(x::Real) = (isfinite(x) && x > 0)
@inline finite_and_nonnegative(x::Real) = (isfinite(x) && x ≥ 0)

"""
    to_float(T, x)

lazily yields `x` converted to floating-point type `T`.

"""
to_float(::Type{T}, x::T   ) where {T<:AbstractFloat} = x
to_float(::Type{T}, x::Real) where {T<:AbstractFloat} = T(x)
