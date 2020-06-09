#
# fitting.jl --
#
# Fitting Point Spread Functions (PSF) models for Julia.
#

module Fitting

import OptimPack.Powell.Newuoa
using ..PointSpreadFunctions
using ..PointSpreadFunctions:
    AbstractPSF,
    throw_bad_argument,
    to_float,
    to_int
import ..PointSpreadFunctions: fit

# An N-dimensional Region Of Interest (ROI).
const ROI{N} = NTuple{N,Union{Colon,AbstractUnitRange{<:Integer}}}

# Union of possible model functions (may be needed in the future to avoid
# ambiguities).
const Model = Any # Union{AbstractPSF,Function}

"""
    fit(psf, pos, [wgt,] dat[, roi]; nonnegative=false) -> psf′, pos′

fits a given *Point Spread Function* (PSF) model on data `dat` with
(optional) respective weights `wgt`.  Argument `psf` is a PSF instance to
specify which PSF model to use and its initial parameters.  Argument `pos`
is the initial PSF position.  The result is a 2-tuple with the fitted PSF
model and position.

The fit is carried out by `OptimPack.Powell.Newuoa.minimize` method.  The
initial parameters should be close enough to the solution for the fit to
behave correctly.

Keyword `nonnegative` indicates whether the intensity of the PSF should be
nonnegative or not.  Keyword `rho = (rhobeg,rhoend)` may be used to specify
the initial and final precision on the parameters.  Additional keywords may
be specified and are passed to the minimizer.

An optional *Region Of Interest* (ROI) may be specified by argument `roi`
as an `N`-tuple of index ranges or colons.  This is useful to limit the
region used to perform the fit.  For example:

    fit(GaussianPSF(2.5), (23,12), dat, (10:30, :))

will fit a Gaussian PSF model on sub-array `dat[10:30,:]`.  The advantages
of specifying a ROI is that the relative position of the ROI is taken into
account in the initial and final position and that a view is used instead
of extracting a sub-array.  Specifying a ROI alos works for weighted data.
Non-rectangular ROIs can be emulated by having weights equal to zero where
data should be ignored.

"""
function fit(psf::PSF,
             pos::NTuple{2,Real},
             dat::AbstractArray{T,2};
             rho::Tuple{Real,Real} = (0.1, 1e-5),
             nonnegative::Bool = false,
             maxeval::Integer = 50*(N + 2),
             kwds...) where {T<:AbstractFloat,
                             N,PSF<:AbstractPSF{N}}
    # Newuoa uses double precision floating-point.
    x = Cdouble[pos..., psf[:]...]
    ans = Newuoa.minimize!(x -> objfun(dat, PSF, x, nonnegative),
                           x, rho...; maxeval=Int(maxeval), kwds...)
    return PSF(x[3:N+2]...), (x[1], x[2])
end

function fit(psf::PSF,
             pos::NTuple{2,Real},
             wgt::AbstractArray{T,2},
             dat::AbstractArray{T,2};
             rho::Tuple{Real,Real} = (0.1, 1e-5),
             nonnegative::Bool = false,
             maxeval::Integer = 50*(N + 2),
             kwds...) where {T<:AbstractFloat,
                             N,PSF<:AbstractPSF{N}}
    @assert axes(wgt) == axes(dat)
    # Newuoa uses double precision floating-point.
    x = Cdouble[pos..., psf[:]...]
    ans = Newuoa.minimize!(x -> objfun(wgt, dat, PSF, x, nonnegative),
                           x, rho...; maxeval=Int(maxeval), kwds...)
    return PSF(x[3:N+2]...), (x[1], x[2])
end

function fit(psf::AbstractPSF,
             pos::NTuple{N,Real},
             dat::AbstractArray{T,N},
             roi::ROI{N};
             kwds...) where {T<:AbstractFloat,N}
    off = offsets(dat, roi)
    psf1, pos1 = fit(psf, pos .- off,
                     view(dat, roi...); kwds...)
    return psf1, (pos1 .+ off)
end

function fit(psf::AbstractPSF,
             pos::NTuple{N,Real},
             wgt::AbstractArray{T,N},
             dat::AbstractArray{T,N},
             roi::ROI{N};
             kwds...) where {T<:AbstractFloat,N}
    off = offsets(dat, roi)
    psf1, pos1 = fit(psf, pos .- off,
                     view(wgt, roi...), view(dat, roi...); kwds...)
    return psf1, (pos1 .+ off)
end

offsets(A::AbstractArray{<:Any,N}, roi::ROI{N}) where {N} =
    offsets(axes(A), roi)

function offsets(inds::NTuple{N,AbstractUnitRange{<:Integer}},
                 roi::ROI{N}) where {N}
    offset(I::AbstractUnitRange{<:Integer}, ::Colon) =
        to_int(first(I)) - 1
    offset(I::AbstractUnitRange{<:Integer}, J::AbstractUnitRange{<:Integer}) =
        to_int(first(J)) - 1
    map(offset, inds, roi)
end

"""
    objfun([wgt,] dat, α, mdl, x0, y0)

yields the value of the objective function for PSF model `α*mdl(x-x0,y-y0)`
and data `dat` with optional weights `wgt`.  The objective function is
defined as:

    f(α, x0, y0) = sum_{x,y} wgt[x,y]*(dat[x,y] - α*mdl(x-x0,y-y0))^2

where the sum is evaluated for coordinates `(x,y)` taking all possible
values for the 2-dimensional array `dat`.  If specified, weights `wgt` must
all be nonnegative and have the same axes as `dat`; if not specified
`wgt[x,y] = 1` for all `(x,y)` is assumed.

The optimal value of the intensity parameter `α` may be automatically
computed given the other parameters:

    objfun([wgt,] dat, mdl, x0, y0, nonnegative=false)

yields the value of:

     min_α f(α, x0, y0) - c

where `c = f(0, x0, y0)` is an additive constant independent
of `(x0,y0)`.

As a convenience:

    objfun([wgt,] dat, trustvalues=false)

yields the value of the objective function when `α = 0`.  If optional
argument `trustvalues` is false, the data and weights values are checked
(data must have finite values and weights must be finite and nonnegative);
if `trustvalues` is true, it assumed that the data and the weights have
correct values and their values are not checked.

""" objfun

# Objective functions when alpha=0.

function objfun(dat::AbstractArray{T},
                trustvalues::Bool = false) where {T<:AbstractFloat}
    c = zero(Float64)
    if trustvalues
        @inbounds @simd for i in eachindex(dat)
            c += to_float(Float64, dat[i]^2)
        end
    else
        dflags = true
        @inbounds @simd for i in eachindex(dat)
            dflags &= isfinite(dat[i])
            c += to_float(Float64, dat[i]^2)
        end
        dflags || throw_bad_argument("there are invalid data values")
    end
    return c
end

function objfun(wgt::AbstractArray{T,N},
                dat::AbstractArray{T,N},
                trustvalues::Bool = false) where {T<:AbstractFloat,N}
    @assert axes(wgt) == axes(dat)
    c = zero(Float64)
    if trustvalues
        @inbounds @simd for i in eachindex(wgt, dat)
            c += to_float(Float64, wgt[i]*dat[i]^2)
        end
    else
        wflags = true
        dflags = true
        @inbounds @simd for i in eachindex(wgt, dat)
            wflags &= isfinite(wgt[i]) & (wgt[i] ≥ zero(wgt[i]))
            dflags &= isfinite(dat[i])
            c += to_float(Float64, wgt[i]*dat[i]^2)
        end
        wflags || throw_bad_argument("there are invalid weights")
        dflags || throw_bad_argument("there are invalid data values")
    end
    return c
end

# Objective functions when alpha is given.

function objfun(dat::AbstractArray{T,2},
                alpha::Real,
                mdl::Model,
                pos::NTuple{2,Real}) where {T<:AbstractFloat}
    objfun(dat, alpha, mdl, pos[1], pos[2])
end

function objfun(wgt::AbstractArray{T,2},
                dat::AbstractArray{T,2},
                alpha::Real,
                mdl::Model,
                pos::NTuple{2,Real}) where {T<:AbstractFloat}
    objfun(wgt, dat, alpha, mdl, pos[1], pos[2])
end

function objfun(dat::AbstractArray{T,2},
                alpha::Real,
                mdl::Model,
                x0::Real,
                y0::Real) where {T<:AbstractFloat}
    objfun(dat, to_float(T, alpha), mdl, to_float(T, x0), to_float(T, y0))
end

function objfun(wgt::AbstractArray{T,2},
                dat::AbstractArray{T,2},
                alpha::Real,
                mdl::Model,
                x0::Real,
                y0::Real) where {T<:AbstractFloat}
    objfun(wgt, dat, to_float(T, alpha), mdl, to_float(T, x0), to_float(T, y0))
end

function objfun(dat::AbstractArray{T,2},
                alpha::T,
                mdl::Model,
                x0::T,
                y0::T) where {T<:AbstractFloat}
    X, Y = axes(dat)
    c = zero(Float64)
    @inbounds for y in Y
        v = to_float(T, y) - y0
        @simd for x in X
            u = to_float(T, x) - x0
            m = mdl(u, v)
            d = dat[x,y]
            r = d - alpha*m
            c += r^2
        end
    end
    return c
end

function objfun(wgt::AbstractArray{T,2},
                dat::AbstractArray{T,2},
                alpha::T,
                mdl::Model,
                x0::T,
                y0::T) where {T<:AbstractFloat}
    X, Y = axes(dat)
    @assert axes(wgt) == (X, Y)
    c = zero(Float64)
    @inbounds for y in Y
        v = to_float(T, y) - y0
        @simd for x in X
            u = to_float(T, x) - x0
            m = mdl(u, v)
            w = wgt[x,y]
            d = dat[x,y]
            r = d - alpha*m
            c += w*r^2
        end
    end
    return c
end

# Objective functions that set alpha automatically.

function objfun(dat::AbstractArray{T,2},
                mdl::Model,
                pos::NTuple{2,Real},
                nonnegative::Bool = false) where {T<:AbstractFloat}
    objfun(dat, mdl, pos[1], pos[2], nonnegative)
end

function objfun(wgt::AbstractArray{T,2},
                dat::AbstractArray{T,2},
                mdl::Model,
                pos::NTuple{2,Real},
                nonnegative::Bool = false) where {T<:AbstractFloat}
    objfun(wgt, dat, mdl, pos[1], pos[2], nonnegative)
end

function objfun(dat::AbstractArray{T,2},
                mdl::Model,
                x0::Real,
                y0::Real,
                nonnegative::Bool = false) where {T<:AbstractFloat}
    objfun(dat, mdl, to_float(T, x0), to_float(T, y0), nonnegative)
end

function objfun(wgt::AbstractArray{T,2},
                dat::AbstractArray{T,2},
                mdl::Model,
                x0::Real,
                y0::Real,
                nonnegative::Bool = false) where {T<:AbstractFloat}
    objfun(wgt, dat, mdl, to_float(T, x0), to_float(T, y0), nonnegative)
end

function objfun(dat::AbstractArray{T,2},
                mdl::Model,
                x0::T,
                y0::T,
                nonnegative::Bool = false) where {T<:AbstractFloat}
    X, Y = axes(dat)
    a = zero(Float64)
    b = zero(Float64)
    @inbounds for y in Y
        v = to_float(T, y) - y0
        @simd for x in X
            u = to_float(T, x) - x0
            m = mdl(u, v)
            d = dat[x,y]
            a += m*m
            b += m*d
        end
    end
    if a > 0 && (nonnegative ? b > 0 : b != 0)
        # best alpha = b/a
        return -(b/a)*b
    else
        return zero(Float64)
    end
end

function objfun(wgt::AbstractArray{T,2},
                dat::AbstractArray{T,2},
                mdl::Model,
                x0::T,
                y0::T,
                nonnegative::Bool = false) where {T<:AbstractFloat}
    X, Y = axes(dat)
    @assert axes(wgt) == (X, Y)
    a = zero(Float64)
    b = zero(Float64)
    @inbounds for y in Y
        v = to_float(T, y) - y0
        @simd for x in X
            u = to_float(T, x) - x0
            m = mdl(u, v)
            w = wgt[x,y]
            d = dat[x,y]
            a += m*w*m
            b += m*w*d
        end
    end
    if a > 0 && (nonnegative ? b > 0 : b != 0)
        # best alpha = b/a
        return -(b/a)*b
    else
        return zero(Float64)
    end
end

# Objective functions designed for NEWUOA.

function objfun(dat::AbstractArray{T,2},
                ::Type{PSF},
                prm::Vector{Cdouble},
                nonnegative::Bool) where {T<:AbstractFloat,
                                          N,PSF<:AbstractPSF{N}}
    @assert length(prm) == N+2
    mdl = _model(PSF, prm)
    isvalid(mdl) ?
        Cdouble(objfun(dat, mdl, prm[1], prm[2], nonnegative)) :
        zero(Cdouble)

end

function objfun(wgt::AbstractArray{T,2},
                dat::AbstractArray{T,2},
                ::Type{PSF},
                prm::Vector{Cdouble},
                nonnegative::Bool) where {T<:AbstractFloat,
                                          N,PSF<:AbstractPSF{N}}
    @assert length(prm) == N+2
    mdl = _model(PSF, prm)
    isvalid(mdl) ?
        Cdouble(objfun(wgt, dat, mdl, prm[1], prm[2], nonnegative)) :
        zero(Cdouble)
end

@inline _model(::Type{PSF}, prm::Vector{Cdouble}) where {PSF<:AbstractPSF{1}} =
    PSF(prm[3])

@inline _model(::Type{PSF}, prm::Vector{Cdouble}) where {PSF<:AbstractPSF{2}} =
    PSF(prm[3], prm[4])

end # module
