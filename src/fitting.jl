#
# fitting.jl --
#
# Fitting Point Spread Functions (PSF) models for Julia.
#

module Fitting

import OptimPack.Powell.Newuoa
using ..PointSpreadFunctions
using ..PointSpreadFunctions: AbstractPSF

"""
    fit(psf, pos, [wgt,] dat; nonnegative=false) -> psf′, pos′

fits a given Point Spread Function (PSF) model on data `dat` with
(optional) respective weights `wgt`.  Argument `psf` is a PSF instance to
specify which PSF model to use and the initial parameters of the model.
Argument `pos` is the initial PSF position.  The result is a 2-tuple with
the fitted PSF model and position.

The fit is carried out by `OptimPack.Powell.Newuoa.minimize` method.

Keyword `nonnegative` indicates whether the instensity of the PSF should be
nonnegative or not.  Keyword `rho = (rhobeg,rhoend)` may be used to specify
the initial and final precision on the parameters.  Additional keywwords
may be specified and are passed to the minimizer.

"""
function fit(psf::PSF,
             pos::NTuple{2,Real},
             dat::AbstractArray{T,2};
             rho::Tuple{Real,Real} = (0.1, 1e-5),
             nonnegative::Bool = false,
             kwds...) where {T<:AbstractFloat,
                             N,PSF<:AbstractPSF{N}}
    x = T[pos..., psf[:]...]
    ans = Newuoa.minimize!(x -> objfun1(dat, PSF(x[3:N+2]...), x[1], x[2], nonnegative),
                           x, rho...; kwds...)
    return PSF(x[3:N+2]...), (x[1], x[2])
end

function fit(psf::PSF,
             pos::NTuple{2,Real},
             wgt::AbstractArray{T,2},
             dat::AbstractArray{T,2};
             rho::Tuple{Real,Real} = (0.1, 1e-5),
             nonnegative::Bool = false,
             kwds...) where {T<:AbstractFloat,
                             N,PSF<:AbstractPSF{N}}
    @assert axes(wgt) == axes(dat)
    x = T[pos..., psf[:]...]
    ans = Newuoa.minimize!(x -> objfun1(wgt, dat, PSF(x[3:N+2]...), x[1], x[2], nonnegative),
                           x, rho...; kwds...)
    return PSF(x[3:N+2]...), (x[1], x[2])
end

function objfun0(dat::AbstractArray{T,2},
                 alpha::T,
                 mdl,
                 x0::T,
                 y0::T) where {T<:AbstractFloat}
    X, Y = axes(dat)
    c = zero(Float64)
    @inbounds for y in Y
        v = T(y) - y0
        @simd for x in X
            u = T(x) - x0
            m = mdl(u, v)
            d = dat[x,y]
            r = d - alpha*m
            c += r^2
        end
    end
    return c
end

function objfun0(wgt::AbstractArray{T,2},
                 dat::AbstractArray{T,2},
                 mdl,
                 alpha::T,
                 x0::T,
                 y0::T) where {T<:AbstractFloat}
    X, Y = axes(dat)
    @assert axes(wgt) == (X, Y)
    c = zero(Float64)
    @inbounds for y in Y
        v = T(y) - y0
        @simd for x in X
            u = T(x) - x0
            m = mdl(u, v)
            w = wgt[x,y]
            d = dat[x,y]
            r = d - alpha*m
            c += w*r^2
        end
    end
    return c
end

function objfun1(dat::AbstractArray{T,2},
                 mdl,
                 x0::T,
                 y0::T,
                 nonnegative::Bool = false) where {T<:AbstractFloat}
    X, Y = axes(dat)
    a = zero(Float64)
    b = zero(Float64)
    @inbounds for y in Y
        v = T(y) - y0
         @simd for x in X
            u = T(x) - x0
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

function objfun1(wgt::AbstractArray{T,2},
                 dat::AbstractArray{T,2},
                 mdl,
                 x0::T,
                 y0::T,
                 nonnegative::Bool = false) where {T<:AbstractFloat}
    X, Y = axes(dat)
    @assert axes(wgt) == (X, Y)
    a = zero(Float64)
    b = zero(Float64)
    @inbounds for y in Y
        v = T(y) - y0
         @simd for x in X
            u = T(x) - x0
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

end # module
