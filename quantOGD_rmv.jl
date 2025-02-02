using Random
using Profile
using PProf
include("2waycounter.jl")
using Distributions

mutable struct quantOGD
    ONE::Int
    overhead::Int
    lazyupdate:: Int
    stepsize:: Int
    adjstepsize::Int
    nonzeros::Int
    cache::TwoWayCounter
end
 
function Base.show(io::IO, q::quantOGD)
    println(io, "QuantOGD Parameters:")
    println(io, "-------------------------")
    println(io, "ONE:      ", q.ONE)
    println(io, "Overhead:  ", q.overhead)
    println(io, "Lazy Update: ", q.lazyupdate)
    println(io, "Step Size:  ", q.stepsize)
    println(io, "Nonzeros:   ", q.nonzeros)
    println(io, "Cache:      ", q.cache)
    println(io, "-------------------------")
end


function step!(q::quantOGD, i::Int)
    if haskey(q.cache.forward, i)
        q.adjstepsize = min(q.ONE - q.cache.forward[i] + q.lazyupdate, q.stepsize)
        add!(q.cache, i, q.adjstepsize)
        q.overhead += q.adjstepsize
    else
        q.nonzeros += 1
        add!(q.cache, i, q.lazyupdate + q.stepsize)
        q.overhead += q.stepsize
    end
    clean_cache!(q)             
end

# here can come in some logic to combine multiple steps
# the idea is to figure out how much overhead would
# disappear by planning, amortized constant time 
# follows from that only constant amount should 
# become zero 

function clean_cache!(q::quantOGD)
    if q.overhead >= q.nonzeros
        q.lazyupdate += 1
        q.overhead -= q.nonzeros
        for i in sort(get_indexes(q.cache, q.lazyupdate))
            remove!(q.cache,i)     
            q.nonzeros -=1
        end
        clean_cache!(q)
    end
end

function get_fraction(q::quantOGD,i)
    if haskey(q.cache.forward, i)
        return (q.cache.forward[i]-q.lazyupdate)/q.ONE
    else
        return 0
    end
end


# Constructor for quantOGD
function init_quantOGD(;
    N = 100,
    C = 10,
    ONE = 100,
    stepsize_real = 10,
    lazyupdate = 0,
    adjstepsize = 0,
    cache = TwoWayCounter())
    stepsize = floor(stepsize_real*ONE)
    nonzeros = 0
    a = min(div(ONE*C,N),ONE)
    for i in 1:N
        add!(cache, i, a)
        nonzeros +=1
    end
    overhead =  N *a-C*ONE
    return quantOGD( ONE, overhead, lazyupdate, stepsize,adjstepsize, nonzeros, cache)
end


# Demonstrate the step! function
function test_time()

@time begin 
    for _ in 1:10^6
        step!(q, rand(1:10^4))
    end
end
end

function profile()
Profile.clear()
@pprof begin 
    for _ in 1:10^5
        step!(q, rand(1:6))
    end
end
pprof()
end


function test_init_quantOGD()
    q = init_quantOGD(N=1000,C= 50, ONE = 1000, stepsize_real = 0.0217)
println(q)
end
# test_init_quantOGD()