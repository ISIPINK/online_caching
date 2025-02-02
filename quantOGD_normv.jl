using Random
using Profile
using PProf
include("2waycounter.jl")

mutable struct quantOGD
    ACCURACY::Int
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
    println(io, "ACCURACY: ", q.ACCURACY)
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
        if q.cache.forward[i]>q.lazyupdate
            q.adjstepsize = min(q.ONE - q.cache.forward[i] + q.lazyupdate, q.stepsize)
            # the add still does removals
            add!(q.cache, i, q.adjstepsize)
            q.overhead += q.adjstepsize
        else
            q.nonzeros += 1
            add!(q.cache, i, q.lazyupdate + q.stepsize - q.cache.forward[i])
            q.overhead += q.stepsize
        end

    else
        q.nonzeros += 1
        add!(q.cache, i, q.lazyupdate + q.stepsize)
        q.overhead += q.stepsize
    end

    if q.overhead >= q.nonzeros 
        clean_cache!(q)             
    end
end

function clean_cache!(q::quantOGD)
    q.lazyupdate += 1
    q.overhead -= q.nonzeros
    # we dont remove it
    # for i in sort(get_indexes(q.cache, q.lazyupdate))
    #     remove!(q.cache,i)     
    #     q.nonzeros -=1
    # end
    q.nonzeros -= length(get_indexes(q.cache, q.lazyupdate))
    if q.overhead >= q.nonzeros 
        clean_cache!(q)             
    end
end


# Constructor for quantOGD
function quantOGD(;
    ACCURACY = 1000,
    ONE = 100,
    overhead = 0,
    lazyupdate = 0,
    stepsize = 10,
    adjstepsize = 0,
    nonzeros = 0,
    cache = TwoWayCounter())

    add!(cache, 1, 100)
    add!(cache, 2, 100)
    add!(cache, 3, 100)
    nonzeros +=3
    return quantOGD(ACCURACY, ONE, overhead, lazyupdate, stepsize,adjstepsize, nonzeros, cache)
end

# Example usage: Create an instance of quantOGD
q = quantOGD(ACCURACY = 1000, ONE = 100, stepsize = 10)
println(q)

# Demonstrate the step! function
step!(q, 1)
step!(q, 2)
step!(q, 4)
step!(q, 6)
step!(q, 5)
println(q)

@time begin 
    for _ in 1:10^5
        step!(q, rand(1:6))
    end
end

Profile.clear()
@pprof begin 
    for _ in 1:10^5
        step!(q, rand(1:6))
    end
end
pprof()
println(q)

println(q)
step!(q, 3)
step!(q, 5)

# Demonstrate the clean_cache! function
clean_cache!(q)

# Further demonstration
step!(q, 7)