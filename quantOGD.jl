using Random
using Distributions
using Profile
using PProf
using Plots
include("zipf.jl")

mutable struct quant_OGD{Typ_count <: Integer, Typ_ind<:Integer}
    ONE::Typ_count
    overhead::Typ_count
    lazy_update::Typ_count
    step_size::Typ_count
    adj_step_size::Typ_count
    nonzeros::Typ_ind
    counter::Dict{Typ_ind, Typ_count}
    counter_val::Dict{Typ_count, Typ_ind}
end
 
function Base.show(io::IO, q::quant_OGD)
    println(io, "QuantOGD Parameters:")
    println(io, "-------------------------")
    println(io, "ONE:      ", q.ONE)
    println(io, "Overhead:  ", q.overhead)
    println(io, "Lazy Update: ", q.lazy_update)
    println(io, "Step Size:  ", q.step_size)
    println(io, "Nonzeros:   ", q.nonzeros)
    println(io, "counter:      ", q.counter)
    println(io, "counter_val:      ", q.counter_val)
    println(io, "-------------------------")
end
 

function step!(q::quant_OGD, i::Int)
    q.counter_val[q.counter[i]] -= 1
    if q.counter[i] > q.lazy_update # is component strictly positive?
        q.adj_step_size = min(q.ONE - q.counter[i] + q.lazy_update, q.step_size)
        q.counter[i] += q.adj_step_size
        q.overhead += q.adj_step_size
    else
        q.nonzeros += 1
        q.counter[i] = q.lazy_update + q.step_size
        q.overhead += q.step_size
    end
    q.counter_val[q.counter[i]] = get(q.counter_val, q.counter[i], 0) + 1 #alloc
    resize_cache!(q)             
end


function resize_cache!(q::quant_OGD)
    while q.overhead >= q.nonzeros
        q.lazy_update += 1
        q.overhead -= q.nonzeros
        q.nonzeros -= get(q.counter_val, q.lazy_update, 0) 
    end
end


get_fraction(q::quant_OGD, i) = q.counter[i] > q.lazy_update ? (q.counter[i] - q.lazy_update) / q.ONE : 0


# Constructor for quantOGD
function init_quant_OGD(;
    N = 100,
    C = 10,
    overhead_cache_rate = 0.01,
    stepsize_real = 0.01,
    Typ_count = UInt32
    )
    ONE = Int(ceil(1/overhead_cache_rate)*div(N,C))
    stepsize = max(floor(stepsize_real*ONE),1)
    nonzeros = N
    adjstepsize = 0
    lazyupdate = 0
    init_count = min(div(ONE*C,N),ONE)
    counter = Dict(i => init_count for i in 1:N)
    counter_val = Dict(init_count=>N) 
    overhead =  N *init_count-C*ONE
    return quant_OGD{Typ_count,UInt16}( ONE, overhead, lazyupdate, stepsize,adjstepsize, nonzeros, counter, counter_val)
end

function zipf_setup(;T=10^5,N=10^4, overhead_cache_rate=0.03, alpha=1.0,Typ_count=UInt32)
    C = div(N,20)

    zipf = ZipfSampler(alpha, N)
    zipf_trace = [ sample(zipf) for _ in 1:T ]

    stepsize_real = sqrt( C*(1- C/N)/T)
    q = init_quant_OGD(N=N, C=C, overhead_cache_rate = overhead_cache_rate, stepsize_real=stepsize_real, Typ_count=Typ_count)
    return q , zipf_trace
end

function test_time(;T=10^5,N=10^4,overhead_cache_rate=0.03,alpha=1.00)
    q, zipf_trace = zipf_setup(T=T,N=N,overhead_cache_rate=overhead_cache_rate, alpha=alpha)
    @time begin 
        for j in zipf_trace
            step!(q, j)
        end
    end
    # println(q)
end

function profile(;T=10^5,N=10^4,overhead_cache_rate=0.03, alpha=1.0)
    q, zipf_trace = zipf_setup(T=T,N=N,overhead_cache_rate=overhead_cache_rate, alpha=alpha)
    Profile.clear()
    @pprof begin 
        for j in zipf_trace
            step!(q, j)
        end
    end
    pprof()
end


function zipf_hitrate_plt(;T=10^5,N=10^4,overhead_cache_rate=0.03, alpha=1.0, window = 1000)
    q, zipf_trace = zipf_setup(T=T,N=N,overhead_cache_rate=overhead_cache_rate, alpha=alpha)
    C = div(N, 20)
    hit = 0.0
    hits = zeros(Float32, length(zipf_trace))
    cachesizes = zeros(Float32, length(zipf_trace))
    println("ONE:",q.ONE)
    println("step_size:",q.step_size)

    for i in 1:length(zipf_trace)
        j = zipf_trace[i]
        hit += get_fraction(q, j)
        hits[i] = hit
        cachesizes[i] = C + q.overhead / q.ONE
        step!(q, j)
    end
    ss = div(length(hits),10^4)  # this for subsampling  
    # Compute the moving maximum over the cachesizes array using the specified window.
    mov_max = [maximum(cachesizes[max(1, i - window + 1) : i]) for i in 1:length(cachesizes)]

    # First plot: moving maximum of the cache sizes.
    p1 = plot(mov_max[1:ss:end],
              label = "Moving Max Cache Size",
              color = :blue,
              xlabel = "time/$ss",
              ylabel = "cache size",
              title = "Cache Size (Moving Max)")
    hline!(p1, [C + N / q.ONE],
           label = "overhead bound",
           color = :red,
           linestyle = :dash)

    # Calculate opt (static optimal hitrate)
    opt = mean(zipf_trace .<= C)
    # Generate the decay curve: cumulative hitrate over time.
    decay_curve = hits ./ (1:length(zipf_trace))

    # Second plot: hitrate decay curve.
    p2 = plot(decay_curve[1:ss:end],
              label = "quantOGD",
              color = :green,
              xlabel = "time/$ss",
              ylabel = "hitrate",
              title = "Hitrate on Zipf Trace")
    hline!(p2, [opt],
           label = "static_opt_hind",
           color = :red,
           linestyle = :dash)

    # Stack the two plots vertically.
    p = plot(p1, p2, layout = (2, 1), size=(600, 800))
    display(p)
end


using Random
function adv_hitrate_plt(; T=10^5, N=10^4, overhead_cache_rate=0.03, alpha=1.0, window=1000, Typ_count=UInt32)

    tmp = collect(1:1000)
    tmp1 = collect(1001:2000)
    l = [] 
    for _ in 1:500
        append!(l,shuffle(tmp))
    end

    for _ in 1:500
        append!(l,shuffle(tmp1))
    end
    l = [ Int(x) for x in l]

    N = length(tmp)+ length(tmp1)
    T = length(l)

    C = div(N, 20)
    hit = 0.0
    hits = zeros(Float32, length(l))
    cachesizes = zeros(Float32, length(l))

    q, zipf_trace = zipf_setup(T=T, N=N, overhead_cache_rate=overhead_cache_rate, alpha=alpha,Typ_count=Typ_count)
    println("ONE:",q.ONE)
    println("step:",q.step_size)
    for i in 1:length(l)
        j = l[i]
        hit += get_fraction(q,j) 
        hits[i] = hit
        cachesizes[i] = C+q.overhead/q.ONE 
        step!(q, j)
    end
    ss = max(div(length(hits), 10^2),1)  # this for subsampling  
    # Compute the moving maximum over the cachesizes array using the specified window.
    # mov_max = [maximum(cachesizes[max(1, i - window + 1):i]) for i in 1:length(cachesizes)]

    # First plot: moving maximum of the cache sizes.
    p1 = plot(cachesizes[1:ss:end],
        label="Cache Size",
        color=:blue,
        xlabel="time/$ss",
        ylabel="cache size",
        title="subsampled cachesize ")
    hline!(p1, [C + N / q.ONE],
        label="C_UPPER",
        color=:red,
        linestyle=:dash)

    # Calculate opt (static optimal hitrate)
    opt = C/N 
    # Generate the decay curve: cumulative hitrate over time.
    decay_curve = hits ./ (1:length(l))

    # Second plot: hitrate decay curve.
    p2 = plot(decay_curve[1:ss:end],
        label="quantOGD",
        color=:green,
        xlabel="time/$ss",
        ylabel="hitrate",
        title="Hitrate on adv trace")
    hline!(p2, [opt],
        label="static_opt_hind",
        color=:red,
        linestyle=:dash)

    # Stack the two plots vertically.
    p = plot(p1, p2, layout=(2, 1), size=(600, 800))
    display(p)
end
# profile(T = 10^6,N=10^4,overhead_cache_rate=0.0001)
# test_time(T = 10^7, N=10^4, overhead_cache_rate =0.01, alpha=0.6)
# zipf_hitrate_plt(T=10^7, N= 10^4, overhead_cache_rate = 0.01, alpha = 0.9)
adv_hitrate_plt(T=10^6, N=10^3, overhead_cache_rate=0.01, alpha=0.8, Typ_count=UInt16)
