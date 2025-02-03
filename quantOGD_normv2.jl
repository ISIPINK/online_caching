using Profile
using PProf
using Plots
include("zipf.jl")

mutable struct quant_OGD_normv{T <: Integer}
    ONE::T
    overhead::T
    lazy_update::T
    step_size::T
    adj_step_size::T
    nonzeros::T
    counter::Dict{T, T}
    counter_val::Dict{T, T}
end
 
function Base.show(io::IO, q::quant_OGD_normv)
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


function step!(q::quant_OGD_normv, i::Int)
    q.counter_val[q.counter[i]] -= 1
    if q.counter[i] > q.lazy_update
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


function resize_cache!(q::quant_OGD_normv)
    if q.overhead >= q.nonzeros
        q.lazy_update += 1
        q.overhead -= q.nonzeros
        q.nonzeros -= get(q.counter_val, q.lazy_update, 0) 
        resize_cache!(q)
    end
end


get_fraction(q::quant_OGD_normv, i) = q.counter[i] > q.lazy_update ? (q.counter[i] - q.lazy_update) / q.ONE : 0


# Constructor for quantOGD
function init_quant_OGD_normv(;
    N = 100,
    C = 10,
    overhead_cache_rate = 0.01,
    stepsize_real = 0.01,
    )
    ONE = Int(ceil(1/overhead_cache_rate)*div(N,C))
    stepsize = floor(stepsize_real*ONE)
    nonzeros = N
    adjstepsize = 0
    lazyupdate = 0
    a = min(div(ONE*C,N),ONE)
    counter = Dict(i => a for i in 1:N)
    counter_val = Dict(a=>N) 
    overhead =  N *a-C*ONE
    return quant_OGD_normv{UInt32}( ONE, overhead, lazyupdate, stepsize,adjstepsize, nonzeros, counter, counter_val)
end

function test_init_quant_OGD_normv()
    T = 10^3
    N = 10^2
    C = div(N,20)
    overhead_cache_rate = 0.01
    stepsize_real = sqrt( C*(1- C/N)/T)
    q = init_quant_OGD_normv(N=N, C=C, overhead_cache_rate = overhead_cache_rate, stepsize_real=stepsize_real)
    println(q)
end
test_init_quant_OGD_normv()

function zipf_setup_normv(;T=10^5,N=10^4, overhead_cache_rate=0.03, alpha=1.0)
    C = div(N,20)

    zipf = ZipfSampler(alpha, N)
    zipf_trace = [ sample(zipf) for _ in 1:T ]

    stepsize_real = sqrt( C*(1- C/N)/T)
    q = init_quant_OGD_normv(N=N, C=C, overhead_cache_rate = overhead_cache_rate, stepsize_real=stepsize_real)
    return q , zipf_trace
end

function test_time(;T=10^5,N=10^4,overhead_cache_rate=0.03,alpha=1.00)
    q, zipf_trace = zipf_setup_normv(T=T,N=N,overhead_cache_rate=overhead_cache_rate, alpha=alpha)
    @time begin 
        for j in zipf_trace
            step!(q, j)
        end
    end
    # println(q)
end
# test_time(T = 10^7, N=10^4, overhead_cache_rate =0.01, alpha=0.6)

function profile(;T=10^5,N=10^4,overhead_cache_rate=0.03, alpha=1.0)
    q, zipf_trace = zipf_setup_normv(T=T,N=N,overhead_cache_rate=overhead_cache_rate, alpha=alpha)
    Profile.clear()
    @pprof begin 
        for j in zipf_trace
            step!(q, j)
        end
    end
    pprof()
end
# profile(T = 10^7,N=10^4)


function zipf_hitrate_plt(;T=10^5,N=10^4,overhead_cache_rate=0.03, alpha=1.0, window = 1000)
    q, zipf_trace = zipf_setup_normv(T=T,N=N,overhead_cache_rate=overhead_cache_rate, alpha=alpha)
    C = div(N, 20)
    hit = 0.0
    hits = zeros(Float32, length(zipf_trace))
    cachesizes = zeros(Float32, length(zipf_trace))

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


zipf_hitrate_plt(T=10^6, N= 10^4, overhead_cache_rate = 0.01, alpha = 0.9)