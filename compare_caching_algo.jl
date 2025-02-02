# o3 mini and deepseek r1 made these plots and classic algos
using DataStructures, Plots
include("classic_caching.jl")
include("quantOGD_rmv.jl")
include("zipf.jl")

# Parameters for Zipf trace and quantOGD
T = 10^6
N = 10^3
C = div(N, 20)
stepsize_real = sqrt( C*(1- C/N)/T )

# Generate Zipf trace
zipf = ZipfSampler(0.8, N)
zipf_trace = [ sample(zipf) for _ in 1:T ]

# --- Simulation for classic caches ---

# We'll simulate FIFO, LRU, LFU, and Optimal on zipf_trace.
# For FIFO, LRU, LFU the process_request function expects a key,
# whereas OptimalCache requires the current index.
function simulate_cache(cache_name::String, trace::Vector{Int}, capacity::Int)
    hits = zeros(Float64, length(trace))
    hit_count = 0
    if cache_name == "Optimal"
        cache = OptimalCache(capacity, trace)
        for i in 1:length(trace)
            if process_request(cache, i)  # pass the index for OptimalCache
                hit_count += 1
            end
            hits[i] = hit_count
        end
    else
        # Instantiate the appropriate cache type.
        if cache_name == "FIFO"
            cache = FIFOCache(capacity)
        elseif cache_name == "LRU"
            cache = LRUCache(capacity)
        elseif cache_name == "LFU"
            cache = LFUCache(capacity)
        else
            error("Unknown cache name: $cache_name")
        end
        for i in 1:length(trace)
            key = trace[i]
            if process_request(cache, key)
                hit_count += 1
            end
            hits[i] = hit_count
        end
    end
    return hits
end

# Run simulations for all classical algorithms
cache_names = ["FIFO", "LRU", "LFU", "Optimal"]
results = Dict{String, Vector{Float64}}()
for name in cache_names
    println("Simulating $name cache...")
    results[name] = simulate_cache(name, zipf_trace, C)
end

# --- Simulation for quantOGD ---
# Initialize quantOGD from the included file
q = init_quantOGD(N=N, C=C, ONE=1000, stepsize_real=stepsize_real)
quant_hits = zeros(Float64, length(zipf_trace))
hit = 0
for i in 1:length(zipf_trace)
    j = zipf_trace[i]
    hit += get_fraction(q, j)
    quant_hits[i] = hit
    step!(q, j)
end

# Compute running hit rates (hit rate = cumulative hits / time step)
function hit_rate_curve(cumulative_hits::Vector{Float64})
    return cumulative_hits ./ (1:length(cumulative_hits))
end

curves = Dict{String, Vector{Float64}}()
for (name, cum_hits) in results
    curves[name] = hit_rate_curve(cum_hits)
end
curves["quantOGD"] = hit_rate_curve(quant_hits)

# Static optimal hit rate (e.g. when the optimal static cache would cover keys 1..C)
# Here we approximate it as the fraction of requests with key ≤ C.
static_opt = mean(zipf_trace .<= C)

# --- Plotting ---
p = plot(title="Running Hit Rate on Zipf Trace",
         xlabel="Time (requests)", ylabel="Hit rate",
         legend=:bottomright)

# Plot classical algorithms and quantOGD
for (name, curve) in curves
    plot!(p, curve, label=name)
end

# Add a horizontal line for the static optimal hit rate.
hline!(p, [static_opt], label="Static optimal", linestyle=:dash, color=:black)

display(p)
