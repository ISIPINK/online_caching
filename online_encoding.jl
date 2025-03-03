# code is generated by o3 mini after some debugging it works hopefully
using Random, Statistics, Plots

# Simple projection onto the set { y in [lower, upper]^N, sum(y)=1 }.
# This is alternating projections
function project_to_simplex_with_bounds(v::Vector{Float64}, lower::Float64, upper::Float64; tol=1e-8, maxiter=1000)
    y = clamp.(v, lower, upper)
    for _ in 1:maxiter
        s = sum(y)
        (abs(s - 1.0) < tol) && break
        adjustment = (1.0 - s) / length(y)
        y = clamp.(y .+ adjustment, lower, upper)
    end
    return y
end

"""
    online_gradient_descent(stream, N; η, ε)

Runs OGD on the loss ℓₜ(y) = -log(y[cₜ]) for a stream of characters.
- `stream`: a vector of integer indices (in 1:N).
- `N`: number of characters.
- `η`: learning rate.
- `ε`: lower bound for y.

Returns the final distribution `y` and the cumulative dynamic entropy.
"""
function online_gradient_descent(stream::Vector{Int}, N::Int; η=0.05, lower=1 / 512, upper = 0.5)
    # Initialize y 
    # y = fill(1 / N, N)
    y = zeros(N)
    y[1]= 0.5
    y[2]= 0.5
    y = project_to_simplex_with_bounds(y,lower,upper)

    dyn_entropy = 0.0
    for c in stream
        dyn_entropy += -log(y[c])
        y[c] +=  η/y[c] # gradient update
        y = project_to_simplex_with_bounds(y, lower, upper)
    end
    return y, dyn_entropy
end

# Compute static (Shannon) entropy in nats for a stream of indices.
function static_entropy(stream::Vector{Int}, N::Int)
    counts = zeros(Float64, N)
    for c in stream
        counts[c] += 1
    end
    p = counts ./ length(stream)
    return -sum(p[i] > 0 ? p[i]*log(p[i]) : 0.0 for i in 1:N)
end


function remap_to_contiguous(arr::Vector{Int})
    # Get the unique values and sort them
    uniq_vals = sort(unique(arr))
    
    # Create a dictionary mapping original values to new contiguous values
    mapping = Dict{Int,Int}()
    for (new_val, orig_val) in enumerate(uniq_vals)
        mapping[orig_val] = new_val
    end
    
    # Remap the array using the dictionary
    return [mapping[x] for x in arr]
end

function text_to_numbers(text::String)
    return remap_to_contiguous([Int(c) for c in text])
end



# Example usage:
N = 6
# Random.seed!(45)
stream = rand(1:N, 10000)  # simulated stream of characters
# stream = Int[]  # explicitly typed as an integer array
# for i in 1:N
#     append!(stream, fill(i, 1000))  # fill creates an array with 1000 copies of i
# end
final_y, dynamic_entropy = online_gradient_descent(stream, N; η=0.0005, lower = 0.001,upper = 1.0)
ste = static_entropy(stream, N)
println("Final distribution: ", final_y)
println("Dynamic entropy: ", dynamic_entropy/length(stream))
println("Static entropy: ", ste)


content =""
try
    content = read("example.txt", String)
    println(length(content))
catch e
    println("Error reading file: $e")
end

stream = text_to_numbers(content)
println("len ",length(stream))
println("len set ",length(Set(stream)))
println("minmax",minimum(stream)," ", maximum(stream))
N = maximum(stream)
final_y, dynamic_entropy = online_gradient_descent(stream, N; η=0.00001, lower = 0.001,upper = 1.0)
ste = static_entropy(stream, N)
println("Final distribution: ", final_y)
println("Dynamic entropy: ", dynamic_entropy/length(stream))
println("Static entropy: ", ste)