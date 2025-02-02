using Plots
struct ZipfSampler
    s::Float64
    N::Int
    cdf::Vector{Float64}
end

function ZipfSampler(s::Real, N::Integer)
    if N ≤ 0
        throw(ArgumentError("N must be a positive integer"))
    end
    if s ≤ 0
        throw(ArgumentError("s must be a positive real number"))
    end
    
    function generalized_harmonic(N, s)
        H = 0.0
        for k in 1:N
            H += 1.0 / (k^s)
        end
        H
    end
    
    H = generalized_harmonic(N, Float64(s))
    probabilities = [1.0 / (k^s) for k in 1:N] ./ H
    cdf = cumsum(probabilities)
    cdf[end] = 1.0  # Ensure the CDF ends exactly at 1.0
    
    return ZipfSampler(Float64(s), N, cdf)
end

function sample(z::ZipfSampler)
    u = rand()
    return searchsortedfirst(z.cdf, u)
end

# Example usage:

function test_zipf_sample()
    zipf = ZipfSampler(1.0, 100)
    ss = [ sample(zipf) for _ in 1:10^2]
    println(ss)
end