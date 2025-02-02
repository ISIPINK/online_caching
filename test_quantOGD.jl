using Plots
include("quantOGD_rmv.jl")
include("zipf.jl")


T = 10^5
N = 10^4
C = div(N,20)
stepsize_real = sqrt( C*(1- C/N)/T)

zipf = ZipfSampler(0.8, N)
zipf_trace = [ sample(zipf) for _ in 1:T ]


# histogram([x for x in zipf_trace if x<1000])

q = init_quantOGD(N=N, C=C, ONE = 10000, stepsize_real=stepsize_real)

println(get_fraction(q,1))
println(q.stepsize)

hit = 0
hits = zeros(Float32, length(zipf_trace)) 
cachesizes = zeros(Float32, length(zipf_trace)) 
 

for i in 1:length(zipf_trace)
    j = zipf_trace[i]
    hit += get_fraction(q, j)
    hits[i] = hit
    cachesizes[i] = q.overhead/q.ONE + C
    # q.stepsize = floor(stepsize_real*sqrt(T/2(T-i+1) )*q.ONE)
    step!(q,j)
end

p = plot(cachesizes, label = "cachesizes")
hline!([C+ N/q.ONE], label="overhead bound", color=:red, linestyle=:dash)
xlabel!("time")
ylabel!("cachesize at time")
title!("cachesizes")
display(p)

# Calculate opt
opt = mean(zipf_trace .<= C)

# Generate the decay curve
decay_curve = hits ./ (1:length(zipf_trace))

# Create the plot
p = plot(decay_curve, label="quantOGD")
hline!([opt], label="static_opt_hind", color=:red, linestyle=:dash)
xlabel!("time")
ylabel!("hitrate until up to time")
title!("hitrate on Zipf Trace")
display(p)