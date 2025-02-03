# online_caching

The goal of this repo is to reproduce and extend http://arxiv.org/abs/2405.01263 (An Online Gradient-Based Caching Policy with Logarithmic Complexity and Regret Guarantees).

I ended up with a modified version of the algorithm that probably is more efficient in time. The current version of the algorithm for the fractional caching problem can be found in quantOGD.jl and the integral caching problem in quantOGD_integral.jl. The 2 algorithms are based on the fact you can quantize all operations in the original algorithm of the paper by projecting/resizing the cache to a clever chosen size slightly bigger then the provided cache, this hopefully preserves the regret guarantee of the original algorithm and make optimizations based on quantized comparisons/operations possible. 

I should probably benchmark on some traces. There are some optimizations I still can do but current implementation is close (a factor of 4) to the optimal by my performance skillset without going parallel.