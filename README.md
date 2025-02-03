# online_caching

The goal of this repo is to reproduce and extend http://arxiv.org/abs/2405.01263 (An Online Gradient-Based Caching Policy with Logarithmic Complexity and Regret Guarantees).

I ended up with a modified version of the algorithm that probably is more efficient in time. The current version of the algorithm for the fractional caching problem can be found in quantOGD.jl and the integral caching problem in quantOGD_integral.jl.

I should probably benchmark on some traces. There are some optimization I still can do but current implementation is close (a factor of 4) to the optimal by my performance skillset without going parallel.