# Introduction

The goal of this repo was to reproduce and extend http://arxiv.org/abs/2405.01263 (An Online Gradient-Based Caching Policy with Logarithmic Complexity and Regret Guarantees). We ended up with a quantized version of the algorithm.  

Here is a small explainer on how the quantized algorithm works, we haven't put in the time to explain all details.

# Online Linear Optimization (OLO)

The caching algorithm in this repo is based on Online Linear Optimization.

## Static Regret
Online Linear Optimization (OLO) is a framework where, at each round, a decision maker selects an action from a convex set before a linear loss function is revealed. The goal is to minimize the cumulative loss compared to the best fixed action in hindsight—a difference known as **static regret**.

In OLO, the **static regret** after $T$ rounds is defined as
$$
\text{Regret}_T = \sum_{t=1}^T \langle g_t, x_t \rangle - \min_{x \in K} \sum_{t=1}^T \langle g_t, x \rangle,
$$ 

where $x_t \in K$ is the decision at round $t$, $g_t$ is the gradient (or loss vector) revealed at round $t$ , and $K$ is the convex decision set. We assume that losses in this case $g_{t}$ is fixed but unknown before starting the algorithm.


## Online Projected Gradient Descent (OPGD)
A popular algorithm in this setting is **Online Projected Gradient Descent (OPGD)**. At each iteration, the algorithm updates the current decision by stepping in the direction of the negative gradient of the loss and then projects back onto the feasible set to maintain constraints. This approach leverages the convexity of the decision set and the linearity of losses, offering a simple yet effective method for minimizing static regret over time.


The update rule for OPGD is

$$
x_{t+1} = \Pi_K\Bigl( x_t - \eta\, g_t \Bigr),
$$ 

where $\eta > 0$ is the learning rate, and $\Pi_K(y) = \arg\min_{x \in K} \|x - y\|$   denotes the projection of $y$  onto the set $K$.

## Regret bound for OPGD
A central result in OLO is that **OPGD achieves a regret bound on the order of $\sqrt{T}$** for a well chosen $\eta$ when $K$ is bounded and convex, meaning that

$$
\text{Regret}_T = O(\sqrt{T}).
$$ 

This $\sqrt{T}$ bound is important because it implies that the **average regret per round**, $\frac{\text{Regret}_T}{T}$, tends to zero as $T$ increases, ensuring that the algorithm performs nearly as well as the best fixed decision in hindsight even against adversarial losses.

The key intuition behind this result is that, with an appropriately chosen learning rate $\eta$, the incremental loss incurred by each OPGD update can be controlled through the properties of convexity and the geometry of the projection step. By carefully balancing the step size and the accumulated errors (using a telescoping sum argument), one shows that the total deviation from the best fixed action does not exceed a term proportional to $\sqrt{T}$. This makes OPGD particularly effective in the online setting, where decisions must be made sequentially without prior knowledge of future loss functions.


## Sources OLO
- o3 mini (
write a text introducing OLO online linear optimization, keep it short introduce static regret and online projected gradient descent, explain that an important result or the essence why OLO works is that OPGD obtains sqrt T regret
)

- Hazan: http://arxiv.org/abs/1909.05207 (Introduction to Online Convex Optimization)

- Ashoks Thesis: https://www-cs.stanford.edu/people/ashokc/papers/thesis.pdf (PhD on PARAMETER-FREE ONLINE
LEARNING )

- Orabona:  http://arxiv.org/abs/1912.13213 (A Modern Introduction to Online Learning)

- convex set wiki: https://en.wikipedia.org/wiki/Convex_set

# OLO framework for caching


Fractional caching can be analyzed as a OLO problem. 

## Representing fractional caching strategy
Let $y$ be a vector where each component $y_{i}$ represents the fraction of item $i$ that is cached,  the set of feasibility for y is (for a cache of size $C$): 

$$
Y = \left\{ y \in [0,1]^N \middle| \sum_{n=1}^N y^{n} \leq C \right\},
$$ 

with $N$ the max possible different items, as there always exist a full cache strategy that outperforms a non full cache strategy all our fractional cache states will be in

$$
Y_{full} = \left\{ y \in [0,1]^N \middle| \sum_{n=1}^N y^{n} = C \right\}.
$$ 
Note that the feasibility set is convex, for integral caching this is not the case so OLO isn't directly applicable.


## Optimal Factional Static Cache 

To understand how fractional caching is an OLO problem, you can also understand how you can frame finding the optimal fractional static cache in hindsight as a constrained convex linear optimization problem or in this case linear programming problem because 
$\mathcal{Y}$ is a linear constraint.

To pose a optimization problem we should chose our goal, loss/utility function. In the case of caching our goal is to minimize the miss rate or equivalently maximize the hit rate. We define the hit rate for a sequence of requests $l_i \in \mathbb{N}$ ($l_{t}$ is the number of the $t$ th requested item) of length $T$ for a static cache $y \in Y_{full}$ as:

$$
H_{T}(y) = \frac{1}{T}\sum_{t=1}^{T} y^{l_t} .
$$ 
to keep the $g_{t}$ the same for different $T$ later we ignore $\frac{1}{T}$ and added it later in doing this preserves the optimization problem, to make this similar to the static regret definition which uses minimization we are going to minimize the negative hit rate and use basis vectors $e^{l_{t}}$ to select component $l_{t}$ of $y$ with the inner product, so finding the optimal fractional static cache in hindsight $y_{opt}$ is framed as following optimization problem:

$$
\begin{align*}
y_{opt} &= \argmin_{y \in Y_{full}} \left(- \sum_{t=1}^{T} \langle e^{l_{t}} , y \rangle \right) \\
        &= \argmin_{y \in Y} \left( \sum_{t=1}^{T} \langle g_{t} , y \rangle \right) 
\end{align*}
$$ 
with $g_{t} = - e^{l_{t}} $, the $g_{t}$ s are the gradients in our OLO caching problem. Now formulating caching as an OLO problem is obvious. An interesting extension is to let $C$ be variable at each time step and adding $C$ to the loss function and constraining it to an interval this way the online caching strategy can also decide on how big to chose the cache. 

## OPGD for caching

Applying OPGD on our fractional caching formulation defines following online fractional caching strategy initialize with an arbitrary $y_{0}\in Y$ then:

\begin{align*}
y_{t+1} &= \Pi_Y \Bigl( y_t - \eta\, g_t \Bigr), \\
        &= \Pi_Y \Bigl( y_t + \eta\, e^{l_{t}} \Bigr) 

\end{align*}



this strategy comes with the regret guarantee of OPGD: 

\begin{align*}
O(\sqrt{T}) &= \sum_{t=1}^T \langle g_t, y_t \rangle -  \sum_{t=1}^T \langle g_t, y_{OPT} \rangle,  \Leftrightarrow \\
\frac{O(\sqrt{T})}{T} &= \frac{1}{T}\sum_{t=1}^T \langle g_t, y_t \rangle -  \frac{1}{T} \sum_{t=1}^T \langle g_t, y_{OPT} \rangle, \Leftrightarrow \\
\frac{O(\sqrt{T})}{T} &= - H_{T}(y_{t}) +  H_{T}(y_{OPT}), \Leftrightarrow \\
\frac{O(\sqrt{T})}{T} &=  H_{T}(y_{OPT})- H_{T}(y_{t})  ,  \\
\end{align*}

this means as $T\rightarrow \infty$ the hit rate of the online strategy converges to the hit rate of the static optimal in hindsight.


## Sources OLO framework for caching
- o3 mini (explain the OLO framework for fractional caching)
- paper Paschos: http://arxiv.org/abs/1904.09849 (introduces OGA $\approx$ OPGD for caching)
- PhD Mazziane: https://theses.hal.science/tel-04681458/file/2024COAZ4014.pdf (Probabilistic analysis for caching)
- PhD Salem: https://dl.acm.org/doi/10.1145/3579342.3579348 (Online Learning for Network Resource Allocation)
- wiki linear programming: https://en.wikipedia.org/wiki/Linear_programming


# fast OPGD for fractional caching 

Throughput and latency are important considerations for caching algorithms. A caching algorithm can be impractical when incoming request arrive faster then the throughput so that the waiting time for handling each request grows unboundedly or when the latency introduced by computing and executing the dynamical cache takes longer then having no cache.   

## steps of OPGD for fractional caching

OPGD for fractional caching consists out of following steps at time $t$: 

- step -1: request for item $l_{t}$ (part of the definition of the caching problem, we don't touch this)  <br>
- step 0: get the fraction of the cached $l_{t}$, and receive the rest (performance depends on how the fraction is stored)  <br>
- step 1: the gradient update $+ \eta e^{l_{t}}$, we call this the gradient update (can be achieved in $O(1)$ time)  <br>
- step 2: projecting on $Y$, we call this the cache resizing step (changes at best all nonzero components of $y_{t}$ but is very symmetric) <br>
- step 3: updating the cache physically from $y_{t}$ to $y_{t+1}$(adds fraction to $l_{t}$ which we have from step 0, and remove fractions from all others)



## OPGD: making the projection fast

One way to think about this projection is to consider the KKT conditions for the corresponding optimization problem but we think this is overkill.

We start by assuming that by initializing the caching strategy in $y_{0} \in Y_{full}$ all projections in later iterations on $Y$ are equal to the projections on $Y_{full}$ or we get a better online caching strategy. (we haven't figured out a simple formal argument for this but to us it visually obvious, $Y$ for $C>1$ is the intersection of a box and a half plane, a n-dimensional box with a cut corner) 

Observe that $Y_{full}$ is an intersection of planes and  box constraints:

$$
Y_{full} = [0,\infty[^{n}  \cap  ] \infty,1 ]^{n} \cap  \left\{y \in \mathbb{R}^n \middle| \sum_{n=1}^{N} y^{n} = C \right\} .
$$ 

Consider the method of alternating projections for these $3$ convex sets, projection on $[0,1]^{n}$ is clipping all the components between $0$ and $1$ and projecting on the plane is subtracting the normal vector with a magnitude the distance in this case it is the $1$ vector (normalized) with magnitude after step 1: 

$$
m_{t} = \sum_{n=1}^{N} y_{t}^{n} + \eta   - C = \eta. 
$$ 

Naive alternating methods of projections would converge geometrically fast but in this case we can improve it to exact in constant amortized amount of iterations through following observations:


- after projecting first on $] \infty,1 ]^{n}$ all following projections stay in $] \infty,1 ]^{n}$ meaning that we only have to project once on $] \infty,1 ]^{n}$, this property follows by the fact that the other projections only reduce the components or maximally brings a component to $0$  

- in the first projection on $] \infty,1 ]^{n}$, you only have to check the component that is updated in the gradient update step, so we merge this projection with the gradient update step ($O(1)$ time )

- after you ignore projecting on $] \infty,1 ]^{n}$, any components that are projected to $0$ should be $0$, this follows from the fact that the minimum of a component in $Y_{full}$ is $0$ and after being projected on $0$ it can only be reduced or maximally brought to $0$ so we found a matching lower and upper bound giving the equality for $0$ 

- maximum $N$ components can be $0$ because of gradient update step can make $0$ components non-zero  which is at most one component. Consequently, amortized over all iterations, the algorithm can set at most one component to zero on average.

Assuming, for contradiction, that on average $1 + \varepsilon$  components are set to zero per iteration, the net change in the number of zero components would be at least $\varepsilon$   per step. Over $1 + \frac{N}{\varepsilon}$ iterations, this would result in an increase of more than $N$ zero components. However, this exceeds the initial maximum $N$ zero components, leading to a contradiction. Thus, the average number of components set to zero per iteration cannot exceed one.

- you can avoid redoing the same $0$ component projections by doing projections on the plane in the subspace where the $0$ components are $0$, here the normal vector is also the $1$ vector but has $0$ at the $0$ components, you can still subtract by the full $1$ vector if you treat all negative components as $0$ later 

- the projections on the plane (subtractions of the normal vector) can be implemented lazily, i.e. never execute the subtraction until you need a component so at the time of the projection you only have to update how much you should subtract by the normal vector later, a more geometric way of thinking is that instead of updating all the components of a point in n-dimensions, you change the coordinate system by moving the origin in a straight line of the normal vector and you keep track how far it went ($O(1)$ time at projection, $O(1)$ time per subtraction when accessing a component) 


- detection of negative components can be done by maintaining on ordered data structure of the components of $y$ as in all steps up until now maximum $O(1)$ components change, the maintenance cost for this is $O(log(N))$ note that lazy subtraction doesn't effect the order of the components  ($O(log(N))$ look up time )


So all these observations combined means that we on average only have to do $4$ projections to get the exact result and it at most cost us $O(log(N))$ time.

## Limitation of fractional caching

Implementing Steps 0 and 3 efficiently simultaneously seems difficult. For step 0, the fractions of an item should be stored together to enable quick access. Conversely, Step 3 requires storing different fractions of items together to allow for fast deletion.

We haven't put in time to try to fix this issue, we expect this needs requires more detailed information on how fractional caching actually can be implemented.


# fast OPGD for integral caching 

By using randomized rounding schemes fractional caching algorithms can be made integral. So now we have to keep track of $y_{t}$ and the current integral cache $X_{t}$ with $y_{t} = E[X_{t}]$ this assures that: 

\begin{align*}
\text{Regret}_T &= \sum_{t=1}^T \langle g_t, y_t \rangle - \min_{y \in Y} \sum_{t=1}^T \langle g_t, y \rangle, \\
&= \sum_{t=1}^T \langle g_t, E[X_{t}] \rangle- \min_{y \in Y} \sum_{t=1}^T \langle g_t, y \rangle \\
&= E\left[\sum_{t=1}^T \langle g_t, X_{t} \rangle\right]- \min_{y \in Y} \sum_{t=1}^T \langle g_t, y \rangle \\
\end{align*}

Sublinear regret implies convergence in expectation of the hitrate:

$$
\lim_{T \to \infty} E[H_{T}(X_{t})] = H_{T}(y_{OPT}) 
.
$$ 


## convergence of randomized rounding

Convergence in expectation doesn't imply convergence in probability, convergence in probability for bounded random variables is equivalent to convergence in norm or MSE because the hitrate already converges in expectation we only require convergence in variance:

\begin{align*}
\lim_{T \to \infty} \text{Var}\left[ H_{T}(X_{t})  \right]&=0  \Leftrightarrow \\
\lim_{T \to \infty} \frac{1}{T^{2}}\text{Var}\left[\sum_{t=1}^T \langle g_t, X_{t} \rangle\right] &=0 \Leftrightarrow \\
\lim_{T \to \infty} \frac{1}{T^{2}}\sum_{t_{1},t_{2}=1}^T\text{Cov}\left[ \langle g_{t_{1}}, X_{t_{1}} \rangle, \langle g_{t_{2}}, X_{t_{2}} \rangle\right] &=0 
\end{align*}

## independent rounding

Given a fractional cache $y$ independent rounding is:

$$
\text{IR}[y^{n}] = 1 \text{ with probability } y^{n} \text{ else } 0 \quad  \forall  n= 1 ,..., N
$$ 
with $\text{IR}[y^{n}]$ forall $n$ independent of each other and independent from everything else. This rounding scheme has convergence to the optimal hitrate as $y_{t}^{n} = E[\text{IR}[y_{t}^{n}]]$  and

\begin{align*}
\lim_{T \to \infty} \frac{1}{T^{2}}\text{Var}\left[\sum_{t=1}^T \langle g_t, \text{IR}[y_{t}^{n}] \rangle\right] &= \lim_{T \to \infty} \frac{1}{T^{2}}\sum_{t=1}^T \text{Var}\left[ \langle g_t, \text{IR}[y_{t}^{n}] \rangle\right] \\
&\le \lim_{T \to \infty} \frac{1}{T^{2}}\sum_{t=1}^T \frac{1}{4}   \\
&= \lim_{T \to \infty} \frac{1}{4T}\\ 
&= 0  \\
\end{align*}

here we used the fact that $\langle - g_t, \text{IR}[y_{t}^{n}] \rangle \in [0,1]$ so its variance can be bounded using Popo's inequality and independence to interchange variance and sum.


To understand the practical challenges of independent sampling, consider a scenario with a round-robin request pattern involving $N = 10^6$ items and a cache capacity of $C = 10^3$. Let $T$ be a multiple of $N$. Then the optimal fractional caching solution is $y^{n}_{OPT} = \frac{C}{N}$  for all $n \in \{1, \ldots, N\}$. However, applying independent rounding ( $\text{IR}[y^{n}_{\text{OPT}}]$ ) to this fractional solution introduces two critical issues:  

1. **Excessive Cache Turnover**: On average, 999 items in the cache would needed to be replaced with each single request this means we have to fetch at least 998 on average which in most cases impractical.  
2. **Capacity Violation**: The cache capacity $C$ is not strictly maintained due to the probabilistic nature of independent rounding, which risks exceeding or undershooting the designated limit.  

This illustrates the impracticality of independent rounding for maintaining stable, capacity-respecting caches in such settings.


## coupled rounding

To avoid the rapid change in the cache from independent rounding, you can do coupled rounding. The idea behind coupled rounding is to keep the sampling decision stable by using the same sampled uniform variable for deciding that a item should be cached or not.

$$
\text{CR}[y^{n}_{t}] = 1 \text{ if } y_{t}^{n} <U_{n}  \text{ else } 0 \quad  \forall  n= 1 ,..., N
$$ 
where $U_{n}$ are independent uniforms from $[0,1]$. 

Coupled rounding fixes the fetching problem completely as a OPGD step only $1$ fractional item gets updated positively the requested item so only that item can potentially go from uncached to cached and that items should be fetched anyway. 

The hitrate of coupled rounding converges in expectance because $y^{n}_{t} = E[\text{CR}[y^{n}_{t}]] $ but the hitrate doesn't converge in MSE. Consider the following example, $2$ items arriving round-robin (so periodic with period $2$) with cache size $1$ and initialize $y_{0} = y_{OPT}$ then $\forall \eta<0.5, \forall t \in \mathbb{N}:y_{t} =y_{t+2} \implies \text{CR}[y_{t}] = \text{CR}[y_{t+2}] \implies \langle g_{t}, \text{CR}[y_{t}]  \rangle = \langle g_{t +2}, \text{CR}[y_{t+2}]  \rangle $ if the initial uniforms a sampled badly when $ 0.5> U_{1}> 0.5- \eta$ and $0.5 +\eta > U_{2} > 0.5$ for example which happens with non-zero probability $\forall \eta>0$ then  
$\forall T \in \mathbb{N}: H_{T}(X_{t}) = 0$ which is way lower then $H_{2T}(y_{OPT})=0.5$ there is also a probability that both uniforms all low and nothing ever enters the cache. 

The capacity violation is also still present. 

Although coupled rounding doesn't converge in MSE, convergence in expectance is still useful because for every possible sampled hitrate lower then the average something higher then the average must be there to balance it. In the previous example if we were lucky we could have sampled a hitrate of $1$.

## fast coupled rounding

As far as we discussed OPGD for fractional caching with coupled rounding has only the capacity violation issue which isn't that bad for big caches. A naive implementation of coupled rounding would require comparing all the non-zero and non-one components of $y_{t}^{n}$ against the uniforms requiring at worst $O(N)$ comparisons or time.

Make the follow observation: at most $1$ item can enter the cache per step so at most $1$ on average can leave it. We have to check the following $\forall n$:
\begin{align*}
y_{t}^{n} < U_{n} \Leftrightarrow
y_{t}^{n}-U_{n} <0  
\end{align*}
as $U_{n}$ stays constant it reduces to the same problem as how we checked for $0$ components of $y_{t}^{n}$ which we can do in $O(log(N))$ time.

## fixing coupled rounding

Intuitively there is not enough (independent) randomness present to obtain convergence in coupled rounding. We believe that in practical scenarios the variance is very small and conjecture that if instead of the $X_{t}$ being independent the $g_{t}$ are independent, or even weaker that the dependence decays exponentially in time we still obtain convergence in MSE.

The convergence issue in coupled rounding should be addressed by somehow introducing independent randomness. One direct approach is to update/resample the uniform random variable associated with the requested item at each step of OPGD or using an MCMC update to limit the change in the uniform variable we guess to the magnitude of the step size. We expect these proposals to guarantee convergence in MSE.

Alternatively, an indirect approach involves subsampling requests: each OPGD step is independently skipped with a fixed probability. This reduces computational cost. Subsampling is justified because it preserves the asymptotic hit rate (in MSE) of the optimal static caching strategy in hindsight, this a consequence of Monte Carlo summation convergence. We think this is insufficient for proving convergence of the coupled rounding but some other random modifications to the request stream might achieve convergence.



## Sources fast OPGD for caching

- wiki projection on convex set: https://en.wikipedia.org/wiki/Projections_onto_convex_sets
- paper Duchi: http://portal.acm.org/citation.cfm?doid=1390156.1390191 (also uses lazy updates to project on the simplex)
- wiki popo's inequality: https://en.wikipedia.org/wiki/Popoviciu%27s_inequality_on_variances
- paper Carra: http://arxiv.org/abs/2405.01263 (introduces efficient implementations for fractional and integral caching)
- paper Salem: http://arxiv.org/abs/2101.12588 (departs from fractional caching to integral caching)


# Quantized Online Caching Descent (qOCD)

To compete with the throughput of established caching algorithms like LRU, online learning-based (OLO) caching approaches must overcome limitations in computational and memory overhead. We introduce quantized Online Caching Descent (qOCD), an improvement in efficiency of Online Projected Gradient Descent (OPGD) through full quantization of all operations. For specific step sizes and a soft constraint on the cache size, this quantization can be performed exactly, without approximation error and runs steps in constant time complexity.


## quantizing cache resizing

We quantize the cache resizing step by imposing smallest measure of change $\varepsilon$ we chose $\varepsilon$ such that $0$ and $1$ can exactly be quantized such that projections on $[0,1]^{N}$ can be done exactly. 

The cache resizing step will be implemented as a lazy subtraction, so all we do is keep count of the amount of $\varepsilon$ of lazy subtractions we have done.  The quantization makes exactly resizing the cache impossible for all possible cache sizes instead we chose to resize to the smallest quantized cache size bigger then the exact cache size, this smallest quantized cache size changes every step but is bounded. 

The projection is implemented by repeatedly doing the smallest cache resize step until the smallest quantized cache size bigger then the exact cache size is reached, no $0$ projections have to happen because negative components have to cross $0$ and can be detected each step. In the implementation everything will be implemented with integers and the rescaling with $\varepsilon$ will be implicit. Here is the julia code associated with the cache resizing step:

```julia
function resize_cache!(q::quant_OGD)
    while q.overhead >= q.nonzeros
        q.lazy_update += 1
        q.overhead -= q.nonzeros
        q.nonzeros -= get(q.counter_val, q.lazy_update, 0) 
    end
end
```

The overhead is how much the total fractional cache is over the cache limit which is smaller and simpler to store and track. We never need to know which components became $0$ only how many (this decides the magnitude of the normal vector) as we treat all negative components (which are positive but negative after the lazy subtraction) as $0$ later. To look up how many components are $0$ we maintain a counter which counts the values of the components + the lazy_update, the lookup only costs $O(1)$ time. 


**TODO fix this** (the amount of nonzeros changes every time and we don't have a simple argument)

> **Lemma: Upper Bound on the overhead before and after Cache Resizing in qOCD** <br>
> After cache resizing the overhead is bounded by $\varepsilon A$ and before $\varepsilon A + \eta$.  Here, $\varepsilon$ represents the smallest quantization unit, and $\eta$ denotes the step size and $A$ the amount of nonzero components. 


> **Lemma: Upper Bound on the Expected Number of Cache Resizing Steps in qOCD** <br>
> The expected number of cache resizing steps is bounded above by $2 +  \frac{E[\eta]}{\varepsilon C} $.
> Here, $\varepsilon$ represents the smallest quantization unit, and $\eta$ denotes the step size.  For the specific choice of step size $E[\eta] = \sqrt{\frac{C(1- \frac{C}{N})}{T}}$, this bound simplifies to $2 + \frac{1}{\varepsilon \sqrt{C T}}  $. (Note to self I added $1$ to deal with ceil.)


## full qOCD step 

The full qOCD step isn't  more complicated as normal OPGD for caching. We have to maintain track of the amount of non-zeros, a counter of values and the overhead. We combine the implementation of the gradient update and projection on $1$. Maybe it is easier to read the code: 

```julia
function step!(q::quant_OGD, i::Int)
    q.counter_val[q.counter[i]] -= 1
    if q.counter[i] > q.lazy_update # is component nonzero?
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
```
Here's a list outlining the variables used in the `step!` function:

*   **`i`**: The index of the requested item.
*   **`q.counter`**: A dictionary mapping item index to the associated number of fractional cache units as multiples of $\varepsilon$ + q.lazy_updates .
*   **`q.counter_val`**: A dictionary that stores the counts of each counter value
*   **`q.step_size`**: The step size but also the multiple of $\varepsilon$  
*   **`q.ONE`**: The amount of $\varepsilon$ that fits in $1$ cache unit

## resetting the lazy update  

In the implementation we presented lazy_update and counter will grow unboundedly therefore also counter_val will have an ever increasing amounts of keys.  counter[i] $\le$ ONE + lazy_update, lazy_update at leasts subtracts $C \varepsilon$ cache which comes from the $\eta$ cache each steps, using the typical step size our we derived that if 
$$
T \le C \varepsilon^{2} I^{2}
$$
no int overflow will happen with $I$ the biggest representable integer, in our experiments we don't encounter integer overflows with $UInt16$ with reasonable $\varepsilon$.

TODO: write a derivation of this bound, we conjecture that the amortized cost of resetting the lazy update is $O(1)$ but requires some smart analysis with the amount on nonzeros that changes, we haven't found an elegant solution to spread this reset over steps without switching in and out a new counter. 

## regret bounds for qOCD

The regret bound from classic OPGD doesn't hold because we aren't projecting on the same set each time but the proof stays the same see A Modern Introduction to Online Learning. The intuition is that there is a set in all the sets that we are projecting on that i.e. our online strategy always had more freedom to choose a better fractional cache configuration then the fractional cache configuration with less cache. 

Note that these regret bounds only work for quantized step sizes, we would also but the proof of the bound on expected regret for independent rounding of the step size here.

TODO: write a out the proof

## convergence of qOCD 

As $\varepsilon$ our smallest possible step size we will not have convergence for $T\rightarrow \infty$ but for $\varepsilon = \text{eps}(UInt32)$ and our typical choice of step size the $T \approx 10^{20}$ still works fine assuming $ \frac{C}{N}< \frac{1}{2}$.

TODO: proof the following conjecture

We conjecture that qOCD can converge with fixed quantization by using independent rounding on the step sizes this would probably achieve $O(\sqrt{T})$ regret bound in expectance and $O(T^{\frac{3}{4}})$ regret bound in MSE against the fractional caching algorithm with $C$ cache. The proof we have in mind is moving the independent rounding from the step size to the gradients and using regrets guarantees and then doing a straight forward calculation of how big the variance from independent rounding based on $T$. This independent rounding would be equivalent to randomly not executing steps bringing the computational cost per step on average to $O(\frac{1}{\sqrt{T}})$.


## limitations of qOCD

Similar for the non quantized version to implement the qOCD we require to achieve step 0 and step 3 simultaneously. 

## integral version of qOCD

By using similar quantization techniques a version of qOCD for the integral caching can be implemented that runs per step on average in $O(1)$ time, the version we implemented does cache resizing until the integral cache constraint is satisfied making it have constant cache but making proofing regret guarantees harder. 

## extending to caching with variable sizes

OPGD stay almost the same for caching with variable sizes the definition of the hitrate changes, the gradient is bigger for bigger items but stays sparse, in the fast projection step the normal vector changes and keeping track of its magnitude is different then counting non-zeros, detection of negative components requires separate ordered data structures increasing the cost to $O(l \log\left(N\right))$ where $l$ is the amount of different sizes.  

All different sizes must be able to be quantized together and if the step size is divisible by all sizes all negative components still pass $0$ preserving the inner workings of qOCD. An approximate algorithm can be made by fixing the allowed sizes as the divisors of $60$ times $\varepsilon$ and using the smallest sizes bigger then the real sizes, do independent rounding for the gradients and resize based on the real cache size.

## Sources for qOCD

- Gemini 2 Flash Thinking help with writing
- Orabona: http://arxiv.org/abs/1912.13213 (intro to parameter free learning, this was where we read the first proof of the regret bounds of OLO)
- unbiased gradients OLO: we are convinced someone already proved that running OPGD with unbiased gradients also works

# TODO
- benchmark some traces
- asynronous/parallel implementation
- fixing issues of the underlying algorithm 