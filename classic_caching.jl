# written by an deepseek r1 (a LLM)
using DataStructures

mutable struct FIFOCache
    capacity::Int
    cache::Set{Int}
    queue::Deque{Int}

    function FIFOCache(capacity::Int)
        new(capacity, Set{Int}(), Deque{Int}())
    end
end

function process_request(fifo::FIFOCache, key::Int)::Bool
    if key in fifo.cache
        return true  # Hit
    end
    # Miss
    if length(fifo.cache) >= fifo.capacity
        evict_key = popfirst!(fifo.queue)
        delete!(fifo.cache, evict_key)
    end
    push!(fifo.cache, key)
    push!(fifo.queue, key)
    return false  # Miss
end

mutable struct LRUCache
    capacity::Int
    cache::OrderedDict{Int, Bool}

    function LRUCache(capacity::Int)
        new(capacity, OrderedDict{Int, Bool}())
    end
end

function process_request(lru::LRUCache, key::Int)::Bool
    if haskey(lru.cache, key)
        # Move to end
        delete!(lru.cache, key)
        lru.cache[key] = true
        return true  # Hit
    end
    # Miss
    if length(lru.cache) >= lru.capacity
        popfirst!(lru.cache)  # Remove oldest
    end
    lru.cache[key] = true
    return false  # Miss
end

mutable struct LFUCache
    capacity::Int
    min_freq::Int
    key_freq::DefaultDict{Int, Int}
    freq_keys::DefaultDict{Int, OrderedDict{Int, Nothing}}

    function LFUCache(capacity::Int)
        key_freq = DefaultDict{Int, Int}(0)
        freq_keys = DefaultDict{Int, OrderedDict{Int, Nothing}}(OrderedDict{Int, Nothing})
        new(capacity, 0, key_freq, freq_keys)
    end
end

function process_request(lfu::LFUCache, key::Int)::Bool
    if haskey(lfu.key_freq, key)
        # Update frequency
        freq = lfu.key_freq[key]
        delete!(lfu.freq_keys[freq], key)
        if isempty(lfu.freq_keys[freq])
            delete!(lfu.freq_keys, freq)
            if lfu.min_freq == freq
                lfu.min_freq += 1
            end
        end
        new_freq = freq + 1
        lfu.key_freq[key] = new_freq
        lfu.freq_keys[new_freq][key] = nothing
        return true  # Hit
    end
    # Miss
    if length(lfu.key_freq) >= lfu.capacity
        # Find the first non-empty frequency starting from min_freq
        while isempty(lfu.freq_keys[lfu.min_freq])
            lfu.min_freq += 1
        end
        evict_dict = lfu.freq_keys[lfu.min_freq]
        evict_key = first(keys(evict_dict))
        delete!(evict_dict, evict_key)
        if isempty(evict_dict)
            delete!(lfu.freq_keys, lfu.min_freq)
        end
        delete!(lfu.key_freq, evict_key)
    end
    # Add new key
    lfu.key_freq[key] = 1
    lfu.freq_keys[1][key] = nothing
    lfu.min_freq = 1
    return false  # Miss
end

mutable struct OptimalCache
    capacity::Int
    cache::Set{Int}
    trace::Vector{Int}
    key_positions::Dict{Int, Vector{Int}}

    function OptimalCache(capacity::Int, trace::Vector{Int})
        key_positions = Dict{Int, Vector{Int}}()
        for (idx, key) in enumerate(trace)
            if haskey(key_positions, key)
                push!(key_positions[key], idx)
            else
                key_positions[key] = [idx]
            end
        end
        new(capacity, Set{Int}(), trace, key_positions)
    end
end

function get_next_use(optimal::OptimalCache, key::Int, current_pos::Int)
    positions = get(optimal.key_positions, key, Int[])
    idx = searchsortedfirst(positions, current_pos + 1)
    return idx <= length(positions) ? positions[idx] : Inf
end

function process_request(optimal::OptimalCache, current_pos::Int)::Bool
    key = optimal.trace[current_pos]
    if key in optimal.cache
        return true  # Hit
    end
    # Miss
    if length(optimal.cache) >= optimal.capacity
        # Find key in cache with furthest next use
        furthest = -Inf
        evict_key = nothing
        for candidate in optimal.cache
            next_use = get_next_use(optimal, candidate, current_pos)
            if next_use > furthest || (next_use == furthest && candidate < evict_key)
                furthest = next_use
                evict_key = candidate
            end
        end
        if evict_key !== nothing
            delete!(optimal.cache, evict_key)
        end
    end
    push!(optimal.cache, key)
    return false  # Miss
end

# Test the caches
trace = repeat([1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5], 4)
capacity = 3

caches = [
    ("FIFO", FIFOCache),
    ("LRU", LRUCache),
    ("LFU", LFUCache),
    ("Optimal", OptimalCache)
]

for (name, CacheType) in caches
    if name == "Optimal"
        cache = CacheType(capacity, trace)
        misses = 0
        for i in 1:length(trace)
            if !process_request(cache, i)
                misses += 1
            end
        end
    else
        cache = CacheType(capacity)
        misses = 0
        for key in trace
            if !process_request(cache, key)
                misses += 1
            end
        end
    end
    total = length(trace)
    println("$name Misses: $misses, Miss rate: $(round(100 * misses / total, digits=2))%")
end