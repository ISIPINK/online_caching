mutable struct TwoWayCounter
    forward::Dict{Int, Int}
    backward::Dict{Int, Vector{Int}}
    TwoWayCounter() = new(Dict{Int, Int}(), Dict{Int, Vector{Int}}())
end

function add!(counter::TwoWayCounter, index::Int, count::Int)
    if haskey(counter.forward, index)
        old_count = counter.forward[index]
        # removing things is slow
        if haskey(counter.backward, old_count)
            # Remove the index from the old count's list
            filter!(x -> x != index, counter.backward[old_count])
            if isempty(counter.backward[old_count])
                delete!(counter.backward, old_count)
            end
        end
        # Update the count in forward
        counter.forward[index] += count
    else
        # Add new index to forward
        counter.forward[index] = count
    end

    new_count = counter.forward[index]

    # pushing things is also slow
    if haskey(counter.backward, new_count)
        push!(counter.backward[new_count], index)
    else
        counter.backward[new_count] = [index]
    end
end

function get_indexes(counter::TwoWayCounter, count::Int)
    return get(counter.backward, count, Int[])
end

# removing things is slow
function remove!(counter::TwoWayCounter, index::Int)
    if haskey(counter.forward, index)
        count = counter.forward[index]
        delete!(counter.forward, index)
        if haskey(counter.backward, count)
            # Remove the index from the count's list
            filter!(x -> x != index, counter.backward[count])
            if isempty(counter.backward[count])
                delete!(counter.backward, count)
            end
        end
    end
end

# Example usage:
function test_2counter()
counter = TwoWayCounter()
add!(counter, 0, 5)
add!(counter, 1, 3)
add!(counter, 4, 5)
println(get_indexes(counter, 5))  # Output: [0, 4]

add!(counter, 0, 7)
println(get_indexes(counter, 5))  # Output: [4]
println(get_indexes(counter, 7))  # Output: []
println(get_indexes(counter, 12)) # Output: [0]
end