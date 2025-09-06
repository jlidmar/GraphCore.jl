# Generators.jl

function empty_graph(n::Integer; directed::Bool=false)
    # Create empty edge list and n vertices
    return build_graph(CoreGraph, Tuple{Int,Int}[];
                      directed=directed, n=n)
end

function complete_graph(n::Integer; directed::Bool=false)
    if directed
        edges = [(u, v) for u in 1:n for v in 1:n if u != v]
    else
        edges = [(u, v) for u in 1:n for v in u+1:n]
    end
    return build_core_graph(edges; directed=directed)
end

function path_graph(n::Integer; directed::Bool=false)
    edges = [(i, i+1) for i in 1:n-1]
    return build_core_graph(edges; directed=directed)
end

function cycle_graph(n::Integer; directed::Bool=false)
    edges = [(i, i+1) for i in 1:n-1]
    push!(edges, (n, 1))
    return build_core_graph(edges; directed=directed)
end

function star_graph(n::Integer; directed::Bool=false)
    edges = [(1, i) for i in 2:n]
    return build_core_graph(edges; directed=directed)
end

function grid_graph(m::Integer, n::Integer; directed::Bool=false)
    # Create proper grid edges (adjacent cells)
    edges = Tuple{Int,Int}[]

    # Add horizontal edges
    for i in 1:m
        for j in 1:n-1
            u = (i-1)*n + j
            v = (i-1)*n + j + 1
            push!(edges, (u, v))
        end
    end

    # Add vertical edges
    for i in 1:m-1
        for j in 1:n
            u = (i-1)*n + j
            v = i*n + j
            push!(edges, (u, v))
        end
    end

    return build_core_graph(edges; directed=directed)
end

function random_graph(n::Integer, m::Integer; directed::Bool=false)
    edges = [(rand(1:n), rand(1:n)) for _ in 1:m]
    # Filter out self-loops
    edges = [(u, v) for (u, v) in edges if u != v]
    return build_core_graph(edges; directed=directed)
end

function erdos_renyi_graph(n::Integer, p::Float64; directed::Bool=false)
    edges = Tuple{Int,Int}[]
    if directed
        for i in 1:n, j in 1:n
            if i != j && rand() < p
                push!(edges, (i, j))
            end
        end
    else
        for i in 1:n, j in i+1:n
            if rand() < p
                push!(edges, (i, j))
            end
        end
    end
    return build_core_graph(edges; directed=directed)
end

function barabasi_albert_graph(n::Integer, m::Integer; directed::Bool=false)
    edges = Tuple{Int,Int}[]
    for i in 2:n
        targets = rand(1:i-1, min(m, i-1))
        for t in targets
            push!(edges, (t, i))
        end
    end
    return build_core_graph(edges; directed=directed)
end

function wheel_graph(n::Integer; directed::Bool=false)
    edges = [(1, i) for i in 2:n]
    for i in 2:n-1
        push!(edges, (i, i+1))
    end
    push!(edges, (n, 2))
    return build_core_graph(edges; directed=directed)
end

function lattice_graph(dims::Integer...; periodic::Bool=true, directed::Bool=false)
    N = length(dims)
    edges = Tuple{Int, Int}[]
    for idx in CartesianIndices(dims)
        for dim in 1:N
            if idx[dim] < dims[dim]
                neighbor = CartesianIndex(ntuple(i -> i == dim ? idx[i] + 1 : idx[i], N))
                push!(edges, (LinearIndices(dims)[idx], LinearIndices(dims)[neighbor]))
            elseif periodic
                neighbor = CartesianIndex(ntuple(i -> i == dim ? 1 : idx[i], N))
                push!(edges, (LinearIndices(dims)[idx], LinearIndices(dims)[neighbor]))
            end
        end
    end
    return build_core_graph(edges; directed=directed)
end