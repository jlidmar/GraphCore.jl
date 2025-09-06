module Conversions

using ..GraphCore
using ..GraphCore: to_core_graph, to_weighted_graph, to_property_graph
using ..GraphCore: to_adj_graph, to_weighted_adj_graph

# Conversion between graph types
export to_core_graph, to_weighted_graph, to_property_graph
export to_adj_graph, to_weighted_adj_graph

# External library conversions
export from_graphs_jl, from_weighted_graphs_jl, to_graphs_jl, to_weighted_graphs_jl
export from_adjacency_matrix, to_adjacency_matrix

using ..GraphCore: GraphCore

# For conversions to/from Graphs.jl
using Graphs: Graphs, AbstractGraph, SimpleGraph, SimpleDiGraph
using SimpleWeightedGraphs: SimpleWeightedGraphs, SimpleWeightedGraph, SimpleWeightedDiGraph

using SparseArrays: sparse, findnz # For adjacency matrix conversions
using LinearAlgebra: issymmetric

# ==============================================================================
# GRAPH CONSTRUCTION AND CONVERSION HELPERS
# ==============================================================================

"""
    from_graphs_jl(g::Graphs.AbstractGraph; directed::Bool = Graphs.is_directed(g)) -> CoreGraph

Convert a graph from the Graphs.jl ecosystem to a CoreGraph.
"""
function from_graphs_jl(g::Graphs.AbstractGraph; directed::Bool = Graphs.is_directed(g))
    n = Graphs.nv(g)
    edge_list = [(Graphs.src(e), Graphs.dst(e)) for e in Graphs.edges(g)]
    return build_core_graph(edge_list; directed=directed, n=n)
end

"""
    from_weighted_graphs_jl(g::Graphs.AbstractGraph, weights::AbstractVector{W};
                           directed::Bool = Graphs.is_directed(g)) -> WeightedGraph{W}

Convert a weighted graph from the Graphs.jl ecosystem to a WeightedGraph.
"""
function from_weighted_graphs_jl(g::Graphs.AbstractGraph, weights::AbstractVector{W};
                                directed::Bool = Graphs.is_directed(g)) where W
    n = Graphs.nv(g)
    edge_list = [(Graphs.src(e), Graphs.dst(e)) for e in Graphs.edges(g)]
    return build_weighted_graph(edge_list, weights; directed=directed, n=n)
end

"""
    to_graphs_jl(g::GraphInterface) -> Graphs.SimpleGraph or Graphs.SimpleDiGraph

Convert a property graph to a Graphs.jl graph (losing properties and weights).
Returns SimpleGraph for undirected graphs, SimpleDiGraph for directed graphs.
"""
function to_graphs_jl(g::GraphInterface)
    if is_directed_graph(g)
        result = Graphs.SimpleDiGraph(num_vertices(g))
    else
        result = Graphs.SimpleGraph(num_vertices(g))
    end

    for (u, v) in GraphCore.edges(g)
        Graphs.add_edge!(result, u, v)
    end

    return result
end

"""
    to_weighted_graphs_jl(g::GraphInterface{V,E,W}) where {V,E,W} ->
        SimpleWeightedGraphs.SimpleWeightedGraph or SimpleWeightedGraphs.SimpleWeightedDiGraph

Convert a WeightedGraphInterface graph to a weighted SimpleWeightedGraphs.jl graph (preserving weights, losing other properties).
Requires SimpleWeightedGraphs.jl package.
"""
function to_weighted_graphs_jl(g::WeightedGraphInterface{W}) where {W<:Number}
    nv = num_vertices(g)
    if is_directed_graph(g)
        result = SimpleWeightedGraphs.SimpleWeightedDiGraph{Int32, W}(nv)
    else
        result = SimpleWeightedGraphs.SimpleWeightedGraph{Int32, W}(nv)
    end

    for (i, (u, v)) in enumerate(GraphCore.edges(g))
        w = edge_weight(g, i)
        SimpleWeightedGraphs.add_edge!(result, u, v, w)
    end
    return result
end

# Matrix-based conversions:

"""
    from_adjacency_matrix(::Type{G}, adj_matrix::AbstractMatrix{W}) where {G<:GraphInterface,W} -> G

Construct a graph of appropriate type from an adjacency matrix.
- Non-zero entries in adj_matrix become edges with those weights
- If `directed` is not specified, it is inferred from the symmetry of adj_matrix

For undirected graphs, adj_matrix should be symmetric.
"""
function from_adjacency_matrix(::Type{G}, adj_matrix::AbstractMatrix{W}; directed=nothing) where {G<:GraphInterface,W<:Number}
    sym = issymmetric(adj_matrix)

    if isnothing(directed) # If directed is not specified, infer from symmetry of adj_matrix
        directed = !sym
    end

    if !sym && !directed
        throw(ArgumentError("Adjacency matrix must be symmetric for undirected graphs"))
    end

    edges = Tuple{Int,Int}[]
    weights = W[]

    for (i, j, w) in zip(findnz(adj_matrix)...)
        if w != 0
            if directed || sym && i < j
                push!(edges, (i, j))
                push!(weights, w)
            end
        end
    end

    if G <: WeightedGraph || G <: WeightedAdjGraph
        return build_graph(G, edges; weights=weights, directed=directed)
    else
        return build_graph(G, edges; directed=directed)
    end
end

"""
    to_adjacency_matrix(g::GraphInterface) -> SparseMatrixCSC{W} where {W}

Convert a graph to its adjacency matrix representation.
- For directed graphs, the matrix is not required to be symmetric.
- For undirected graphs, the matrix will be symmetric if the weights are.
- For unweighted graphs, entries are 1 where edges exist.
- For weighted graphs, entries are the edge weights
  (which are not necessarily symmetric even for undirected graphs).
"""
function to_adjacency_matrix(g::GraphInterface)
    u_, v_ = Int32[], Int32[]
    for u in vertices(g), v in neighbor_indices(g, u)
        push!(u_, u)
        push!(v_, v)
    end
    return sparse(u_, v_, ones(Int32, length(u_)))
end

function to_adjacency_matrix(g::WeightedGraphInterface{W}) where W
    u_, v_, w_ = Int32[], Int32[], W[]
    for u in vertices(g), (v, w) in neighbor_weights(g, u)
        push!(u_, u)
        push!(v_, v)
        push!(w_, w)
    end
    return sparse(u_, v_, w_)
end

end # module Conversions