# AdjGraph.jl

# Copyright (c) 2025 Jack Lidmar
# All rights reserved.

# This software is licensed under the MIT License. See the LICENSE file for details.

# SPDX-FileCopyrightText: 2025 Jack Lidmar <jlidmar@kth.se>
# SPDX-License-Identifier: MIT

"""
Adjacency List Graph Structures
===============================

Dynamic graph implementations using Vector{Vector{Int32}} for neighbor storage.
Optimized for frequent structural modifications at the cost of some memory overhead.

## Design Rationale

**Adjacency List Benefits**:
- **Dynamic Structure**: O(1) edge/vertex additions, efficient removals
- **Memory Flexibility**: Grows and shrinks with graph structure
- **Cache Locality**: Good performance for sparse graphs
- **Mutation Friendly**: No expensive reconstruction on modifications

**Trade-offs**:
- **Memory Overhead**: Vector storage + pointers vs. flat CSR arrays
- **Index Stability**: Edge indices may be invalidated by removals
- **Construction Cost**: Less efficient than CSR for bulk construction
- **Memory Fragmentation**: Potential issues with many small vectors

## Storage Layout

```
AdjGraph{false} (undirected):
├── neighbors: [[2,3], [1,3], [1,2]]           # Per-vertex neighbor lists
├── neighbor_to_edge: [[1,3], [1,2], [3,2]]    # Maps to undirected edge indices
└── num_edges: 3                                # Undirected edge count

AdjGraph{true} (directed):
├── neighbors: [[2], [3], []]                   # Per-vertex out-neighbor lists
├── neighbor_to_edge: [[], [], []]              # Empty (not needed)
└── num_edges: 2                                # Directed edge count
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Neighbor access | O(1) | Direct vector access |
| Edge existence | O(degree) | Linear search in neighbor vector |
| Add vertex | O(1) | Push to vectors |
| Add edge | O(1) amortized | Vector push operations |
| Remove vertex | O(V + incident edges) | Requires updating references |
| Remove edge | O(degree) | Find and remove from vectors |

Use AdjGraph for dynamic scenarios with frequent structural modifications.
For static analysis, consider building with AdjGraph then converting to CoreGraph.
"""

# ==============================================================================
# 1. ADJACENCY LIST GRAPH (DIRECTED OR UNDIRECTED)
# ==============================================================================

"""
    AdjGraph{D} <: GraphInterface

Dynamic graph using adjacency lists (Vector{Vector{Int32}}) for neighbor storage.
Optimized for structural mutations with reasonable query performance.

# Type Parameters
- `D::Bool`: Directedness flag (true = directed, false = undirected)

# Fields (Internal - Access via interface methods)
- `neighbors::Vector{Vector{Int32}}`: Per-vertex neighbor lists
- `neighbor_to_edge::Vector{Vector{Int32}}`: Maps neighbor positions to edge indices (undirected only)
- `num_edges::Int32`: Number of (undirected) edges

# Construction
Use `build_adj_graph()` or `build_graph(AdjGraph, ...)` for safe construction:

```julia
# Basic construction
edges = [(1,2), (2,3), (1,3)]
g = build_adj_graph(edges; directed=false)

# Direct type construction with mutation
g = build_graph(AdjGraph, edges; directed=false)
add_edge!(g, 4, 1)  # Efficient dynamic modification
```

# Memory Layout Example
For graph with edges [(1,2), (2,3), (1,3)], undirected:
```
neighbors = [[2,3], [1,3], [1,2]]           # Vertex 1: neighbors 2,3; Vertex 2: neighbors 1,3, etc.
neighbor_to_edge = [[1,3], [1,2], [3,2]]    # Maps: v1's neighbor 2→edge 1, v1's neighbor 3→edge 3, etc.
```

# Performance Notes
- **Best for**: Dynamic graphs with frequent add/remove operations
- **Mutations**: O(1) additions, O(degree) removals
- **Memory**: ~16-24 bytes per directed edge (vector overhead + pointers)
- **Cache**: Good for sparse graphs, less optimal for dense graphs

# Mutation Support
```julia
# Efficient dynamic operations
new_vertex = add_vertex!(g)           # O(1) - just adds empty vectors
edge_idx = add_edge!(g, u, v)         # O(1) amortized - vector push
success = remove_edge!(g, u, v)       # O(degree) - find and remove
success = remove_vertex!(g, v)        # O(V + incident edges) - updates all references

# ⚠️ Warning: Removals may invalidate edge indices
# External arrays indexed by edges will become inconsistent
```
"""
mutable struct AdjGraph{D} <: GraphInterface
    const neighbors::Vector{Vector{Int32}}          # neighbors[v] = list of neighbors of v
    const neighbor_to_edge::Vector{Vector{Int32}}   # neighbor_to_edge[v][i] = undirected edge index (undirected only)
    num_edges::Int32                          # number of edges (undirected count)

    # Constructor for undirected graphs (D=false)
    function AdjGraph{false}(neighbors, neighbor_to_edge, num_edges)
        @assert length(neighbors) == length(neighbor_to_edge)
        @assert all(length(neighbors[v]) == length(neighbor_to_edge[v]) for v in 1:length(neighbors))
        new{false}(neighbors, neighbor_to_edge, num_edges)
    end

    # Constructor for directed graphs (D=true) - no neighbor_to_edge needed
    function AdjGraph{true}(neighbors, num_edges)
        # Create empty neighbor_to_edge to maintain struct consistency
        neighbor_to_edge = [Int32[] for _ in 1:length(neighbors)]
        new{true}(neighbors, neighbor_to_edge, num_edges)
    end
end

# Type aliases for clarity
const UndirectedAdjGraph = AdjGraph{false}
const DirectedAdjGraph = AdjGraph{true}

# Basic interface
@inline num_vertices(g::AdjGraph) = length(g.neighbors)
@inline num_edges(g::AdjGraph) = Int(g.num_edges)
@inline num_directed_edges(g::AdjGraph{D}) where D = D ? Int(g.num_edges) : 2 * Int(g.num_edges)
@inline has_vertex(g::AdjGraph, v::Integer) = 1 ≤ v ≤ num_vertices(g)
@inline is_directed_graph(::AdjGraph{D}) where D = D

@inline Base.@propagate_inbounds function neighbor_indices(g::AdjGraph, v::Integer)
    return g.neighbors[v]
end

@inline Base.@propagate_inbounds function has_edge(g::AdjGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    return @inbounds v in g.neighbors[u]
end

# Extended neighbor access
@inline Base.@propagate_inbounds function neighbor(g::AdjGraph, v::Integer, k::Integer)
    return g.neighbors[v][k]
end

# Edge indexing (only for undirected graphs)
@inline Base.@propagate_inbounds function edge_indices(g::AdjGraph{false}, v::Integer)
    return g.neighbor_to_edge[v]
end

@inline Base.@propagate_inbounds function edge_index(g::AdjGraph{false}, v::Integer, k::Integer)
    return g.neighbor_to_edge[v][k]
end

@inline Base.@propagate_inbounds function find_edge_index(g::AdjGraph{false}, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    @inbounds begin
        neighbors_list = g.neighbors[u]  # Safe after bounds check
        edge_indices_list = g.neighbor_to_edge[u]  # Safe after bounds check
        for (i, neighbor) in enumerate(neighbors_list)
            if neighbor == v
                return edge_indices_list[i]  # Safe after bounds check
            end
        end
    end
    return Int32(0)
end

# For directed graphs, edge and directed edge indexing are the same
@inline Base.@propagate_inbounds edge_indices(g::AdjGraph{true}, v::Integer) = directed_edge_indices(g, v)
@inline Base.@propagate_inbounds edge_index(g::AdjGraph{true}, v::Integer, k::Integer) = directed_edge_index(g, v, k)
@inline Base.@propagate_inbounds find_edge_index(g::AdjGraph{true}, u::Integer, v::Integer) = find_directed_edge_index(g, u, v)

# Directed edge indexing (both directed and undirected)
@inline Base.@propagate_inbounds function directed_edge_indices(g::AdjGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    degree_v = length(@inbounds g.neighbors[v])
    if degree_v == 0
        return 1:0  # Empty range
    end

    # Calculate starting index by summing degrees of vertices 1 to v-1
    start_idx = 1
    for u in 1:(v-1)
        start_idx += length(@inbounds g.neighbors[u])
    end

    return start_idx:(start_idx + degree_v - 1)
end

@inline Base.@propagate_inbounds function directed_edge_index(g::AdjGraph, v::Integer, k::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    # Calculate starting index by summing degrees of vertices 1 to v-1
    start_idx = 1
    for u in 1:(v-1)
        @inbounds start_idx += length(g.neighbors[u])
    end
    return Int32(start_idx + k - 1)
end

@inline Base.@propagate_inbounds function find_directed_edge_index(g::AdjGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    neighbors_list = @inbounds g.neighbors[u]  # Safe after bounds check
    for (i, neighbor) in enumerate(neighbors_list)
        if neighbor == v
            return @inbounds directed_edge_index(g, u, i)  # Safe after bounds check
        end
    end
    return Int32(0)
end

# ==============================================================================
# 2. WEIGHTED ADJACENCY LIST GRAPH (DIRECTED OR UNDIRECTED)
# ==============================================================================

"""
    WeightedAdjGraph{W,D} <: WeightedGraphInterface{W}

Weighted dynamic graph extending AdjGraph with parallel weight storage.
Combines the mutation efficiency of adjacency lists with type-safe weights.

# Type Parameters
- `W<:Number`: Weight type (Float64, Int32, etc.)
- `D::Bool`: Directedness flag

# Key Features
- **Same mutation performance as AdjGraph** for structural operations
- **Type-safe weights** with compile-time guarantees
- **Directional weights** even for undirected graphs
- **Parallel storage** maintaining weight-neighbor correspondence

# Weight Semantics
**Important**: Weights are always directional, even for undirected graphs.
This allows asymmetric edge properties while maintaining undirected connectivity.

```julia
# For undirected edge with different directional costs:
g = build_weighted_adj_graph([(1,2)], [1.5]; directed=false)
# Internally stores: neighbors[1]=[2], weights[1]=[1.5]
#                   neighbors[2]=[1], weights[2]=[1.5]
# But weights can be modified independently if needed

# Access via directional indexing:
idx_12 = find_directed_edge_index(g, 1, 2)  # Different from (2,1)
idx_21 = find_directed_edge_index(g, 2, 1)
weight_12 = edge_weight(g, idx_12)  # Initial: 1.5
weight_21 = edge_weight(g, idx_21)  # Initial: 1.5 (same value, different storage)
```

# Mutation Examples
```julia
edges = [(1,2), (2,3)]
weights = [1.0, 2.0]
g = build_weighted_adj_graph(edges, weights; directed=false)

# Add weighted edge
edge_idx = add_edge!(g, 3, 1, 1.5)  # O(1) amortized

# Efficient weight access during iteration
for (neighbor, weight) in neighbor_weights(g, v)
    process_weighted_neighbor(neighbor, weight)
end
```
"""
mutable struct WeightedAdjGraph{W<:Number,D} <: WeightedGraphInterface{W}
    const neighbors::Vector{Vector{Int32}}      # neighbors[v] = list of neighbors of v
    const weights::Vector{Vector{W}}            # weights[v] = weights parallel to neighbors[v]
    const neighbor_to_edge::Vector{Vector{Int32}}   # neighbor_to_edge[v][i] = undirected edge index (undirected only)
    num_edges::Int32                            # number of edges (undirected count)

    # Constructor for undirected weighted graphs (D=false)
    function WeightedAdjGraph{W,false}(neighbors, weights, neighbor_to_edge, num_edges) where W
        @assert length(neighbors) == length(weights) == length(neighbor_to_edge)
        @assert all(length(neighbors[v]) == length(weights[v]) == length(neighbor_to_edge[v])
                   for v in 1:length(neighbors))
        new{W,false}(neighbors, weights, neighbor_to_edge, num_edges)
    end

    # Constructor for directed weighted graphs (D=true)
    function WeightedAdjGraph{W,true}(neighbors, weights, num_edges) where W
        @assert length(neighbors) == length(weights)
        @assert all(length(neighbors[v]) == length(weights[v]) for v in 1:length(neighbors))
        # Create empty neighbor_to_edge for consistency
        neighbor_to_edge = [Int32[] for _ in 1:length(neighbors)]
        new{W,true}(neighbors, weights, neighbor_to_edge, num_edges)
    end
end

# Type aliases
const UndirectedWeightedAdjGraph{W} = WeightedAdjGraph{W,false}
const DirectedWeightedAdjGraph{W} = WeightedAdjGraph{W,true}

# Basic interface (same as AdjGraph)
@inline num_vertices(g::WeightedAdjGraph) = length(g.neighbors)
@inline num_edges(g::WeightedAdjGraph) = Int(g.num_edges)
@inline num_directed_edges(g::WeightedAdjGraph{W,D}) where {W,D} = D ? Int(g.num_edges) : 2 * Int(g.num_edges)
@inline has_vertex(g::WeightedAdjGraph, v::Integer) = 1 ≤ v ≤ num_vertices(g)
@inline is_directed_graph(::WeightedAdjGraph{W,D}) where {W,D} = D

@inline Base.@propagate_inbounds function neighbor_indices(g::WeightedAdjGraph, v::Integer)
    return g.neighbors[v]
end

@inline Base.@propagate_inbounds function has_edge(g::WeightedAdjGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    return @inbounds v in g.neighbors[u]
end

# Weight-specific methods
@inline function edge_weights(g::WeightedAdjGraph)
    # Flatten all weights into a single vector
    return [(g.weights...)...]
end

@inline Base.@propagate_inbounds function edge_weights(g::WeightedAdjGraph, v::Integer)
    return g.weights[v]
end

@inline Base.@propagate_inbounds function neighbor_weights(g::WeightedAdjGraph{W}, v::Integer) where W
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    @inbounds neighbors_list = g.neighbors[v]
    @inbounds weights_list = g.weights[v]
    return zip(neighbors_list, weights_list)
end

# Extended neighbor access (same pattern as AdjGraph)
@inline Base.@propagate_inbounds function neighbor(g::WeightedAdjGraph, v::Integer, k::Integer)
    return g.neighbors[v][k]
end

# Edge indexing (undirected graphs only)
@inline Base.@propagate_inbounds function edge_index(g::WeightedAdjGraph{W,false}, v::Integer, k::Integer) where W
    return g.neighbor_to_edge[v][k]
end

@inline Base.@propagate_inbounds function edge_indices(g::WeightedAdjGraph{W,false}, v::Integer) where W
    return g.neighbor_to_edge[v]
end

function find_edge_index(g::WeightedAdjGraph{W,false}, u::Integer, v::Integer) where W
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    neighbors_list = @inbounds g.neighbors[u]  # Safe after bounds check
    edge_indices_list = @inbounds g.neighbor_to_edge[u]  # Safe after bounds check
    for (i, neighbor) in enumerate(neighbors_list)
        if neighbor == v
            return @inbounds edge_indices_list[i]  # Safe after bounds check
        end
    end
    return Int32(0)
end

# For directed weighted graphs:
edge_indices(g::WeightedAdjGraph{W,true}, v::Integer) where W = directed_edge_indices(g, v)
edge_index(g::WeightedAdjGraph{W,true}, v::Integer, k::Integer) where W = directed_edge_index(g, v, k)
find_edge_index(g::WeightedAdjGraph{W,true}, u::Integer, v::Integer) where W = find_directed_edge_index(g, u, v)

# Directed edge indexing (same for both directed and undirected)
function directed_edge_index(g::WeightedAdjGraph, v::Integer, k::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    # Calculate starting index by summing degrees of vertices 1 to v-1
    start_idx = 1
    for u in 1:(v-1)
        @inbounds start_idx += length(g.neighbors[u])
    end

    return Int32(start_idx + k - 1)
end

function directed_edge_indices(g::WeightedAdjGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    @inbounds degree_v = length(g.neighbors[v])
    if degree_v == 0
        return 1:0  # Empty range
    end

    # Calculate starting index by summing degrees of vertices 1 to v-1
    start_idx = 1
    for u in 1:(v-1)
        @inbounds start_idx += length(g.neighbors[u])
    end

    return start_idx:(start_idx + degree_v - 1)
end

function find_directed_edge_index(g::WeightedAdjGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    neighbors_list = @inbounds g.neighbors[u]  # Safe after bounds check
    for (i, neighbor) in enumerate(neighbors_list)
        if neighbor == v
            return @inbounds directed_edge_index(g, u, i)  # Safe after bounds check
        end
    end
    return Int32(0)
end

# Weight access by edge indices
function edge_weight(g::WeightedAdjGraph, directed_edge_idx::Integer)
    @boundscheck 1 <= directed_edge_idx <= num_directed_edges(g) || throw(BoundsError(g, directed_edge_idx))
    # Find which vertex and position this directed edge corresponds to
    current_idx = 1
    for v in 1:num_vertices(g)
        degree_v = @inbounds length(g.neighbors[v])
        if directed_edge_idx < current_idx + degree_v
            local_idx = directed_edge_idx - current_idx + 1
            return @inbounds g.weights[v][local_idx]
        end
        current_idx += degree_v
    end
    throw(BoundsError(g, directed_edge_idx))
end

# ==============================================================================
# TYPE UNIONS FOR CONSTRUCTION
# ==============================================================================

const AdjGraphTypes = Union{AdjGraph, WeightedAdjGraph}

# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

"""
    to_adj_graph(g::GraphInterface) -> AdjGraph

Convert any GraphInterface graph to an AdjGraph.
Preserves the directedness of the original graph.
"""
function to_adj_graph(g::GraphInterface)
    directed = is_directed_graph(g)
    nv = num_vertices(g)
    neighbors = [collect(neighbor_indices(g, v)) for v in 1:nv]

    if directed
        return AdjGraph{true}(neighbors, num_edges(g))
    else
        neighbor_to_edge = [collect(edge_indices(g, v)) for v in 1:nv]
        return AdjGraph{false}(neighbors, neighbor_to_edge, num_edges(g))
    end
end

"""
    to_weighted_adj_graph(g::WeightedGraphInterface{W}) -> WeightedAdjGraph{W}

Convert any WeightedGraphInterface to a WeightedAdjGraph.
Preserves directedness and all weights.
"""
function to_weighted_adj_graph(g::WeightedGraphInterface{W}) where W
    directed = is_directed_graph(g)
    nv = num_vertices(g)

    neighbors = [collect(neighbor_indices(g, v)) for v in 1:nv]
    weights = [collect(edge_weights(g, v)) for v in 1:nv]

    if directed
        return WeightedAdjGraph{W,true}(neighbors, weights, num_edges(g))
    else
        neighbor_to_edge = [collect(edge_indices(g, v)) for v in 1:nv]
        return WeightedAdjGraph{W,false}(neighbors, weights, neighbor_to_edge, num_edges(g))
    end
end

# ==============================================================================
# CONSTRUCTOR-BASED CONVERSIONS (Idiomatic Julia Style)
# ==============================================================================

"""
    AdjGraph(g::GraphInterface) -> AdjGraph
    AdjGraph{D}(g::GraphInterface) -> AdjGraph{D}

Constructor-based conversion from any GraphInterface to AdjGraph.
The type-safe variant validates directedness matching.

See [`to_adj_graph`](@ref) for detailed documentation.
"""
AdjGraph(g::GraphInterface) = to_adj_graph(g)

function AdjGraph{D}(g::GraphInterface) where D
    @assert is_directed_graph(g) == D "Directedness mismatch: AdjGraph{$D} requires $(D ? "directed" : "undirected") graph, got $(is_directed_graph(g) ? "directed" : "undirected") graph"
    return to_adj_graph(g)
end

"""
    WeightedAdjGraph(g::WeightedGraphInterface{W}) -> WeightedAdjGraph{W}
    WeightedAdjGraph{W}(g::WeightedGraphInterface{W}) -> WeightedAdjGraph{W}
    WeightedAdjGraph{W,D}(g::WeightedGraphInterface{W}) -> WeightedAdjGraph{W,D}

Constructor-based conversion from any WeightedGraphInterface to WeightedAdjGraph.
Supports multiple forms with different levels of type specification.
The type-safe variant validates directedness matching.

See [`to_weighted_adj_graph`](@ref) for detailed documentation.
"""
WeightedAdjGraph(g::WeightedGraphInterface{W}) where W = to_weighted_adj_graph(g)
WeightedAdjGraph{W}(g::WeightedGraphInterface{W}) where W = to_weighted_adj_graph(g)

function WeightedAdjGraph{W,D}(g::WeightedGraphInterface{W}) where {W,D}
    @assert is_directed_graph(g) == D "Directedness mismatch: WeightedAdjGraph{$W,$D} requires $(D ? "directed" : "undirected") graph, got $(is_directed_graph(g) ? "directed" : "undirected") graph"
    return to_weighted_adj_graph(g)
end

# ==============================================================================
# UNIFIED BUILD FUNCTION FOR ADJGRAPH FAMILY
# ==============================================================================

"""
    build_graph(::Type{G}, edges; kwargs...) where {G<:AdjGraphTypes}

Build adjacency list graph optimized for dynamic modifications and mutations.

# Arguments
- `edges`: Vector of (u,v) tuples/pairs representing graph edges
- `directed=true`: Whether to build a directed graph
- `weights=[]`: Edge weights (for WeightedAdjGraph types)
- `validate=true`: Enable input validation

# Performance Characteristics
- **Dynamic-friendly**: Efficient vertex/edge additions and removals
- **Memory flexible**: Grows naturally, higher overhead than CSR
- **Mutation-optimized**: O(1) edge additions, efficient vertex operations

Use `AdjGraph` types when frequent graph modifications are expected.
For static graphs with performance-critical traversals, prefer `CoreGraph` types.
"""
function build_graph(::Type{G},
                     edges;
                     directed::Bool=true,
                     n::Integer=0,
                     weights::AbstractVector{W}=Number[],
                     validate::Bool=true) where {G<:AdjGraphTypes, W<:Number}

    # Quick validation (always enabled - very cheap)
    if !isempty(weights) && length(weights) != length(edges)
        throw(ArgumentError("weights must have same length as edges"))
    end

    if validate
        # Type validation (cheap)
        if !(eltype(edges) <: Union{Tuple{<:Integer,<:Integer}, Pair{<:Integer,<:Integer}})
            throw(ArgumentError("edges must be Vector of Tuples or Pairs, got Vector{$(eltype(edges))}"))
        end

        if !isempty(weights) && !(eltype(weights) <: Number)
            throw(ArgumentError("weights must be numeric, got $(eltype(weights))"))
        end
    end

    # Auto-detect vertex count from edges
    nv = if !isempty(edges)
        maximum(max(u, v) for (u, v) in edges)
    else
        0  # Empty graph with 0 vertices
    end

    # Determine final vertex count
    if n > 0
        # User specified vertex count - validate it's sufficient for the edges
        if nv > n
            throw(ArgumentError("Specified vertex count n=$n is insufficient for edges (need at least $nv vertices)"))
        end
        nv = n
    end

    # Initialize adjacency lists
    neighbors = [Int32[] for _ in 1:nv]

    if G <: WeightedAdjGraph && isempty(weights)
        weights = ones(W, length(edges))
    end

    if G <: WeightedAdjGraph
        neighbor_weights = [W[] for _ in 1:nv]
    end

    if !directed
        neighbor_to_edge = [Int32[] for _ in 1:nv]
    end

    # Fill adjacency lists AND validate edges in single loop
    for (edge_idx, (u, v)) in enumerate(edges)
        u32, v32 = Int32(u), Int32(v)

        # Validation during the loop (minimal overhead)
        if validate
            if !(1 ≤ u32 ≤ nv) || !(1 ≤ v32 ≤ nv)
                throw(ArgumentError("edge ($u32, $v32) contains vertex out of bounds [1, $nv]"))
            end
            if u32 == v32
                throw(ArgumentError("self-loops not supported: edge ($u32, $v32)"))
            end
        end

        # Add u → v
        push!(neighbors[u32], v32)
        if G <: WeightedAdjGraph
            push!(neighbor_weights[u32], weights[edge_idx])
        end

        if !directed
            # Add v → u for undirected
            push!(neighbors[v32], u32)
            if G <: WeightedAdjGraph
                push!(neighbor_weights[v32], weights[edge_idx])
            end
            push!(neighbor_to_edge[u32], Int32(edge_idx))
            push!(neighbor_to_edge[v32], Int32(edge_idx))
        end
    end

    # Construct appropriate type
    num_edges_val = Int32(length(edges))

    if G <: AdjGraph
        if directed
            return AdjGraph{true}(neighbors, num_edges_val)
        else
            return AdjGraph{false}(neighbors, neighbor_to_edge, num_edges_val)
        end
    elseif G <: WeightedAdjGraph
        if directed
            return WeightedAdjGraph{W,true}(neighbors, neighbor_weights, num_edges_val)
        else
            return WeightedAdjGraph{W,false}(neighbors, neighbor_weights, neighbor_to_edge, num_edges_val)
        end
    else
        throw(ArgumentError("Unsupported graph type: $G"))
    end
end

# ==============================================================================
# CONVENIENCE FUNCTIONS (NOW DELEGATE TO build_graph)
# ==============================================================================

"""Build adjacency list graph with same interface as CoreGraph."""
function build_adj_graph(edges; directed::Bool=true, kwargs...)
    return build_graph(AdjGraph, edges; directed=directed, kwargs...)
end

function build_weighted_adj_graph(edges, weights::AbstractVector{W}; directed::Bool=true, kwargs...) where W
    return build_graph(WeightedAdjGraph{W}, edges; directed=directed, weights=weights, kwargs...)
end

# ==============================================================================
# DISPLAY METHODS
# ==============================================================================

function Base.show(io::IO, g::AdjGraph{D}) where D
    direction = D ? "directed" : "undirected"
    print(io, "AdjGraph{$D} ($direction): $(num_vertices(g)) vertices, $(num_edges(g)) edges")
end

function Base.show(io::IO, g::WeightedAdjGraph{W,D}) where {W,D}
    direction = D ? "directed" : "undirected"
    print(io, "WeightedAdjGraph{$W,$D} ($direction): $(num_vertices(g)) vertices, $(num_edges(g)) edges")
end

# ==============================================================================
# MUTABLE GRAPH OPERATIONS FOR ADJACENCY LISTS
# ==============================================================================

# AdjGraph-specific: add_vertex!(g::AdjGraph) -> Int32
# O(1) create empty neighbor lists; see GraphInterface.add_vertex! for API docs.
function add_vertex!(g::AdjGraph)
    new_vertex = num_vertices(g) + 1

    # Add empty neighbor list
    push!(g.neighbors, Int32[])

    # Add empty neighbor_to_edge list (even for directed graphs for consistency)
    push!(g.neighbor_to_edge, Int32[])

    return Int32(new_vertex)
end

# WeightedAdjGraph-specific: add_vertex!(g::WeightedAdjGraph{W}) -> Int32
# O(1) create empty neighbor and weight lists; see GraphInterface.add_vertex! for API docs.
function add_vertex!(g::WeightedAdjGraph{W}) where W
    new_vertex = num_vertices(g) + 1

    # Add empty neighbor list
    push!(g.neighbors, Int32[])

    # Add empty weights list
    push!(g.weights, W[])

    # Add empty neighbor_to_edge list (even for directed graphs for consistency)
    push!(g.neighbor_to_edge, Int32[])

    return Int32(new_vertex)
end

# AdjGraph-specific: add_edge!(g::AdjGraph, u::Integer, v::Integer) -> Int32
# O(1) amortized vector push; see GraphInterface.add_edge! for API docs.
function add_edge!(g::AdjGraph, u::Integer, v::Integer)
    u32, v32 = Int32(u), Int32(v)

    # Check if vertices exist
    @boundscheck begin
        has_vertex(g, u32) || throw(BoundsError(g, u32))
        has_vertex(g, v32) || throw(BoundsError(g, v32))
    end

    # Check if edge already exists
    if has_edge(g, u32, v32)
        return Int32(0)  # Edge already exists
    end

    # Increment edge count
    g.num_edges += 1
    new_edge_idx = g.num_edges

    # Add u -> v
    push!(g.neighbors[u32], v32)

    if !is_directed_graph(g)
        # For undirected graphs, add both directions and update neighbor_to_edge
        push!(g.neighbors[v32], u32)
        push!(g.neighbor_to_edge[u32], new_edge_idx)
        push!(g.neighbor_to_edge[v32], new_edge_idx)
    end

    return new_edge_idx
end

# WeightedAdjGraph-specific: add_edge!(g::WeightedAdjGraph{W}, u::Integer, v::Integer, weight::W) -> Int32
# O(1) amortized with weight storage; see GraphInterface.add_edge! for API docs.
function add_edge!(g::WeightedAdjGraph{W}, u::Integer, v::Integer, weight::W) where W
    u32, v32 = Int32(u), Int32(v)

    # Check if vertices exist
    @boundscheck begin
        has_vertex(g, u32) || throw(BoundsError(g, u32))
        has_vertex(g, v32) || throw(BoundsError(g, v32))
    end

    # Check if edge already exists
    if has_edge(g, u32, v32)
        return Int32(0)  # Edge already exists
    end

    # Increment edge count
    g.num_edges += 1
    new_edge_idx = g.num_edges

    # Add u -> v with weight
    push!(g.neighbors[u32], v32)
    push!(g.weights[u32], weight)

    if !is_directed_graph(g)
        # For undirected graphs, add both directions with same weight
        push!(g.neighbors[v32], u32)
        push!(g.weights[v32], weight)
        push!(g.neighbor_to_edge[u32], new_edge_idx)
        push!(g.neighbor_to_edge[v32], new_edge_idx)
    end

    return new_edge_idx
end

# AdjGraph-specific: remove_vertex!(g::AdjGraph, v::Integer) -> Bool
# O(V+E) update/compaction; see GraphInterface.remove_vertex! for API docs.
function remove_vertex!(g::AdjGraph, v::Integer)
    v32 = Int32(v)

    # Check if vertex exists
    if !has_vertex(g, v32)
        return false
    end

    nv = num_vertices(g)

    # Remove all edges incident to v (this also updates edge count)
    # First collect neighbors to avoid modifying while iterating
    neighbors_to_remove = copy(g.neighbors[v32])
    for neighbor in neighbors_to_remove
        remove_edge!(g, v32, neighbor)
    end

    # For directed graphs, also remove incoming edges
    if is_directed_graph(g)
        for u in 1:nv
            if u != v32 && has_edge(g, u, v32)
                remove_edge!(g, u, v32)
            end
        end
    end

    # Remove vertex from adjacency lists
    deleteat!(g.neighbors, v32)
    deleteat!(g.neighbor_to_edge, v32)

    # Update all neighbor references: vertices > v become vertices > v-1
    for u in 1:length(g.neighbors)
        for i in 1:length(g.neighbors[u])
            if g.neighbors[u][i] > v32
                g.neighbors[u][i] -= 1
            end
        end
    end

    return true
end

# WeightedAdjGraph-specific: remove_vertex!(g::WeightedAdjGraph, v::Integer) -> Bool
# O(V+E) with weight maintenance; see GraphInterface.remove_vertex! for API docs.
function remove_vertex!(g::WeightedAdjGraph, v::Integer)
    v32 = Int32(v)

    # Check if vertex exists
    if !has_vertex(g, v32)
        return false
    end

    nv = num_vertices(g)

    # Remove all edges incident to v
    neighbors_to_remove = copy(g.neighbors[v32])
    for neighbor in neighbors_to_remove
        remove_edge!(g, v32, neighbor)
    end

    # For directed graphs, also remove incoming edges
    if is_directed_graph(g)
        for u in 1:nv
            if u != v32 && has_edge(g, u, v32)
                remove_edge!(g, u, v32)
            end
        end
    end

    # Remove vertex from adjacency lists
    deleteat!(g.neighbors, v32)
    deleteat!(g.weights, v32)
    deleteat!(g.neighbor_to_edge, v32)

    # Update all neighbor references
    for u in 1:length(g.neighbors)
        for i in 1:length(g.neighbors[u])
            if g.neighbors[u][i] > v32
                g.neighbors[u][i] -= 1
            end
        end
    end

    return true
end

# AdjGraph-specific: remove_edge!(g::AdjGraph, u::Integer, v::Integer) -> Bool
# O(degree) remove via linear search; see GraphInterface.remove_edge! for API docs.
function remove_edge!(g::AdjGraph, u::Integer, v::Integer)
    u32, v32 = Int32(u), Int32(v)

    # Check if vertices exist
    @boundscheck begin
        has_vertex(g, u32) || throw(BoundsError(g, u32))
        has_vertex(g, v32) || throw(BoundsError(g, v32))
    end

    # Check if edge exists
    if !has_edge(g, u32, v32)
        return false
    end

    # Find position of v in u's neighbor list
    u_neighbors = g.neighbors[u32]
    v_pos = findfirst(==(v32), u_neighbors)

    if v_pos === nothing
        return false  # Edge doesn't exist
    end

    # Remove v from u's neighbor list
    deleteat!(g.neighbors[u32], v_pos)

    if !is_directed_graph(g)
        # For undirected graphs, also remove u from v's neighbor list
        v_neighbors = g.neighbors[v32]
        u_pos = findfirst(==(u32), v_neighbors)

        if u_pos !== nothing
            deleteat!(g.neighbors[v32], u_pos)
            deleteat!(g.neighbor_to_edge[v32], u_pos)
        end

        # Remove from neighbor_to_edge for u
        deleteat!(g.neighbor_to_edge[u32], v_pos)

        # Decrement edge count (only once for undirected)
        g.num_edges -= 1
    else
        # For directed graphs, decrement edge count
        g.num_edges -= 1
    end

    return true
end

# WeightedAdjGraph-specific: remove_edge!(g::WeightedAdjGraph, u::Integer, v::Integer) -> Bool
# O(degree) remove with weight cleanup; see GraphInterface.remove_edge! for API docs.
function remove_edge!(g::WeightedAdjGraph, u::Integer, v::Integer)
    u32, v32 = Int32(u), Int32(v)

    # Check if vertices exist
    if !has_vertex(g, u32) || !has_vertex(g, v32)
        return false
    end

    # Check if edge exists
    if !has_edge(g, u32, v32)
        return false
    end

    # Find position of v in u's neighbor list
    u_neighbors = g.neighbors[u32]
    v_pos = findfirst(==(v32), u_neighbors)

    if v_pos === nothing
        return false  # Edge doesn't exist
    end

    # Remove v from u's neighbor list and corresponding weight
    deleteat!(g.neighbors[u32], v_pos)
    deleteat!(g.weights[u32], v_pos)

    if !is_directed_graph(g)
        # For undirected graphs, also remove from v's lists
        v_neighbors = g.neighbors[v32]
        u_pos = findfirst(==(u32), v_neighbors)

        if u_pos !== nothing
            deleteat!(g.neighbors[v32], u_pos)
            deleteat!(g.weights[v32], u_pos)
            deleteat!(g.neighbor_to_edge[v32], u_pos)
        end

        # Remove from neighbor_to_edge for u
        deleteat!(g.neighbor_to_edge[u32], v_pos)

        # Decrement edge count (only once for undirected)
        g.num_edges -= 1
    else
        # For directed graphs, decrement edge count
        g.num_edges -= 1
    end

    return true
end

# ==============================================================================
# EDGE INDEX STABILITY NOTES FOR ADJACENCY LISTS
# ==============================================================================

"""
Edge Index Stability for AdjGraph Types:
=======================================

ADDITIONS (add_vertex!, add_edge!):
✅ STABLE: All existing edge indices remain valid
✅ STABLE: All existing vertex indices remain valid
✅ NEW: New edges get indices > previous maximum

REMOVALS (remove_vertex!, remove_edge!):
❌ UNSTABLE: Edge indices may be invalidated
❌ UNSTABLE: Vertex indices > removed_vertex are decremented

RECOMMENDATIONS:
1. Batch all additions before any removals
2. If removals are necessary, rebuild external arrays afterward
3. For frequent mutations, consider using Dict-based indexing instead of arrays
4. Use vertex/edge properties within the graph rather than external arrays when possible

EXAMPLE SAFE USAGE:
```julia
g = build_adj_graph(edges, nv)

# Safe: Adding only
edge_props = Vector{String}(undef, num_edges(g))
add_edge!(g, u, v)  # ✅ Previous indices still valid
resize!(edge_props, num_edges(g))  # Expand array for new edge

# Unsafe: Removing invalidates indices
remove_edge!(g, u, v)  # ❌ edge_props indices now invalid!

# Better: Use PropertyGraph instead
pg = PropertyGraph(g, vertex_props, edge_props)
remove_edge!(pg, u, v)  # ✅ Properties maintained automatically
```
"""
