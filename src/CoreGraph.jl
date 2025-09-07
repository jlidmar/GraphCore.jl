# CoreGraph.jl

# Copyright (c) 2025 Jack Lidmar
# All rights reserved.

# This software is licensed under the MIT License. See the LICENSE file for details.

# SPDX-FileCopyrightText: 2025 Jack Lidmar <jlidmar@kth.se>
# SPDX-License-Identifier: MIT

"""
Core Graph Structures
=======================================

High-performance graph implementations using Compressed Sparse Row (CSR) format.

## Design Rationale

**CSR Format Benefits**:
- **Cache Efficiency**: Neighbors stored contiguously for optimal memory access patterns
- **Space Efficiency**: Minimal pointer overhead compared to adjacency lists
- **Index Stability**: External arrays remain valid during analysis phases
- **O(1) Access**: Direct index computation without hash table lookups

**Trade-offs**:
- **Static Structure**: Expensive to modify after construction (use AdjGraph for mutations)
- **Construction Cost**: Requires two passes (degree counting + filling)
- **Memory Allocation**: Fixed-size arrays determined at construction time

## Storage Layout

```
CoreGraph{false} (undirected):
├── vertex_offsets: [1, 3, 6, 8, ...]     # CSR row pointers
├── neighbors: [2, 3, 1, 3, 1, 2, ...]    # Flattened neighbor lists
├── neighbor_to_edge: [1, 2, 1, 3, 2, 3, ...] # Maps to undirected edge indices
└── num_edges: 3                           # Undirected edge count

CoreGraph{true} (directed):
├── vertex_offsets: [1, 3, 5, 6, ...]     # CSR row pointers
├── neighbors: [2, 3, 3, 4, ...]          # Flattened neighbor lists
├── neighbor_to_edge: []                   # Empty (not needed for directed)
└── num_edges: 4                           # Directed edge count
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Neighbor access | O(1) | Direct offset computation |
| Edge existence | O(degree) | Linear search in neighbor list |
| Edge index lookup | O(degree) | With neighbor_to_edge mapping |
| Construction | O(V + E) | Two-pass algorithm |
| Mutation | O(V + E) | Requires full reconstruction |

Use CoreGraph for analysis-heavy workloads where the graph structure is stable.
For frequent mutations, consider AdjGraph or build-convert-analyze-convert workflows.
"""

# ==============================================================================
# 1. CORE GRAPH (DIRECTED OR UNDIRECTED)
# ==============================================================================

"""
    CoreGraph{D} <: GraphInterface

High-performance graph using Compressed Sparse Row (CSR) storage format.

# Type Parameters
- `D::Bool`: Directedness flag (true = directed, false = undirected)

# Fields (Internal - Access via interface methods)
- `vertex_offsets::Vector{Int32}`: CSR row pointers (length = nv + 1)
- `neighbors::Vector{Int32}`: Flattened neighbor lists
- `neighbor_to_edge::Vector{Int32}`: Maps neighbor positions to undirected edge indices (undirected only)
- `num_edges::Int32`: Number of (undirected) edges

# Construction
Use `build_core_graph()` or `build_graph(CoreGraph, ...)` for safe construction:

```julia
# Basic construction
edges = [(1,2), (2,3), (1,3)]
g = build_core_graph(edges; directed=false)

# With validation disabled (faster, but unsafe)
g = build_graph(CoreGraph, edges; directed=false, validate=false)
```

# Memory Layout Example
For graph with edges [(1,2), (2,3), (1,3)], undirected:
```
vertex_offsets = [1, 3, 6, 8]       # Vertex 1: neighbors[1:2], Vertex 2: neighbors[3:5], etc.
neighbors = [2, 3, 1, 3, 1, 2]      # Flattened: [neighbors(1), neighbors(2), neighbors(3)]
neighbor_to_edge = [1, 3, 1, 2, 3, 2] # Maps each neighbor to its edge index
```

# Performance Notes
- **Best for**: Static graphs with frequent neighbor access
- **Avoid for**: Graphs requiring frequent structural modifications
- **Memory**: ~12-16 bytes per directed edge (depending on architecture)
- **Cache**: Excellent locality for neighbor iteration
"""
mutable struct CoreGraph{D} <: GraphInterface
    const vertex_offsets::Vector{Int32}     # vertex_offsets[v] = start index for vertex v
    const neighbors::Vector{Int32}          # neighbor lists (flattened)
    const neighbor_to_edge::Vector{Int32}   # maps neighbor position -> undirected edge index (undirected only)
    num_edges::Int32                        # number of edges (undirected count) - mutable for efficiency

    # Constructor for undirected graphs (D=false)
    function CoreGraph{false}(vertex_offsets, neighbors, neighbor_to_edge, num_edges)
        @assert length(vertex_offsets) >= 1
        @assert vertex_offsets[1] == 1
        @assert vertex_offsets[end] == length(neighbors) + 1
        @assert length(neighbors) == length(neighbor_to_edge)
        new{false}(vertex_offsets, neighbors, neighbor_to_edge, num_edges)
    end

    # Constructor for directed graphs (D=true) - no neighbor_to_edge needed
    function CoreGraph{true}(vertex_offsets, neighbors, num_edges)
        @assert length(vertex_offsets) >= 1
        @assert vertex_offsets[1] == 1
        @assert vertex_offsets[end] == length(neighbors) + 1
        # Create empty neighbor_to_edge to maintain struct consistency
        neighbor_to_edge = Int32[]
        new{true}(vertex_offsets, neighbors, neighbor_to_edge, num_edges)
    end
end

# Type aliases for clarity
const UndirectedCoreGraph = CoreGraph{false}
const DirectedCoreGraph = CoreGraph{true}

# Basic interface
@inline num_vertices(g::CoreGraph) = length(g.vertex_offsets) - 1
@inline num_edges(g::CoreGraph) = Int(g.num_edges)
@inline num_directed_edges(g::CoreGraph{D}) where D = D ? Int(g.num_edges) : 2 * Int(g.num_edges)
@inline has_vertex(g::CoreGraph, v::Integer) = 1 ≤ v ≤ num_vertices(g)
@inline is_directed_graph(::CoreGraph{D}) where D = D

@inline Base.@propagate_inbounds function neighbor_indices(g::CoreGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    end_idx = @inbounds g.vertex_offsets[v + 1] - Int32(1)
    return @inbounds @view g.neighbors[start_idx:end_idx]
end

@inline Base.@propagate_inbounds function has_edge(g::CoreGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    neighbors_view = @inbounds neighbor_indices(g, u)  # Safe after bounds check
    return v in neighbors_view
end

@inline Base.@propagate_inbounds function edge_indices(g::CoreGraph{false}, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    end_idx = @inbounds g.vertex_offsets[v + 1] - Int32(1)
    return @inbounds @view g.neighbor_to_edge[start_idx:end_idx]
end

@inline Base.@propagate_inbounds function find_edge_index(g::CoreGraph{false}, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    @inbounds begin
        start_idx = g.vertex_offsets[u]
        end_idx = g.vertex_offsets[u + 1] - Int32(1)
        neighbors_view = @view g.neighbors[start_idx:end_idx]
        edge_indices_view = @view g.neighbor_to_edge[start_idx:end_idx]
        for (i, neighbor) in enumerate(neighbors_view)
            if neighbor == v
                return edge_indices_view[i]
            end
        end
    end
    return Int32(0)
end

# For directed graphs, edge and directed edge indexing are the same
@inline Base.@propagate_inbounds edge_indices(g::CoreGraph{true}, v::Integer) = directed_edge_indices(g, v)
@inline Base.@propagate_inbounds edge_index(g::CoreGraph{true}, v::Integer, k::Integer) = directed_edge_index(g, v, k)
@inline Base.@propagate_inbounds find_edge_index(g::CoreGraph{true}, u::Integer, v::Integer) = find_directed_edge_index(g, u, v)

# Directed edge indexing (both directed and undirected)
@inline Base.@propagate_inbounds function directed_edge_indices(g::CoreGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    end_idx = @inbounds g.vertex_offsets[v + 1] - Int32(1)
    return start_idx:end_idx
end

@inline Base.@propagate_inbounds function directed_edge_index(g::CoreGraph, v::Integer, k::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    return Int32(start_idx + k - 1)
end

@inline Base.@propagate_inbounds function find_directed_edge_index(g::CoreGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    @inbounds begin
        start_idx = g.vertex_offsets[u]
        end_idx = g.vertex_offsets[u + 1] - Int32(1)
        neighbors_view = @view g.neighbors[start_idx:end_idx]
        for (i, neighbor) in enumerate(neighbors_view)
            if neighbor == v
                return Int32(start_idx + i - 1)
            end
        end
    end
    return Int32(0)
end

# ==============================================================================
# 2. WEIGHTED GRAPH (DIRECTED OR UNDIRECTED)
# ==============================================================================

"""
    WeightedGraph{W,D} <: WeightedGraphInterface{W}

Weighted graph extending CoreGraph with parallel weight storage.

# Type Parameters
- `W<:Number`: Weight type (Float64, Int32, etc.)
- `D::Bool`: Directedness flag

# Key Features
- **Same performance as CoreGraph** for structural operations
- **Type-safe weights** with compile-time guarantees
- **Directional weights** even for undirected graphs
- **Parallel storage** for cache-efficient weight access

# Weight Semantics
**Important**: Weights are always directional, even for undirected graphs.
This allows asymmetric edge properties (e.g., different costs per direction).

```julia
# For undirected edge (1,2) with weight 1.5:
edge_weight(g, find_directed_edge_index(g, 1, 2)) # → 1.5
edge_weight(g, find_directed_edge_index(g, 2, 1)) # → 1.5 (same value, different index)

# But can be set differently if needed:
weights = [1.5, 2.0]  # Different costs for each direction
g = build_weighted_graph([(1,2), (2,1)], weights; directed=true)
```

# Construction Examples
```julia
# Undirected weighted graph
edges = [(1,2), (2,3)]
weights = [1.5, 2.0]
g = build_weighted_graph(edges, weights; directed=false)

# Type-specific construction
g = build_graph(WeightedGraph{Float32}, edges; weights=weights, directed=false)
```
"""
mutable struct WeightedGraph{W<:Number,D} <: WeightedGraphInterface{W}
    const vertex_offsets::Vector{Int32}     # vertex_offsets[v] = start index for vertex v
    const neighbors::Vector{Int32}          # neighbor lists (flattened)
    const weights::Vector{W}                # weights parallel to neighbors
    const neighbor_to_edge::Vector{Int32}   # maps neighbor position -> undirected edge index (undirected only)
    num_edges::Int32                        # number of edges (undirected count) - mutable for efficiency

    # Constructor for undirected weighted graphs (D=false)
    function WeightedGraph{W,false}(vertex_offsets, neighbors, weights,
                                   neighbor_to_edge, num_edges) where W
        @assert length(vertex_offsets) >= 1
        @assert vertex_offsets[1] == 1
        @assert vertex_offsets[end] == length(neighbors) + 1
        @assert length(neighbors) == length(weights) == length(neighbor_to_edge)
        new{W,false}(vertex_offsets, neighbors, weights, neighbor_to_edge, num_edges)
    end

    # Constructor for directed weighted graphs (D=true)
    function WeightedGraph{W,true}(vertex_offsets, neighbors, weights, num_edges) where W
        @assert length(vertex_offsets) >= 1
        @assert vertex_offsets[1] == 1
        @assert vertex_offsets[end] == length(neighbors) + 1
        @assert length(neighbors) == length(weights)
        # Create empty neighbor_to_edge for consistency
        neighbor_to_edge = Int32[]
        new{W,true}(vertex_offsets, neighbors, weights, neighbor_to_edge, num_edges)
    end
end

# Type aliases
const UndirectedWeightedGraph{W} = WeightedGraph{W,false}
const DirectedWeightedGraph{W} = WeightedGraph{W,true}

# Basic interface (same as CoreGraph)
@inline num_vertices(g::WeightedGraph) = length(g.vertex_offsets) - 1
@inline num_edges(g::WeightedGraph) = Int(g.num_edges)
@inline num_directed_edges(g::WeightedGraph{W,D}) where {W,D} = D ? Int(g.num_edges) : 2 * Int(g.num_edges)
@inline has_vertex(g::WeightedGraph, v::Integer) = 1 ≤ v ≤ num_vertices(g)
@inline is_directed_graph(::WeightedGraph{W,D}) where {W,D} = D

@inline Base.@propagate_inbounds function neighbor_indices(g::WeightedGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    end_idx = @inbounds g.vertex_offsets[v + 1] - Int32(1)
    return @inbounds @view g.neighbors[start_idx:end_idx]
end

@inline Base.@propagate_inbounds function has_edge(g::WeightedGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    neighbors_view = @inbounds neighbor_indices(g, u)  # Safe after bounds check
    return v in neighbors_view
end

# Weight-specific methods
@inline function edge_weights(g::WeightedGraph)
    return g.weights
end

@inline Base.@propagate_inbounds function edge_weights(g::WeightedGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    end_idx = @inbounds g.vertex_offsets[v + 1] - Int32(1)
    return @inbounds @view g.weights[start_idx:end_idx]
end

@inline Base.@propagate_inbounds function neighbor_weights(g::WeightedGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    @inbounds begin
        start_idx = g.vertex_offsets[v]
        end_idx = g.vertex_offsets[v+1] - Int32(1)
        neighbors_view = @view g.neighbors[start_idx:end_idx]
        weights_view = @view g.weights[start_idx:end_idx]
        return zip(neighbors_view, weights_view)
    end
end

# Extended neighbor access (same pattern as CoreGraph)
@inline Base.@propagate_inbounds function neighbor(g::WeightedGraph, v::Integer, k::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    return @inbounds g.neighbors[start_idx + k - 1]
end

# Edge indexing (undirected graphs only)
@inline Base.@propagate_inbounds function edge_index(g::WeightedGraph{W,false}, v::Integer, k::Integer) where W
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    return @inbounds g.neighbor_to_edge[start_idx + k - 1]
end

@inline Base.@propagate_inbounds function edge_indices(g::WeightedGraph{W,false}, v::Integer) where W
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    end_idx = @inbounds g.vertex_offsets[v + 1] - Int32(1)
    return @inbounds @view g.neighbor_to_edge[start_idx:end_idx]
end

@inline Base.@propagate_inbounds function find_edge_index(g::WeightedGraph{W,false}, u::Integer, v::Integer) where W
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    @inbounds begin
        start_idx = g.vertex_offsets[u]
        end_idx = g.vertex_offsets[u+1] - Int32(1)
        neighbors_view = @view g.neighbors[start_idx:end_idx]
        edge_indices_view = @view g.neighbor_to_edge[start_idx:end_idx]
        for (i, neighbor) in enumerate(neighbors_view)
            if neighbor == v
                return edge_indices_view[i]
            end
        end
    end
    return Int32(0)
end

# For directed weighted graphs:
@inline Base.@propagate_inbounds edge_indices(g::WeightedGraph{W,true}, v::Integer) where W = directed_edge_indices(g, v)
@inline Base.@propagate_inbounds edge_index(g::WeightedGraph{W,true}, v::Integer, k::Integer) where W = directed_edge_index(g, v, k)
@inline Base.@propagate_inbounds find_edge_index(g::WeightedGraph{W,true}, u::Integer, v::Integer) where W = find_directed_edge_index(g, u, v)

# Directed edge indexing (same for both directed and undirected)
@inline Base.@propagate_inbounds function directed_edge_index(g::WeightedGraph, v::Integer, k::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    return Int32(start_idx + k - 1)
end

@inline Base.@propagate_inbounds function directed_edge_indices(g::WeightedGraph, v::Integer)
    @boundscheck has_vertex(g, v) || throw(BoundsError(g, v))
    start_idx = @inbounds g.vertex_offsets[v]
    end_idx = @inbounds g.vertex_offsets[v + 1] - Int32(1)
    return start_idx:end_idx
end

@inline Base.@propagate_inbounds function find_directed_edge_index(g::WeightedGraph, u::Integer, v::Integer)
    @boundscheck begin
        has_vertex(g, u) || throw(BoundsError(g, u))
        has_vertex(g, v) || throw(BoundsError(g, v))
    end
    @inbounds begin
        start_idx = g.vertex_offsets[u]
        end_idx = g.vertex_offsets[u+1] - Int32(1)
        neighbors_view = @view g.neighbors[start_idx:end_idx]
        for (i, neighbor) in enumerate(neighbors_view)
            if neighbor == v
                return Int32(start_idx + i - 1)
            end
        end
    end
    return Int32(0)
end

# Weight access by edge indices
@inline Base.@propagate_inbounds function edge_weight(g::WeightedGraph, directed_edge_idx::Integer)
    return g.weights[directed_edge_idx]
end

# ==============================================================================
# TYPE UNIONS FOR CONSTRUCTION
# ==============================================================================

const GraphCoreTypes = Union{CoreGraph, WeightedGraph}

# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

"""
    to_core_graph(g::GraphInterface) -> CoreGraph

Convert any GraphInterface implementation to a CoreGraph.
Preserves the directedness of the original graph.
"""
function to_core_graph(g::GraphInterface)
    vertex_offsets = Int32[1; accumulate(+, degree(g, v) for v in vertices(g); init=one(Int32))]
    neighbors = reduce(vcat, [neighbor_indices(g, v) for v in vertices(g)])  # Array, not generator!
    if is_directed_graph(g)
        return CoreGraph{true}(vertex_offsets, neighbors, num_edges(g))
    else
        neighbor_to_edge = reduce(vcat, [edge_indices(g, v) for v in vertices(g)])  # Array, not generator!
        return CoreGraph{false}(vertex_offsets, neighbors, neighbor_to_edge, num_edges(g))
    end
end

"""
    to_weighted_graph(g::WeightedGraphInterface{W}) -> WeightedGraph{W}

Convert any WeightedGraphInterface to a WeightedGraph.
Preserves the directedness and all weights of the original graph.
"""
function to_weighted_graph(g::WeightedGraphInterface{W}) where W
    # Build CSR offsets (same as to_core_graph)
    vertex_offsets = Int32[1; accumulate(+, degree(g, v) for v in vertices(g); init=one(Int32))]

    # Build neighbors and weights arrays using reduce
    neighbors = reduce(vcat, [neighbor_indices(g, v) for v in vertices(g)])
    weights = reduce(vcat, [edge_weights(g, v) for v in vertices(g)])

    if is_directed_graph(g)
        return WeightedGraph{W,true}(vertex_offsets, neighbors, weights, num_edges(g))
    else
        neighbor_to_edge = reduce(vcat, [edge_indices(g, v) for v in vertices(g)])
        return WeightedGraph{W,false}(vertex_offsets, neighbors, weights, neighbor_to_edge, num_edges(g))
    end
end

# ==============================================================================
# CONSTRUCTOR-BASED CONVERSIONS (Idiomatic Julia Style)
# ==============================================================================

"""
    CoreGraph(g::GraphInterface) -> CoreGraph
    CoreGraph{D}(g::GraphInterface) -> CoreGraph{D}

Constructor-based conversion from any GraphInterface to CoreGraph.
The type-safe variant validates directedness matching.

See [`to_core_graph`](@ref) for detailed documentation.
"""
CoreGraph(g::GraphInterface) = to_core_graph(g)

function CoreGraph{D}(g::GraphInterface) where D
    @assert is_directed_graph(g) == D "Directedness mismatch: CoreGraph{$D} requires $(D ? "directed" : "undirected") graph, got $(is_directed_graph(g) ? "directed" : "undirected") graph"
    return to_core_graph(g)
end

"""
    WeightedGraph(g::WeightedGraphInterface{W}) -> WeightedGraph{W}
    WeightedGraph{W}(g::WeightedGraphInterface{W}) -> WeightedGraph{W}
    WeightedGraph{W,D}(g::WeightedGraphInterface{W}) -> WeightedGraph{W,D}

Constructor-based conversion from any WeightedGraphInterface to WeightedGraph.
Supports multiple forms with different levels of type specification.
The type-safe variant validates directedness matching.

See [`to_weighted_graph`](@ref) for detailed documentation.
"""
WeightedGraph(g::WeightedGraphInterface{W}) where W = to_weighted_graph(g)
WeightedGraph{W}(g::WeightedGraphInterface{W}) where W = to_weighted_graph(g)

function WeightedGraph{W,D}(g::WeightedGraphInterface{W}) where {W,D}
    @assert is_directed_graph(g) == D "Directedness mismatch: WeightedGraph{$W,$D} requires $(D ? "directed" : "undirected") graph, got $(is_directed_graph(g) ? "directed" : "undirected") graph"
    return to_weighted_graph(g)
end

# ==============================================================================
# UNIFIED BUILD FUNCTION
# ==============================================================================

"""
    build_graph(::Type{G}, edges; kwargs...) -> G

Build a graph from an edge list with comprehensive validation and flexible options.

# Arguments
- `edges`: Vector of (u,v) tuples/pairs representing graph edges
- `directed=true`: Whether to build a directed graph
- `n=0`: Number of vertices (0 = auto-detect from edges)
- `weights=[]`: Edge weights (for WeightedGraph types)
- `validate=true`: Enable input validation (recommended for safety)

# Examples
```julia
# Basic graphs
g = build_graph(CoreGraph, [(1,2), (2,3)]; directed=false)
wg = build_graph(WeightedGraph{Float64}, [(1,2), (2,3)]; weights=[1.5, 2.0], directed=false)

# Graph with isolated vertices
g = build_graph(CoreGraph, [(1,2)]; n=5, directed=false)  # Creates isolated vertices 3,4,5

# High-performance mode (skip validation)
g = build_graph(CoreGraph, trusted_edges; directed=false, validate=false)
```

Optimized for CSR representation with efficient construction and memory usage.
For dynamic graphs requiring frequent mutations, consider `AdjGraph` types.
"""
function build_graph(::Type{G},
                     edges;
                     directed::Bool=true,
                     n::Integer=0,
                     weights::AbstractVector{W}=Float64[],
                     validate::Bool=true) where {G<:GraphCoreTypes, W<:Number}

    # Quick validation (always enabled - very cheap)
    if !isempty(weights) && length(weights) != length(edges)
        throw(ArgumentError("weights must have same length as edges: got $(length(weights)) weights for $(length(edges)) edges"))
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

    # Handle empty weights for weighted graphs
    if G <: WeightedGraph && isempty(weights)
        weights = ones(W, length(edges))
    end

    # Count degrees AND validate edges in single loop
    degrees = zeros(Int32, nv)
    for (u, v) in edges
        # Validation during the loop (minimal overhead)
        if validate
            if !(1 ≤ u ≤ nv) || !(1 ≤ v ≤ nv)
                throw(ArgumentError("edge ($u, $v) contains vertex out of bounds [1, $nv]"))
            end
            if u == v
                throw(ArgumentError("self-loops not supported: edge ($u, $u)"))
            end
        end

        # Degree counting (always needed)
        degrees[u] += 1
        if !directed
            degrees[v] += 1
        end
    end

    # Build CSR structure
    vertex_offsets = Vector{Int32}(undef, nv + 1)
    vertex_offsets[1] = 1
    for i in 1:nv
        vertex_offsets[i + 1] = vertex_offsets[i] + degrees[i]
    end

    total_directed_edges = vertex_offsets[end] - 1
    neighbors = Vector{Int32}(undef, total_directed_edges)

    # Type-specific storage
    if G <: WeightedGraph
        neighbor_weights = Vector{W}(undef, total_directed_edges)
    end

    if !directed
        neighbor_to_edge = Vector{Int32}(undef, total_directed_edges)
    else
        neighbor_to_edge = Int32[]
    end

    # Fill adjacency structure
    current_pos = copy(vertex_offsets[1:nv])

    for (edge_idx, (u, v)) in enumerate(edges)
        # Add u → v
        @inbounds neighbors[current_pos[u]] = Int32(v)
        if G <: WeightedGraph
            @inbounds neighbor_weights[current_pos[u]] = weights[edge_idx]
        end
        if !directed
            @inbounds neighbor_to_edge[current_pos[u]] = Int32(edge_idx)
        end
        @inbounds current_pos[u] += 1

        # Add v → u for undirected graphs
        if !directed
            @inbounds neighbors[current_pos[v]] = Int32(u)
            if G <: WeightedGraph
                @inbounds neighbor_weights[current_pos[v]] = weights[edge_idx]  # Same weight
            end
            @inbounds neighbor_to_edge[current_pos[v]] = Int32(edge_idx)
            @inbounds current_pos[v] += 1
        end
    end

    # Construct the appropriate graph type
    num_edges_val = Int32(length(edges))

    if G <: CoreGraph
        if directed
            return CoreGraph{true}(vertex_offsets, neighbors, num_edges_val)
        else
            return CoreGraph{false}(vertex_offsets, neighbors, neighbor_to_edge, num_edges_val)
        end
    elseif G <: WeightedGraph
        if directed
            return WeightedGraph{W,true}(vertex_offsets, neighbors, neighbor_weights, num_edges_val)
        else
            return WeightedGraph{W,false}(vertex_offsets, neighbors, neighbor_weights, neighbor_to_edge, num_edges_val)
        end
    else
        throw(ArgumentError("Unsupported graph type: $G"))
    end
end

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

"""Build unweighted core graph (most common case)"""
build_core_graph(edges; directed::Bool=true, kwargs...) =
    build_graph(CoreGraph, edges; directed=directed, kwargs...)

"""Build weighted graph"""
build_weighted_graph(edges, weights::AbstractVector{W}; directed::Bool=true, kwargs...) where W =
    build_graph(WeightedGraph{W}, edges; directed=directed, weights=weights, kwargs...)

# ==============================================================================
# MUTATION METHODS FOR COREGRAPH (convenience - not efficient!)
# ==============================================================================

"""
    add_vertex!(g::CoreGraph) -> Int32

Add a new isolated vertex to the graph and return its index.

**CoreGraph-specific**: O(1) operation extending vertex_offsets array.

See [`add_vertex!`](@ref) for the general interface documentation.
"""
function add_vertex!(g::CoreGraph{D}) where D
    # Simply extend vertex_offsets with the same offset as the last vertex
    # This creates an isolated vertex with no neighbors
    last_offset = g.vertex_offsets[end]
    push!(g.vertex_offsets, last_offset)

    return Int32(num_vertices(g))
end

"""
    add_edge!(g::CoreGraph, u::Integer, v::Integer) -> Int32

Add an edge from vertex u to vertex v and return the edge index.

**CoreGraph-specific**: O(degree) operation that extends CSR arrays and updates vertex offsets.
Most efficient for graphs that are mostly static after construction.

See [`add_edge!`](@ref) for the general interface documentation.
"""
function add_edge!(g::CoreGraph{D}, u::Integer, v::Integer) where D
    # Validate vertices exist
    if !has_vertex(g, u) || !has_vertex(g, v)
        throw(ArgumentError("Vertices $u and $v must exist in the graph"))
    end

    # Check if edge already exists
    if has_edge(g, u, v)
        return Int32(0)  # Edge already exists
    end

    nv = num_vertices(g)

    # For undirected graphs, add edge in both directions
    if D
        # Directed: only add u -> v
        _add_directed_edge!(g, u, v)
        g.num_edges += Int32(1)
        return g.num_edges
    else
        # Undirected: add both u -> v and v -> u, sharing the same edge index
        edge_idx = g.num_edges + Int32(1)
        _add_directed_edge!(g, u, v, edge_idx)
        _add_directed_edge!(g, v, u, edge_idx)
        g.num_edges += Int32(1)
        return edge_idx
    end
end

"""
    _add_directed_edge!(g::CoreGraph, u::Integer, v::Integer, edge_idx=nothing)

Internal helper to add a single directed edge u -> v to the CSR structure.
For undirected graphs, edge_idx specifies the undirected edge index to store.
"""
function _add_directed_edge!(g::CoreGraph{D}, u::Integer, v::Integer, edge_idx=nothing) where D
    nv = num_vertices(g)

    # Find insertion point in u's neighbor list
    u_start = g.vertex_offsets[u]
    u_end = g.vertex_offsets[u + 1] - 1

    # Find where to insert v in u's sorted neighbor list (optional optimization)
    insert_pos = u_end + 1
    for i in u_start:u_end
        if g.neighbors[i] > v
            insert_pos = i
            break
        end
    end

    # Insert the new neighbor
    insert!(g.neighbors, insert_pos, Int32(v))

    # For undirected graphs, insert the edge index
    if !D
        if edge_idx === nothing
            throw(ArgumentError("edge_idx required for undirected graphs"))
        end
        insert!(g.neighbor_to_edge, insert_pos, edge_idx)
    end

    # Update all vertex offsets after u
    for i in (u+1):(nv+1)
        g.vertex_offsets[i] += 1
    end
end

"""
    remove_vertex!(g::CoreGraph, v::Integer) -> Bool

Remove vertex v and all its incident edges from the graph.

**CoreGraph-specific**: O(V+E) operation requiring CSR array rebuilding and vertex renumbering.

See [`remove_vertex!`](@ref) for the general interface documentation.
"""
function remove_vertex!(g::CoreGraph{D}, v::Integer) where D
    if !has_vertex(g, v)
        return false
    end

    nv = num_vertices(g)

    if nv == 1
        # Removing the last vertex - create empty graph
        resize!(g.vertex_offsets, 2)
        g.vertex_offsets[1] = Int32(1)
        g.vertex_offsets[2] = Int32(1)
        resize!(g.neighbors, 0)
        if !D
            resize!(g.neighbor_to_edge, 0)
        end
        g.num_edges = Int32(0)
        return true
    end

    # Strategy: Remove all edges incident to v, then compact and renumber

    # 1. Remove all outgoing edges from v
    v_start = g.vertex_offsets[v]
    v_end = g.vertex_offsets[v + 1] - 1
    v_degree = v_end - v_start + 1

    if v_degree > 0
        # Remove from neighbors and neighbor_to_edge arrays
        deleteat!(g.neighbors, v_start:v_end)
        if !D
            deleteat!(g.neighbor_to_edge, v_start:v_end)
        end

        # Update offsets for vertices after v
        for i in (v+1):(nv+1)
            g.vertex_offsets[i] -= v_degree
        end
    end

    # 2. Remove all incoming edges to v (scan all other vertices)
    edges_removed = 0
    for u in 1:nv
        if u == v
            continue
        end

        # Find and remove v from u's neighbor list
        u_start = g.vertex_offsets[u]
        u_end = g.vertex_offsets[u + 1] - 1

        remove_positions = Int[]
        for i in u_start:u_end
            if g.neighbors[i] == v
                push!(remove_positions, i)
            end
        end

        # Remove in reverse order to maintain indices
        for pos in reverse(remove_positions)
            deleteat!(g.neighbors, pos)
            if !D
                deleteat!(g.neighbor_to_edge, pos)
            end
            edges_removed += 1

            # Update offsets for vertices after u
            for j in (u+1):(nv+1)
                g.vertex_offsets[j] -= 1
            end
        end
    end

    # 3. Renumber vertices > v to v, v+1, v+2, ...
    for i in 1:length(g.neighbors)
        if g.neighbors[i] > v
            g.neighbors[i] -= 1
        end
    end

    # 4. Remove vertex v from vertex_offsets
    deleteat!(g.vertex_offsets, v)

    # 5. Update edge count (for undirected graphs, each edge is counted once)
    if D
        g.num_edges -= Int32(edges_removed + v_degree)
    else
        # For undirected graphs, we removed each incident edge twice
        g.num_edges -= Int32((edges_removed + v_degree) ÷ 2)
    end

    return true
end

"""
    remove_edge!(g::CoreGraph, u::Integer, v::Integer) -> Bool

Remove the edge from vertex u to vertex v from the graph.

**CoreGraph-specific**: O(degree) operation removing entries from CSR arrays and updating offsets.

See [`remove_edge!`](@ref) for the general interface documentation.
"""
function remove_edge!(g::CoreGraph{D}, u::Integer, v::Integer) where D
    # Validate vertices exist
    if !has_vertex(g, u) || !has_vertex(g, v)
        return false
    end

    # Check if edge exists
    if !has_edge(g, u, v)
        return false
    end

    if D
        # Directed: only remove u -> v
        _remove_directed_edge!(g, u, v)
        g.num_edges -= Int32(1)
    else
        # Undirected: remove both u -> v and v -> u
        _remove_directed_edge!(g, u, v)
        _remove_directed_edge!(g, v, u)
        g.num_edges -= Int32(1)
    end

    return true
end

"""
    _remove_directed_edge!(g::CoreGraph, u::Integer, v::Integer)

Internal helper to remove a single directed edge u -> v from the CSR structure.
"""
function _remove_directed_edge!(g::CoreGraph{D}, u::Integer, v::Integer) where D
    nv = num_vertices(g)

    # Find the position of v in u's neighbor list
    u_start = g.vertex_offsets[u]
    u_end = g.vertex_offsets[u + 1] - 1

    remove_pos = 0
    for i in u_start:u_end
        if g.neighbors[i] == v
            remove_pos = i
            break
        end
    end

    if remove_pos == 0
        return  # Edge not found (shouldn't happen if has_edge was checked)
    end

    # Remove the neighbor
    deleteat!(g.neighbors, remove_pos)

    # For undirected graphs, also remove the edge index
    if !D
        deleteat!(g.neighbor_to_edge, remove_pos)
    end

    # Update all vertex offsets after u
    for i in (u+1):(nv+1)
        g.vertex_offsets[i] -= 1
    end
end

# ==============================================================================
# MUTATION METHODS FOR WEIGHTEDGRAPH (convenience - not efficient!)
# ==============================================================================

"""
    add_vertex!(g::WeightedGraph) -> Int32

Add a new isolated vertex to the weighted graph and return its index.

**Efficient Implementation**: O(1) operation that simply extends the vertex_offsets array.
"""
function add_vertex!(g::WeightedGraph{W,D}) where {W,D}
    # Simply extend vertex_offsets with the same offset as the last vertex
    # This creates an isolated vertex with no neighbors
    last_offset = g.vertex_offsets[end]
    push!(g.vertex_offsets, last_offset)

    return Int32(num_vertices(g))
end

"""
    add_edge!(g::WeightedGraph{W}, u::Integer, v::Integer, weight::W) -> Int32

Add a weighted edge from vertex u to vertex v and return the edge index.

**WeightedGraph-specific**: Adds edge with type-safe weight storage.
Weight type `W` must match the graph's weight type.

See [`add_edge!`](@ref) for the general interface documentation.
"""
function add_edge!(g::WeightedGraph{W,D}, u::Integer, v::Integer, weight::W) where {W,D}
    # Validate vertices exist
    if !has_vertex(g, u) || !has_vertex(g, v)
        throw(ArgumentError("Vertices $u and $v must exist in the graph"))
    end

    # Check if edge already exists
    if has_edge(g, u, v)
        return Int32(0)  # Edge already exists
    end

    # For undirected graphs, add edge in both directions
    if D
        # Directed: only add u -> v
        _add_directed_weighted_edge!(g, u, v, weight)
        g.num_edges += Int32(1)
        return g.num_edges
    else
        # Undirected: add both u -> v and v -> u, sharing the same edge index
        edge_idx = g.num_edges + Int32(1)
        _add_directed_weighted_edge!(g, u, v, weight, edge_idx)
        _add_directed_weighted_edge!(g, v, u, weight, edge_idx)
        g.num_edges += Int32(1)
        return edge_idx
    end
end

"""
    _add_directed_weighted_edge!(g::WeightedGraph, u::Integer, v::Integer, weight, edge_idx=nothing)

Internal helper to add a single directed weighted edge u -> v to the CSR structure.
"""
function _add_directed_weighted_edge!(g::WeightedGraph{W,D}, u::Integer, v::Integer, weight::W, edge_idx=nothing) where {W,D}
    nv = num_vertices(g)

    # Find insertion point in u's neighbor list
    u_start = g.vertex_offsets[u]
    u_end = g.vertex_offsets[u + 1] - 1

    # Find where to insert v in u's sorted neighbor list (optional optimization)
    insert_pos = u_end + 1
    for i in u_start:u_end
        if g.neighbors[i] > v
            insert_pos = i
            break
        end
    end

    # Insert the new neighbor and weight
    insert!(g.neighbors, insert_pos, Int32(v))
    insert!(g.weights, insert_pos, weight)

    # For undirected graphs, insert the edge index
    if !D
        if edge_idx === nothing
            throw(ArgumentError("edge_idx required for undirected graphs"))
        end
        insert!(g.neighbor_to_edge, insert_pos, edge_idx)
    end

    # Update all vertex offsets after u
    for i in (u+1):(nv+1)
        g.vertex_offsets[i] += 1
    end
end

"""
    remove_vertex!(g::WeightedGraph, v::Integer) -> Bool

Remove vertex v and all its incident edges from the weighted graph.

**WeightedGraph-specific**: O(V+E) operation with CSR rebuilding and weight array maintenance.

See [`remove_vertex!`](@ref) for the general interface documentation.
"""
function remove_vertex!(g::WeightedGraph{W,D}, v::Integer) where {W,D}
    if !has_vertex(g, v)
        return false
    end

    nv = num_vertices(g)

    if nv == 1
        # Removing the last vertex - create empty graph
        resize!(g.vertex_offsets, 2)
        g.vertex_offsets[1] = Int32(1)
        g.vertex_offsets[2] = Int32(1)
        resize!(g.neighbors, 0)
        resize!(g.weights, 0)
        if !D
            resize!(g.neighbor_to_edge, 0)
        end
        g.num_edges = Int32(0)
        return true
    end

    # Strategy: Remove all edges incident to v, then compact and renumber

    # 1. Remove all outgoing edges from v
    v_start = g.vertex_offsets[v]
    v_end = g.vertex_offsets[v + 1] - 1
    v_degree = v_end - v_start + 1

    if v_degree > 0
        # Remove from neighbors, weights, and neighbor_to_edge arrays
        deleteat!(g.neighbors, v_start:v_end)
        deleteat!(g.weights, v_start:v_end)
        if !D
            deleteat!(g.neighbor_to_edge, v_start:v_end)
        end

        # Update offsets for vertices after v
        for i in (v+1):(nv+1)
            g.vertex_offsets[i] -= v_degree
        end
    end

    # 2. Remove all incoming edges to v (scan all other vertices)
    edges_removed = 0
    for u in 1:nv
        if u == v
            continue
        end

        # Find and remove v from u's neighbor list
        u_start = g.vertex_offsets[u]
        u_end = g.vertex_offsets[u + 1] - 1

        remove_positions = Int[]
        for i in u_start:u_end
            if g.neighbors[i] == v
                push!(remove_positions, i)
            end
        end

        # Remove in reverse order to maintain indices
        for pos in reverse(remove_positions)
            deleteat!(g.neighbors, pos)
            deleteat!(g.weights, pos)
            if !D
                deleteat!(g.neighbor_to_edge, pos)
            end
            edges_removed += 1

            # Update offsets for vertices after u
            for j in (u+1):(nv+1)
                g.vertex_offsets[j] -= 1
            end
        end
    end

    # 3. Renumber vertices > v to v, v+1, v+2, ...
    for i in 1:length(g.neighbors)
        if g.neighbors[i] > v
            g.neighbors[i] -= 1
        end
    end

    # 4. Remove vertex v from vertex_offsets
    deleteat!(g.vertex_offsets, v)

    # 5. Update edge count (for undirected graphs, each edge is counted once)
    if D
        g.num_edges -= Int32(edges_removed + v_degree)
    else
        # For undirected graphs, we removed each incident edge twice
        g.num_edges -= Int32((edges_removed + v_degree) ÷ 2)
    end

    return true
end

"""
    remove_edge!(g::WeightedGraph, u::Integer, v::Integer) -> Bool

Remove the edge from vertex u to vertex v from the weighted graph.

**WeightedGraph-specific**: O(degree) operation removing CSR entries and weight data.

See [`remove_edge!`](@ref) for the general interface documentation.
"""
function remove_edge!(g::WeightedGraph{W,D}, u::Integer, v::Integer) where {W,D}
    # Validate vertices exist
    if !has_vertex(g, u) || !has_vertex(g, v)
        return false
    end

    # Check if edge exists
    if !has_edge(g, u, v)
        return false
    end

    if D
        # Directed: only remove u -> v
        _remove_directed_weighted_edge!(g, u, v)
        g.num_edges -= Int32(1)
    else
        # Undirected: remove both u -> v and v -> u
        _remove_directed_weighted_edge!(g, u, v)
        _remove_directed_weighted_edge!(g, v, u)
        g.num_edges -= Int32(1)
    end

    return true
end

"""
    _remove_directed_weighted_edge!(g::WeightedGraph, u::Integer, v::Integer)

Internal helper to remove a single directed weighted edge u -> v from the CSR structure.
"""
function _remove_directed_weighted_edge!(g::WeightedGraph{W,D}, u::Integer, v::Integer) where {W,D}
    nv = num_vertices(g)

    # Find the position of v in u's neighbor list
    u_start = g.vertex_offsets[u]
    u_end = g.vertex_offsets[u + 1] - 1

    remove_pos = 0
    for i in u_start:u_end
        if g.neighbors[i] == v
            remove_pos = i
            break
        end
    end

    if remove_pos == 0
        return  # Edge not found (shouldn't happen if has_edge was checked)
    end

    # Remove the neighbor and weight
    deleteat!(g.neighbors, remove_pos)
    deleteat!(g.weights, remove_pos)

    # For undirected graphs, also remove the edge index
    if !D
        deleteat!(g.neighbor_to_edge, remove_pos)
    end

    # Update all vertex offsets after u
    for i in (u+1):(nv+1)
        g.vertex_offsets[i] -= 1
    end
end

# ==============================================================================
# DISPLAY METHODS
# ==============================================================================

function Base.show(io::IO, g::CoreGraph{D}) where D
    direction = D ? "directed" : "undirected"
    print(io, "CoreGraph{$D} ($direction): $(num_vertices(g)) vertices, $(num_edges(g)) edges")
end

function Base.show(io::IO, g::WeightedGraph{W,D}) where {W,D}
    direction = D ? "directed" : "undirected"
    print(io, "WeightedGraph{$W,$D} ($direction): $(num_vertices(g)) vertices, $(num_edges(g)) edges")
end
