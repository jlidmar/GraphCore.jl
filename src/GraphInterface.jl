# GraphInterface.jl

# Copyright (c) 2025 Jack Lidmar
# All rights reserved.

# This software is licensed under the MIT License. See the LICENSE file for details.

# SPDX-FileCopyrightText: 2025 Jack Lidmar <jlidmar@kth.se>
# SPDX-License-Identifier: MIT

"""
Common Graph Interface Definition
================================

This module defines a minimal, extensible interface for graph types that provides:

1. **Core Graph Operations**: Basic queries (vertices, edges, neighbors)
2. **Dual Indexing System**: Support for both undirected and directed edge indexing
3. **Type Safety**: Parametric types for weights and properties
4. **External Array Support**: Stable indexing for user-managed data
5. **Graphs.jl Compatibility**: Full AbstractGraph interface implementation
6. **Performance Focus**: O(1) operations where possible, minimal overhead

## Design Philosophy

The interface separates **structure** (connectivity) from **data** (weights, properties),
allowing users to choose the optimal storage strategy for their use case:

- **Structure**: Handled by CoreGraph (CSR) or AdjGraph (adjacency lists)  
- **Weights**: Optional typed weights with directional support
- **Properties**: Optional typed vertex/edge properties
- **External Data**: User-managed arrays with stable index mapping

## Dual Indexing System

GraphCore provides two complementary edge indexing schemes:

```julia
# Undirected edge indices (1:num_edges) - for shared properties
edge_capacities = Vector{Float64}(undef, num_edges(g))
edge_idx = find_edge_index(g, u, v)  # Same for (u,v) and (v,u)

# Directed edge indices (1:num_directed_edges) - for directional properties  
edge_flows = Vector{Float64}(undef, num_directed_edges(g))
flow_uv = find_directed_edge_index(g, u, v)  # Different from (v,u)
```

All concrete graph types should implement this interface for maximum interoperability.
"""

import Graphs: Graphs, AbstractGraph

# ==============================================================================
# CORE ABSTRACT INTERFACE
# ==============================================================================

"""
    GraphInterface <: AbstractGraph{Int32}

Base abstract type for all graphs in the GraphCore ecosystem.
All vertices are indexed by Int32 integers 1, 2, ..., num_vertices(g).
"""
abstract type GraphInterface <: AbstractGraph{Int32} end

"""
    WeightedGraphInterface <: GraphInterface

Interface for weighted graphs, extending the core graph interface.
`W` is the type of edge weights, typically a numeric type.
"""
abstract type WeightedGraphInterface{W} <: GraphInterface end

"""
    PropertyGraphInterface{V,E,W} <: GraphInterface
    Abstract interface for property graphs, which support vertex and edge properties.
"""
abstract type PropertyGraphInterface{V,E} <: GraphInterface end

# Core interface methods

"""
    num_vertices(g::GraphInterface) -> Int32

Return the number of vertices in the graph.
"""
function num_vertices end

"""
    num_edges(g::GraphInterface) -> Int32

Return the number of edges in the graph.
For undirected graphs, this counts each edge once.
For directed graphs, this counts directed edges.
"""
function num_edges end

"""
    num_directed_edges(g::GraphInterface) -> Int32

Return the total number of directed edges in the graph.
- For undirected graphs: `2 * num_edges(g)`
- For directed graphs: actual count of directed edges

This count determines the size needed for directed edge property arrays.
"""
function num_directed_edges end

"""
    has_vertex(g::GraphInterface, v::Integer) -> Bool

Test whether vertex v exists in the graph.
Vertices are always integers in range 1:num_vertices(g).
"""
function has_vertex end

"""
    has_edge(g::GraphInterface, u::Integer, v::Integer) -> Bool

Test whether there is a directed edge from vertex u to vertex v.
For undirected graphs, `has_edge(g, u, v) == has_edge(g, v, u)`.
"""
function has_edge end

"""
    neighbor_indices(g::GraphInterface, v::Integer) -> view or iterator

Return an iterator over the neighbor indices of vertex v.

**For directed graphs**: Returns out-neighbors only  
**For undirected graphs**: Returns all neighbors

**Performance guarantee**: The returned iterator must support:
- Fast iteration: `for neighbor in neighbor_indices(g, v)`
- Length query: `length(neighbor_indices(g, v))` 
- Index access: `neighbor_indices(g, v)[i]` (implementation dependent)

**Memory efficiency**: Implementations should return views when possible
to avoid allocation during neighbor traversal.

# Examples
```julia
# Basic iteration
for neighbor in neighbor_indices(g, v)
    process_neighbor(neighbor)
end

# Combined with indexing for edge properties
for (i, neighbor) in enumerate(neighbor_indices(g, v))
    edge_idx = edge_index(g, v, i)        # O(1) edge index lookup
    process_edge(neighbor, edge_idx)
end
```
"""
function neighbor_indices end

"""
    is_directed_graph(g::GraphInterface) -> Bool

Return true if the graph is directed, false if undirected.
This affects the interpretation of edges and neighbor relationships.
"""
function is_directed_graph end

# ==============================================================================
# EXTENDED CORE INTERFACE (additional commonly needed methods)
# ==============================================================================

"""
    neighbor(g::GraphInterface, v::Integer, i::Integer) -> Int32

Return the i-th neighbor of vertex v in the graph g.

Default implementation provided - concrete types may override for efficiency.
"""
function neighbor end

"""
    find_edge_index(g::GraphInterface, u::Integer, v::Integer) -> Int32

Find the undirected edge index for the edge between vertices u and v of the graph g.
Returns 0 if no such edge exists.

For undirected graphs: `find_edge_index(g, u, v) == find_edge_index(g, v, u)`
For directed graphs: only finds the edge in the specified direction (u -> v)

This index is used for edge property access (shared properties) and
for indexing external arrays of size num_edges(g).
For directed graphs this is the same as `find_directed_edge_index(g, u, v)`
"""
function find_edge_index end

"""
    find_directed_edge_index(g::GraphInterface, u::Integer, v::Integer) -> Int32

Find the directed edge index for the directed edge from vertices u to v of the graph g.
Returns 0 if no such edge exists.

Always directional: `find_directed_edge_index(g, u, v) ≠ find_directed_edge_index(g, v, u)`

This index is used for directed edge weight access and other directional properties,
and for indexing external arrays of size `num_directed_edges(g)`.
"""
function find_directed_edge_index end

"""
    edge_index(g::GraphInterface, v::Integer, i::Integer) -> Int32

Get the undirected edge index for the i-th neighbor of vertex v.

This provides **O(1) conversion** from neighbor position to edge index,
enabling efficient indexing into external edge property arrays.

**Relationship**: `edge_index(g, v, i) == find_edge_index(g, v, neighbor_indices(g, v)[i])`

**Use case**: Processing neighbors with associated edge data
```julia
edge_weights = Vector{Float64}(undef, num_edges(g))
for (i, neighbor) in enumerate(neighbor_indices(g, v))
    edge_idx = edge_index(g, v, i)           # O(1) - no search needed!
    weight = edge_weights[edge_idx]          # Direct array access
    process_neighbor_with_weight(neighbor, weight)
end
```

**Index stability**: Edge indices remain stable during graph analysis,
but may be invalidated by structural modifications (add/remove operations).
"""
function edge_index end

"""
    directed_edge_index(g::GraphInterface, v::Integer, i::Integer) -> Int32

Get the directed edge index for the i-th neighbor of vertex v.

Similar to `edge_index` but for directional properties. Always provides
directional indexing even for undirected graphs.

**Key difference**: For undirected graphs:
- `edge_index(g, u, i) == edge_index(g, v, j)` if neighbors are the same edge
- `directed_edge_index(g, u, i) ≠ directed_edge_index(g, v, j)` (always directional)

This enables asymmetric properties on undirected graphs (e.g., different costs
for traversing an edge in each direction).
"""
function directed_edge_index end

"""
    edge_indices(g::GraphInterface, v::Integer) -> view or iterator

Return an iterator over the undirected edge indices for edges from vertex v.
The i-th edge index corresponds to the i-th neighbor in `neighbor_indices(g, v)`.
"""
function edge_indices end

"""
    directed_edge_indices(g::GraphInterface, v::Integer) -> view or iterator

Return an iterator over the directed edge indices for edges from vertex v.
The i-th edge index corresponds to the i-th neighbor in `neighbor_indices(g, v)``.
For directed graphs this is the same as `directed_edge_indices(g, v)`.
"""
function directed_edge_indices end

# ==============================================================================
# METHODS FOR WEIGHTED GRAPHS
# ==============================================================================

"""
    edge_weights(g::WeightedGraphInterface, v::Integer) -> view or iterator
    edge_weights(g::WeightedGraphInterface) -> view or iterator

Return weights for edges from vertex v, or all edge weights.

**Important**: Weights are **always directional**, even for undirected graphs.
This design allows asymmetric weights (e.g., different traversal costs in each direction).

**Ordering**: The i-th weight corresponds to the i-th neighbor in `neighbor_indices(g, v)`.

# Examples
```julia
# Process neighbors with weights
for (neighbor, weight) in zip(neighbor_indices(g, v), edge_weights(g, v))
    process_weighted_edge(v, neighbor, weight)
end

# More convenient combined iteration
for (neighbor, weight) in neighbor_weights(g, v)
    process_weighted_edge(v, neighbor, weight)
end
```
"""
function edge_weights end

"""
    edge_weight(g::WeightedGraphInterface, directed_edge_idx::Integer) -> W
    edge_weight(g::WeightedGraphInterface, edge::Pair{<:Integer,<:Integer}) -> W

Get the weight of the directed edge at the given directed edge index.
Uses the directional indexing system for O(1) weight lookups.
The second form allows querying by vertex pair, equivalent to `edge_weight(g, find_edge_index(g, u, v))`.
"""
function edge_weight end

"""
    neighbor_weights(g::WeightedGraphInterface, v::Integer) -> iterator

Return an iterator over `(neighbor_index, weight)` pairs for vertex v.
More efficient than separate iteration over `neighbor_indices(g, v)` and `edge_weights(g, v)`.

Usage:
```julia
    for (neighbor, weight) in neighbor_weights(g, v)
        # process neighbor and weight together
    end
```
See also: [`neighbor_indices`](@ref), [`edge_weights`](@ref)
"""
function neighbor_weights end


# ==============================================================================
# PROPERTY ACCESS AND MUTATION INTERFACE (for mutable graphs)
# ==============================================================================

"""
    vertex_property(g::GraphInterface, v::Integer) -> V

Get the property associated with vertex v.
"""
function vertex_property end

"""
    edge_property(g::PropertyGraphInterface, edge_idx::Integer) -> E

Get the property associated with edge at the given edge index.
Uses undirected edge indexing (1:num_edges).
"""
function edge_property end

"""
    set_vertex_property!(g::PropertyGraphInterface{V,E,W}, v::Integer, prop::V) -> prop

Set the property of vertex v to prop.
Only available for mutable graph types.
"""
function set_vertex_property! end

"""
    set_edge_property!(g::PropertyGraphInterface{V,E,W}, edge_idx::Integer, prop::E) -> prop

Set the property of edge at edge_idx to prop.
Only available for mutable graph types.
"""
function set_edge_property! end

"""
    Base.getindex(g::PropertyGraphInterface, v::Integer) -> property

Get the vertex property for vertex v of graph g. Equivalent to `vertex_property(g, v)`.

# Example
```julia
g = build_property_graph(edges, ["Alice", "Bob", "Charlie"], edge_props)
name = g[1]  # Returns "Alice"
```
"""
@inline Base.@propagate_inbounds Base.getindex(g::PropertyGraphInterface, v::Integer) = vertex_property(g, v)

"""
    Base.setindex!(g::PropertyGraphInterface, prop, v::Integer)

Set the vertex property for vertex v of graph g. Equivalent to `set_vertex_property!(g, prop, v)`.

# Example
```julia
g[1] = "Alice_Updated"  # Sets vertex 1's property
```
"""
@inline Base.@propagate_inbounds Base.setindex!(g::PropertyGraphInterface, prop, v::Integer) = set_vertex_property!(g, v, prop)

"""
    Base.getindex(g::PropertyGraphInterface, edge::Pair{<:Integer,<:Integer}) -> property

Get the edge property for the edge defined by the given vertex pair.
Use as `g[u => v]` to get the property of the edge from u to v.
Equivalent to `edge_property(g, find_edge_index(g, u, v))`
"""
@inline Base.@propagate_inbounds Base.getindex(g::PropertyGraphInterface, edge::Pair{<:Integer,<:Integer}) = begin
    edge_idx = find_edge_index(g, edge.first, edge.second)
    return edge_property(g, edge_idx)
end

"""
    Base.setindex!(g::PropertyGraphInterface, prop, edge::Pair{<:Integer,<:Integer})

Set the edge property for the edge defined by the given vertex pair.
Use as `g[u => v] = prop` to set the property of the edge from u to v.
Equivalent to `set_edge_property!(g, find_edge_index(g, u, v), prop)`
"""
Base.setindex!(g::PropertyGraphInterface, prop, edge::Pair{<:Integer,<:Integer}) = begin
    edge_idx = find_edge_index(g, edge.first, edge.second)
    set_edge_property!(g, edge_idx, prop)
end

"""
    vertex_properties(g::PropertyGraphInterface) -> iterator

Return an iterator over all vertex properties in order.

# Example
```julia
g = build_property_graph(edges, ["A", "B", "C"], edge_props, 3)
for (i, prop) in enumerate(vertex_properties(g))
    println("Vertex ", i, " has property: ", prop)
end
```
"""
function vertex_properties end

"""
    edge_properties(g::PropertyGraphInterface) -> iterator

Return an iterator over all edge properties in edge index order.
"""
function edge_properties end

"""
    set_edge_weight!(g::PropertyGraphInterface{V,E,W}, directed_edge_idx::Integer, weight::W) -> Nothing

Set the weight of the directed edge at directed_edge_idx to weight.
Only available for mutable graph types.
"""
function set_edge_weight! end

# ==============================================================================
# GRAPH MUTATION INTERFACE (for dynamic graphs)
# ==============================================================================

"""
    add_vertex!(g::GraphInterface, ...) -> Int32

Add a new vertex with optional properties.
Returns the index of the newly added vertex.
Only available for mutable graph types.
"""
function add_vertex! end

"""
    add_edge!(g::GraphInterface, u::Integer, v::Integer, ...) -> Int32

Add an edge from u to v with the optinal properties.
Returns the edge index of the newly added edge, or 0 if edge already exists.
For undirected graphs, this adds the edge in both directions internally.
Only available for mutable graph types.
"""
function add_edge! end

"""
    remove_vertex!(g::GraphInterface, v::Integer) -> Bool

Remove vertex v and all its incident edges.
Returns true if successful, false if vertex doesn't exist.
Only available for mutable graph types.
"""
function remove_vertex! end

"""
    remove_edge!(g::GraphInterface, u::Integer, v::Integer) -> Bool

Remove the edge from u to v.
Returns true if successful, false if edge doesn't exist.
Only available for mutable graph types.
"""
function remove_edge! end

# ==============================================================================
# DERIVED OPERATIONS (implemented in terms of core interface)
# ==============================================================================

# Return neighbor i of vertex v
@inline Base.@propagate_inbounds function neighbor(g::GraphInterface, v::Integer, i::Integer)
    neighbor_indices(g, v)[i]   # Default implementation
end

@inline Base.@propagate_inbounds function edge_index(g::GraphInterface, v::Integer, i::Integer)
    edge_indices(g, v)[i]   # Default implementation with automatic bounds checking
end

"""
    degree(g::GraphInterface, v::Integer) -> Int32

Return the degree of vertex v (number of neighbors).
"""
@inline Base.@propagate_inbounds function degree(g::GraphInterface, v::Integer)
    return Int32(length(neighbor_indices(g, v)))
end

"""
    edge_property(g::GraphInterface, u::Integer, v::Integer) -> E

Get the property of the edge between u and v.
Uses undirected edge indexing - for undirected graphs, this returns the
same property regardless of direction.
"""
@inline Base.@propagate_inbounds function edge_property(g::GraphInterface, u::Integer, v::Integer)
    edge_idx = find_edge_index(g, u, v)
    edge_idx == 0 && throw(ArgumentError("Edge ($u, $v) does not exist"))
    return @inbounds edge_property(g, edge_idx)
end

"""
    vertices(g::GraphInterface) -> UnitRange{Int}

Return a range over all vertex indices.
"""
@inline function vertices(g::GraphInterface)
    return Base.OneTo(num_vertices(g))
end

"""
    edge_indices(g::GraphInterface) -> UnitRange{Int}

Return a range over all undirected edge indices.
Suitable for sizing and indexing external edge property arrays.
"""
@inline function edge_indices(g::GraphInterface)
    return Base.OneTo(num_edges(g))
end

"""
    directed_edge_indices(g::GraphInterface) -> UnitRange{Int}

Return a range over all directed edge indices.
Suitable for sizing and indexing external directed edge property arrays.
"""
@inline function directed_edge_indices(g::GraphInterface)
    return Base.OneTo(num_directed_edges(g))
end

"""
    edges(g::GraphInterface) -> Iterator

Return an iterator over all edges in the graph.
- For directed graphs: yields (source, target) pairs for each directed edge
- For undirected graphs: yields (u, v) pairs where u ≤ v (each edge once)

# Examples
```julia
for (u, v) in edges(g)
    println("Edge from ", u, " to ", v)
end

# Collect all edges
edge_list = collect(edges(g))
```
"""
function edges(g::GraphInterface)
    return EdgeIterator(g,false)
end

"""
    all_edges(g::GraphInterface) -> Iterator

Alias for `edges(g)`. Provided for disambiguation when using multiple graph libraries.

See also: [`edges`](@ref)
"""
all_edges(g::GraphInterface) = edges(g)

"""
    all_directed_edges(g::GraphInterface) -> Iterator

Return an iterator over all directed edges in the graph.
- For directed graphs: yields (source, target) pairs for each directed edge  
- For undirected graphs: yields (u, v) and (v, u) pairs for each undirected edge

# Examples
```julia
for (u, v) in all_directed_edges(g)
    println("Directed edge from ", u, " to ", v)
end
```
"""
function all_directed_edges(g::GraphInterface)
    return EdgeIterator(g,true)
end

"""
    EdgeIterator{G<:GraphInterface,D}
    EdgeIterator(graph::G, D::Bool) -> EdgeIterator{G,D}

Unified iterator over edges in a graph, yielding (source, target) tuples.

# Type Parameters
- `G<:GraphInterface`: The graph type
- `D::Bool`: Direction mode
  - `D=false` (undirected mode): For directed graphs yields each edge once as (u,v). 
                                For undirected graphs yields each edge once in canonical form (u,v) where u ≤ v.
  - `D=true` (directed mode): For directed graphs yields each edge once as (u,v).
                              For undirected graphs yields each edge twice as (u,v) and (v,u).

See also: [`edges`](@ref), [`all_directed_edges`](@ref)
"""
struct EdgeIterator{G<:GraphInterface,D}
    graph::G
    
    EdgeIterator{G,D}(graph::G) where {G<:GraphInterface,D} = new{G,D}(graph)
    EdgeIterator(graph::G, D::Bool) where {G<:GraphInterface} = new{G,D}(graph)
end

function Base.iterate(iter::EdgeIterator{G,D}, state=nothing) where {G,D}
    g = iter.graph
    
    if state === nothing
        # Initialize: start with vertex 1, neighbor index 0
        state = (Int32(1), Int32(0))
    end
    
    vertex, neighbor_idx = state
    
    # Find next edge
    while vertex <= num_vertices(g)
        neighbors = @inbounds neighbor_indices(g, vertex)
        neighbor_idx += Int32(1)
        
        if neighbor_idx <= length(neighbors)
            @inbounds target = neighbors[neighbor_idx]  # Safe bounds check above
            
            if D  # Directed mode: yield all directed edges
                return (vertex, target), (vertex, neighbor_idx)
            else  # Undirected mode: yield canonical edges only
                if is_directed_graph(g) || vertex <= target
                    return (vertex, target), (vertex, neighbor_idx)
                else
                    # Skip this edge in undirected graphs (non-canonical), continue to next
                    continue
                end
            end
        else
            # Move to next vertex
            vertex += Int32(1)
            neighbor_idx = Int32(0)
        end
    end
    
    return nothing  # No more edges
end

@inline function Base.length(iter::EdgeIterator{G,D}) where {G,D}
    return D ? num_directed_edges(iter.graph) : num_edges(iter.graph)
end

Base.eltype(::EdgeIterator) = Tuple{Int, Int}

# Custom show method for EdgeIterator
function Base.show(io::IO, iter::EdgeIterator{G,D}) where {G,D}
    g = iter.graph
    n_edges = length(iter)
    
    # Determine iterator type and description
    if D
        iter_type = n_edges == 1 ? "directed edge" : "directed edges"
    else
        iter_type = n_edges == 1 ? "edge" : "edges"
    end
    
    # Graph type name (without parameters for cleaner display)
    graph_type = string(nameof(typeof(g)))
    direction_str = is_directed_graph(g) ? "directed" : "undirected"
    
    print(io, "EdgeIterator over $n_edges $iter_type from $graph_type ($direction_str)")
    
    # Show a preview of edges if the graph has any
    if n_edges > 0
        # Get first few edges for preview (limit to avoid performance issues)
        preview_count = min(n_edges, 3)
        preview_edges = collect(Iterators.take(iter, preview_count))
        
        edge_strs = ["($u, $v)" for (u,v) in preview_edges]
        if n_edges <= preview_count
            print(io, ": ", join(edge_strs, ", "))
        else
            print(io, ": ", join(edge_strs, ", "), ", ...")
        end
    end
end

# ==============================================================================
# DEFAULT IMPLEMENTATIONS
# ==============================================================================

@inline Base.@propagate_inbounds function edge_weight(g::WeightedGraphInterface, v::Integer, i::Integer)
    return edge_weights(g, v)[i]
end

# Edge weight access by vertex pairs
@inline Base.@propagate_inbounds function edge_weight(g::WeightedGraphInterface, e::Union{Tuple{Int,Int},Pair{Int,Int}})
    (u, v) = e
    directed_idx = find_directed_edge_index(g, u, v)
    directed_idx == 0 && throw(ArgumentError("Edge $e does not exist"))
    return @inbounds edge_weight(g, directed_idx)
end

@inline Base.@propagate_inbounds function neighbor_weights(g::WeightedGraphInterface, v::Integer)
    return Iterators.zip(neighbor_indices(g, v), edge_weights(g, v))
end

# ==============================================================================
# SHOW METHODS
# ==============================================================================

function Base.show(io::IO, g::GraphInterface)
    print(io, typeof(g), ifelse(is_directed_graph(g), " (directed)", " (undirected)"), ": ",
        num_vertices(g), " vertices, ", num_edges(g), " edges")
end

# ==============================================================================
# GRAPHS.JL COMPATIBILITY LAYER
# ==============================================================================

"""
AbstractGraph interface compatibility.
These methods delegate to our minimal interface.
"""

# Basic graph properties
@inline Graphs.nv(g::GraphInterface) = num_vertices(g)
@inline Graphs.ne(g::GraphInterface) = num_edges(g)
@inline Graphs.vertices(g::GraphInterface) = vertices(g)
@inline Base.@propagate_inbounds Graphs.has_vertex(g::GraphInterface, v::Integer) = has_vertex(g, v)
@inline Base.@propagate_inbounds Graphs.has_edge(g::GraphInterface, u::Integer, v::Integer) = has_edge(g, u, v)
@inline Graphs.is_directed(g::GraphInterface) = is_directed_graph(g)

# Neighbor queries
@inline Base.@propagate_inbounds function Graphs.outneighbors(g::GraphInterface, v::Integer)
    return neighbor_indices(g, v)
end

"""
    Graphs.inneighbors(g::GraphInterface, v::Integer) -> Vector{Int}
    
Return vertices that have outgoing edges to vertex v.

⚠️  **Performance Warning**: For directed graphs, this is O(V) operation.
    Consider using a reverse adjacency structure for frequent in-neighbor queries.
"""
@inline Base.@propagate_inbounds function Graphs.inneighbors(g::GraphInterface, v::Integer)
    @boundscheck has_vertex(g, v) || throw(ArgumentError("Vertex $v does not exist in graph"))
    if is_directed_graph(g)
        # For directed graphs, find all vertices that have v as an out-neighbor
        # return vertices(g) |> filter(u -> v in neighbor_indices(g, u)) |> collect
        # Optimized: use sizehint and avoid intermediate collections
        result = Int32[]
        sizehint!(result, 8)  # Reasonable guess for initial capacity
        
        for u in vertices(g)
            # Use view to avoid allocation, then use optimized search
            neighbors = @inbounds neighbor_indices(g, u)
            for neighbor in neighbors
                if neighbor == v
                    push!(result, u)
                    break  # Found it, no need to continue
                end
            end
        end
        return result
    else
        # For undirected graphs, in-neighbors = out-neighbors
        return @inbounds neighbor_indices(g, v)
    end
end

# Edge iteration for Graphs.jl
@inline function Graphs.edges(g::GraphInterface)
    # Use a generator instead of array comprehension to reduce allocations
    return (Graphs.SimpleEdge(u, v) for (u, v) in edges(g))
end

# Edge type
Graphs.edgetype(::GraphInterface) = Graphs.SimpleEdge{Int32}

# ==============================================================================
# INTERFACE VALIDATION HELPERS
# ==============================================================================

"""
    validate_interface(g::GraphInterface)

Validate that a graph type correctly implements the GraphInterface.

Performs comprehensive checks including:
- Method availability for the graph type
- Basic functionality (if graph is non-empty)
- Index range consistency  
- Directed vs undirected edge count relationships
- Iterator and view consistency

**Usage**: Call during development to ensure interface compliance.
```julia
g = build_core_graph(edges; directed=false)
validate_interface(g)  # Throws descriptive errors if implementation is incorrect
```
"""
function validate_interface(g::GraphInterface)
    # Core interface method availability
    required_methods = [
        (num_vertices, (typeof(g),)),
        (num_edges, (typeof(g),)),
        (num_directed_edges, (typeof(g),)),
        (has_vertex, (typeof(g), Int)),
        (has_edge, (typeof(g), Int, Int)),
        (neighbor_indices, (typeof(g), Int)),
        (is_directed_graph, (typeof(g),))
    ]
    
    for (method, signature) in required_methods
        if !hasmethod(method, signature)
            error("Graph type $(typeof(g)) must implement $method$signature")
        end
    end

    # Test basic functionality if graph is non-empty
    if num_vertices(g) > 0
        v1 = first(vertices(g))
        
        # Test neighbor iteration
        neighbors = neighbor_indices(g, v1)
        # @assert neighbors isa AbstractNeighborIterator
        @assert length(neighbors) >= 0
        
        # Test index ranges
        @assert edge_indices(g) == 1:num_edges(g)
        @assert directed_edge_indices(g) == 1:num_directed_edges(g)
        
        # Test consistency
        @assert num_directed_edges(g) >= num_edges(g)
        if !is_directed_graph(g)
            @assert num_directed_edges(g) == 2 * num_edges(g)
        end
    end
    
    println("Interface validation passed for $(typeof(g))")
end

"""
    interface_summary()

Print a summary of the interface requirements.
"""
function interface_summary()
    println("""
    GraphInterface Interface Summary:
    =======================================
    
    REQUIRED METHODS (Core Interface):
    - num_vertices(g) -> Int32
    - num_edges(g) -> Int32
    - num_directed_edges(g) -> Int32
    - has_vertex(g, v) -> Bool
    - has_edge(g, u, v) -> Bool
    - neighbor_indices(g, v) -> view or iterator
    - is_directed_graph(g) -> Bool
    - find_edge_index(g, u, v) -> Int32
    - find_directed_edge_index(g, u, v) -> Int32

    EXTENDED INTERFACE (Optional but Common):
    - edge_index(g, v, i) -> Int32
    - directed_edge_index(g, v, i) -> Int32
    - edge_indices(g, v) -> view or iterator
    - directed_edge_indices(g, v) -> view or iterator

    WEIGHTED GRAPH INTERFACE:
    - edge_weights(g, v) -> view or iterator
    - edge_weights(g) -> view or iterator  
    - edge_weight(g, directed_edge_idx) -> W
    - neighbor_weights(g, v) -> iterator

    PROPERTY GRAPH INTERFACE:
    - vertex_property(g, v) -> V
    - edge_property(g, edge_idx) -> E
    - vertex_properties(g) -> iterator
    - edge_properties(g) -> iterator

    MUTABLE GRAPH INTERFACE:
    - add_vertex!(g, ...) -> Int32
    - add_edge!(g, u, v, ...) -> Int32
    - remove_vertex!(g, v) -> Bool
    - remove_edge!(g, u, v) -> Bool

    DERIVED OPERATIONS (automatically provided):
    - degree(g, v) -> Int32
    - vertices(g) -> UnitRange{Int}
    - edge_indices(g) -> UnitRange{Int}
    - directed_edge_indices(g) -> UnitRange{Int}
    - edges(g) -> EdgeIterator{G,false}
    - all_edges(g) -> EdgeIterator{G,false}  
    - all_directed_edges(g) -> EdgeIterator{G,true}
    - neighbor(g, v, i) -> Int32

    GRAPHS.JL COMPATIBILITY (automatically provided):
    - Graphs.nv, Graphs.ne, Graphs.vertices, Graphs.edges
    - Graphs.outneighbors, Graphs.inneighbors
    - Graphs.has_vertex, Graphs.has_edge, Graphs.is_directed
    - Graphs.edgetype

    SUBMODULES (import explicitly):
    - GraphCore.Conversions: from_graphs_jl, to_graphs_jl, etc.
    - GraphCore.GraphConstruction: GraphBuilder, empty_graph, complete_graph, etc.
    """)
end

# ==============================================================================
# EDGE INDEX STABILITY GUARANTEES
# ==============================================================================

"""
Edge Index Stability Guarantees:
===============================

1. UNDIRECTED EDGE INDICES (1:num_edges):
   - Remain stable when adding vertices or edges
   - May be invalidated when removing edges or vertices
   - Used for shared edge properties (e.g., edge labels, capacities)
   - For undirected graphs: find_edge_index(g, u, v) == find_edge_index(g, v, u)

2. DIRECTED EDGE INDICES (1:num_directed_edges):  
   - Remain stable when adding vertices or edges
   - May be invalidated when removing edges or vertices
   - Used for directional edge properties (e.g., flows, costs)
   - Always directional: find_directed_edge_index(g, u, v) ≠ find_directed_edge_index(g, v, u)

3. EXTERNAL ARRAY USAGE PATTERNS:
   
   # Undirected edge properties (shared between directions)
   edge_labels = Vector{String}(undef, num_edges(g))
   edge_capacities = Vector{Float64}(undef, num_edges(g))
   
   # Directional edge properties  
   edge_flows = Vector{Float64}(undef, num_directed_edges(g))
   edge_costs = Vector{Float64}(undef, num_directed_edges(g))
   
   # Usage:
   for (u, v) in edges(g)
       edge_idx = find_edge_index(g, u, v)
       label = edge_labels[edge_idx]
       capacity = edge_capacities[edge_idx]
       
       fwd_idx = find_directed_edge_index(g, u, v)
       flow_uv = edge_flows[fwd_idx]
       cost_uv = edge_costs[fwd_idx]
       
       if !is_directed_graph(g)
           rev_idx = find_directed_edge_index(g, v, u)  
           flow_vu = edge_flows[rev_idx]
           cost_vu = edge_costs[rev_idx]
       end
   end

4. IMPLEMENTATION REQUIREMENTS:
    - Implementations should document their specific stability guarantees beyond these minimums
    - Edge indices must be consistent within a single graph instance
    - Index mappings should be deterministic and reproducible
    - Performance characteristics should be documented for index operations
"""
