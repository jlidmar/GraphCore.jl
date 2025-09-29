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
    PropertyGraphInterface{G,V,E} <: GraphInterface
    Abstract interface for property graphs, which support vertex and edge properties.

    # Type Parameters
    - `G<:GraphInterface`: Base graph type (CoreGraph, WeightedGraph, AdjGraph, etc.)
    - `V`: Vertex property type
    - `E`: Edge property type
"""
abstract type PropertyGraphInterface{G,V,E} <: GraphInterface where G<:GraphInterface end

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

"""
    is_weighted_graph(g::GraphInterface) -> Bool

Return true if the graph has explicit edge weights, false if unweighted.

This function allows algorithms to choose between weighted and unweighted implementations:
- **Weighted graphs**: Use actual edge weights for computations
- **Unweighted graphs**: Can use more efficient algorithms (e.g., BFS instead of Dijkstra)

Note that unweighted graphs still support `edge_weight`/`edge_weights` functions
(returning 1), but `is_weighted_graph` distinguishes between explicit and implicit weights.

# Examples
```julia
if is_weighted_graph(g)
    result = dijkstra_shortest_paths(g, source)  # Use weights
else
    result = bfs_shortest_paths(g, source)       # More efficient for unweighted
end

# Alternative: type-based dispatch for performance-critical code
g isa WeightedGraphInterface{Float64}  # Specific weight type
g isa WeightedGraphInterface           # Any weighted graph
```

See also: [`WeightedGraphInterface`](@ref), [`edge_weight`](@ref), [`edge_weights`](@ref)
"""
function is_weighted_graph end

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
For directed graphs: only finds the edge in the specified direction (u -> v).

This index is used for edge property access (shared properties) and
for indexing external arrays of size `num_edges(g)`.
For directed graphs this is the same as `find_directed_edge_index(g, u, v)`.
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
    edge_weights(g::GraphInterface, v::Integer) -> view or iterator
    edge_weights(g::GraphInterface) -> view or iterator

Return weights for edges from vertex v, or all edge weights.

**Performance**: O(1) per vertex for all graph types. Use `neighbor_weights(g, v)` for combined neighbor-weight iteration.

**Important**: Weights are always directional, even for undirected graphs.
The i-th weight corresponds to the i-th neighbor in `neighbor_indices(g, v)`.

**For weighted graphs**: Returns actual stored weights.
**For unweighted graphs**: Returns 1s (mathematical consistency).
"""
function edge_weights end

"""
    edge_weight(g::GraphInterface, v::Integer, i::Integer) -> W
    edge_weight(g::GraphInterface, directed_edge_idx::Integer) -> W
    edge_weight(g::GraphInterface, edge::Pair{<:Integer,<:Integer}) -> W

Access edge weights by different methods:
1. Direct access: `edge_weight(g, v, i)` gets weight to i-th neighbor of v (O(1))
2. By directed edge index: `edge_weight(g, directed_idx)` (⚠️ O(V) for AdjGraph, O(1) for CoreGraph)
3. By pair of vertices: `edge_weight(g, u => v)` (uses directed index internally) (not the most efficient)

For iterations perefer `edge_weights(g, v)` or `neighbor_weights(g, v)` (most efficient)

**Important**: Edge weights use directional indexing (always), unlike edge properties which follow graph directedness.

**For weighted graphs**: Returns actual stored weights.
**For unweighted graphs**: Returns 1 (mathematical consistency).
"""
function edge_weight end

"""
    neighbor_weights(g::GraphInterface, v::Integer) -> iterator

Returns `(neighbor, weight)` pairs for vertex v.

Most efficient pattern for graph algorithms. O(1) per neighbor for all graph types.

```julia
for (neighbor, weight) in neighbor_weights(g, v)
    # Process neighbor and weight together
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

Get edge property by edge index (1:num_edges).
Edge properties follow graph directedness (shared for undirected graphs).
"""
function edge_property end

"""
    set_vertex_property!(g::PropertyGraphInterface{G,V,E}, v::Integer, prop::V) -> prop

Set the property of vertex v to prop.
Only available for mutable graph types.
"""
function set_vertex_property! end

"""
    set_edge_property!(g::PropertyGraphInterface{G,V,E}, edge_idx::Integer, prop::E) -> prop

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
    set_edge_weight!(g::WeightedGraphInterface{W}, directed_edge_idx::Integer, weight::W) -> weight
    set_edge_weight!(g::WeightedGraphInterface{W}, v::Integer, k::Integer, weight::W) -> weight

Set the weight of the directed edge at directed_edge_idx to weight.
The second form sets the weight of the edge from vertex v to its k-th neighbor.
Only available for graph types with weighted base graphs.
"""
function set_edge_weight! end

# ==============================================================================
# GRAPH MUTATION INTERFACE (for dynamic graphs)
# ==============================================================================

"""
    add_vertex!(g::GraphInterface, [vertex_property]) -> Int32

Add a new isolated vertex to the graph with optional vertex property.

# Returns
- Index (Int32) of the newly added vertex

# Behavior
- **New vertex**: Has no neighbors initially (degree 0)
- **Index assignment**: Gets the next available vertex index (typically `num_vertices(g) + 1`)
- **Index stability**: All existing vertex/edge indices remain valid
- **Connections**: Use `add_edge!` to connect the new vertex to others

# Type-Specific Performance
- `CoreGraph`: O(1) - extends vertex arrays
- `AdjGraph`: O(1) - pushes to vertex arrays
- `WeightedGraph`: O(1) - same as base type
- `PropertyGraph`: O(1) - same as base type + property storage

# Property Handling
- **No properties**: `add_vertex!(g)` - adds vertex without properties
- **With properties**: `add_vertex!(g, prop)` - property type must match graph's vertex property type

# Examples
```julia
# Add vertex without properties
g = build_graph(AdjGraph, [(1,2)]; directed=false)
new_v = add_vertex!(g)  # Returns 3
add_edge!(g, 1, new_v)  # Connect to existing vertex

# Add vertex with property
pg = build_graph(PropertyGraph{AdjGraph,String,Nothing}, [(1,2)];
                 directed=false, vertex_properties=["Alice", "Bob"])
new_v = add_vertex!(pg, "Charlie")  # Returns 3 with property
```

# Availability
Only available for mutable graph types (`AdjGraph`, `WeightedAdjGraph`, `PropertyGraph` with mutable base).
"""
function add_vertex! end

"""
    add_edge!(g::GraphInterface, u::Integer, v::Integer, [weight], [edge_property]) -> Int32

Add an edge from vertex `u` to vertex `v` with optional weight and/or edge property.

# Returns
- Edge index (Int32) of the newly added edge
- `0` if the edge already exists (no modification performed)

# Behavior
- **Directed graphs**: Adds only u→v edge
- **Undirected graphs**: Adds both u→v and v→u internally (same edge index)
- **Index assignment**: New edges get the next available edge index
- **Duplicate detection**: Checks if edge exists before adding
- **Index stability**: Existing edge/vertex indices remain valid after addition

# Type-Specific Performance
- `CoreGraph`: O(degree) - extends CSR arrays, efficient for static analysis
- `AdjGraph`: O(1) amortized - vector push operations, efficient for dynamic graphs
- `WeightedGraph`: Same as base type + weight storage
- `PropertyGraph`: Same as base type + property storage

# Error Conditions
- Throws `BoundsError` if either vertex doesn't exist
- For weighted graphs, weight type must match graph's weight type
- For property graphs, property type must match graph's edge property type

# Examples
```julia
# Basic edge addition
g = build_graph(CoreGraph, [(1,2)]; directed=false)
edge_idx = add_edge!(g, 1, 3)  # Returns edge index

# Weighted edge
wg = build_graph(WeightedGraph, [(1,2,1.0)]; directed=true)
edge_idx = add_edge!(wg, 2, 3, 2.5)  # Add weighted edge

# Property edge
pg = build_graph(PropertyGraph{AdjGraph,Nothing,String}, [(1,2)];
                 directed=false, edge_properties=["friendship"])
edge_idx = add_edge!(pg, 1, 3, "colleague")  # Add edge with property
```

# Availability
Only available for mutable graph types (`AdjGraph`, `WeightedAdjGraph`, `PropertyGraph` with mutable base).
"""
function add_edge! end

"""
    remove_vertex!(g::GraphInterface, v::Integer) -> Bool

Remove vertex `v` and all its incident edges from the graph.

# Returns
- `true` if vertex was successfully removed
- `false` if vertex doesn't exist (no modification performed)

# Behavior
- **Edge removal**: All edges incident to vertex `v` are automatically removed
- **Index compaction**: Vertices with indices > `v` are renumbered to fill the gap
- **Property handling**: Vertex and edge properties are automatically maintained
- **Index invalidation**: Vertex indices > `v` become invalid after removal

# Type-Specific Performance
- `CoreGraph`: O(V+E) - rebuilds CSR arrays with compacted indices
- `AdjGraph`: O(V+E) - updates all vertex references and compacts arrays
- `WeightedGraph`: Same as base type + weight array maintenance
- `PropertyGraph`: Same as base type + property array maintenance

# Index Management
After removing vertex `v`, all vertices with indices > `v` get decremented by 1:
- Vertex `v+1` becomes vertex `v`
- Vertex `v+2` becomes vertex `v+1`, etc.
- External arrays indexed by vertex must be updated accordingly

# Error Conditions
- No error for non-existent vertices (returns `false`)
- For property graphs, properties are removed automatically

# Examples
```julia
# Basic vertex removal
g = build_graph(AdjGraph, [(1,2), (2,3), (1,3)]; directed=false)
success = remove_vertex!(g, 2)  # Remove vertex 2, returns true
# Vertex 3 is now vertex 2, edges (1,3) becomes (1,2)

# Property graph vertex removal
pg = build_graph(PropertyGraph{AdjGraph,String,Nothing}, [(1,2), (2,3)];
                 directed=false, vertex_properties=["Alice", "Bob", "Charlie"])
remove_vertex!(pg, 2)  # Removes "Bob", "Charlie" becomes vertex 2
```

# Availability
Only available for mutable graph types (`AdjGraph`, `WeightedAdjGraph`, `PropertyGraph` with mutable base).

# ⚠️ Important
This operation invalidates vertex indices > `v`. Update any external data structures accordingly.
"""
function remove_vertex! end

"""
    remove_edge!(g::GraphInterface, u::Integer, v::Integer) -> Bool

Remove the edge from vertex `u` to vertex `v`.

# Returns
- `true` if edge was successfully removed
- `false` if edge doesn't exist (no modification performed)

# Behavior
- **Directed graphs**: Removes only the u→v edge
- **Undirected graphs**: Removes both u→v and v→u edges (same operation)
- **Edge properties**: Automatically removed for property graphs
- **Index stability**: Vertex indices remain unchanged, edge indices may be affected

# Type-Specific Performance
- `CoreGraph`: O(degree) - updates CSR arrays and shifts subsequent edges
- `AdjGraph`: O(degree) - searches neighbor list and removes entry
- `WeightedGraph`: Same as base type + weight removal
- `PropertyGraph`: Same as base type + property removal

# Edge Index Effects
- **CoreGraph/WeightedGraph**: Edge indices may change for edges after the removed edge
- **AdjGraph/WeightedAdjGraph**: Edge indices remain stable
- **PropertyGraph**: Inherits behavior from base graph type

# Error Conditions
- No error for non-existent edges (returns `false`)
- Throws `BoundsError` if either vertex doesn't exist

# Examples
```julia
# Basic edge removal
g = build_graph(AdjGraph, [(1,2), (2,3), (1,3)]; directed=false)
success = remove_edge!(g, 1, 2)  # Returns true
@assert !has_edge(g, 1, 2) && !has_edge(g, 2, 1)  # Both directions removed

# Directed graph edge removal
dg = build_graph(AdjGraph, [(1,2), (2,1)]; directed=true)
remove_edge!(dg, 1, 2)  # Removes only 1→2
@assert !has_edge(dg, 1, 2) && has_edge(dg, 2, 1)  # 2→1 still exists

# Property graph edge removal
pg = build_graph(PropertyGraph{AdjGraph,Nothing,String}, [(1,2), (2,3)];
                 directed=false, edge_properties=["friend", "colleague"])
remove_edge!(pg, 1, 2)  # Removes edge and "friend" property
```

# Availability
Only available for mutable graph types (`AdjGraph`, `WeightedAdjGraph`, `PropertyGraph` with mutable base).
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

# Default implementations for unweighted graphs
@inline is_weighted_graph(::GraphInterface) = false
@inline is_weighted_graph(::WeightedGraphInterface) = true

# Default edge weight methods for unweighted graphs (excludes WeightedGraphInterface)
@inline edge_weight(g::GraphInterface, ::Integer) = one(Int32)
@inline edge_weight(g::GraphInterface, ::Integer, ::Integer) = one(Int32)
@inline edge_weight(g::GraphInterface, ::Union{Tuple{<:Integer,<:Integer},Pair{<:Integer,<:Integer}}) = one(Int32)
@inline edge_weights(g::GraphInterface, v::Integer) = Iterators.repeated(one(Int32), degree(g, v))
@inline edge_weights(g::GraphInterface) = Iterators.repeated(one(Int32), num_directed_edges(g))

# Default implementations for weighted graphs (may be overridden for efficiency)
@inline Base.@propagate_inbounds function edge_weight(g::WeightedGraphInterface, v::Integer, i::Integer)
    return edge_weights(g, v)[i]
end
@inline Base.@propagate_inbounds function edge_weight(g::WeightedGraphInterface, directed_indx::Integer)
    return edge_weights(g)[directed_indx]
end

# Edge weight access by vertex pairs
@inline Base.@propagate_inbounds function edge_weight(g::WeightedGraphInterface, e::Union{Tuple{<:Integer,<:Integer},Pair{<:Integer,<:Integer}})
    (u, v) = e
    directed_idx = find_directed_edge_index(g, u, v)
    directed_idx == 0 && throw(ArgumentError("Edge $e does not exist"))
    return @inbounds edge_weight(g, directed_idx)
end

@inline Base.@propagate_inbounds function neighbor_weights(g::GraphInterface, v::Integer)
    return Iterators.zip(neighbor_indices(g, v), edge_weights(g, v))
end

@inline Base.@propagate_inbounds function set_edge_weight!(g::WeightedGraphInterface{W}, directed_edge_idx::Integer, weight::W) where {W<:Number}
    return edge_weights(g)[directed_edge_idx] = weight
end

@inline Base.@propagate_inbounds function set_edge_weight!(g::WeightedGraphInterface{W}, v::Integer, k::Integer, weight::W) where {W<:Number}
    return edge_weights(g,v)[k] = weight
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
    - is_weighted_graph(g) -> Bool

    EDGE WEIGHT INTERFACE (All Graphs):
    - edge_weights(g, v) -> view or iterator
    - edge_weights(g) -> view or iterator
    - edge_weight(g, directed_edge_idx) -> W
    - neighbor_weights(g, v) -> iterator

    WEIGHTED GRAPH INTERFACE (Specialized):
    - set_edge_weight!(g, directed_edge_idx, weight) -> weight
    - set_edge_weight!(g, v, k, weight) -> weight
    - Additional optimized implementations of above functions

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
    - GraphCore.Builders: GraphBuilder, empty_graph, complete_graph, etc.
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
