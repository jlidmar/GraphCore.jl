# PropertyGraph.jl

# Copyright (c) 2025 Jack Lidmar
# All rights reserved.

# This software is licensed under the MIT License. See the LICENSE file for details.

# SPDX-FileCopyrightText: 2025 Jack Lidmar <jlidmar@kth.se>
# SPDX-License-Identifier: MIT

"""
Unified Property Graph Implementation
====================================

This module provides a single, universal property graph type that wraps any base graph
with typed vertex and edge properties. It automatically supports mutations when the
underlying graph type supports them.
"""

# ==============================================================================
# UNIFIED PROPERTY GRAPH TYPE
# ==============================================================================

"""
    PropertyGraph{G,V,E} <: PropertyGraphInterface{V,E}

Universal property graph that wraps any base graph type with typed vertex and edge properties.

# Type Parameters
- `G<:GraphInterface`: Base graph type (CoreGraph, WeightedGraph, AdjGraph, etc.)
- `V`: Vertex property type
- `E`: Edge property type

# Design Benefits
- **Universal wrapper**: Works with any GraphInterface implementation
- **Zero-cost delegation**: All structural operations forwarded to base graph
- **Type safety**: Compile-time property type guarantees
- **Automatic mutations**: Inherits mutation capabilities from base graph
- **Memory efficiency**: No overhead when properties are unused

# Usage Patterns
```julia
# Static analysis with CoreGraph base
core_g = build_core_graph(edges; directed=false)
vertex_labels = ["Alice", "Bob", "Charlie"]
edge_types = ["friend", "colleague", "family"]
pg = PropertyGraph(core_g, vertex_labels, edge_types)

# Access patterns
name = pg[1]                           # Vertex property via indexing
edge_type = edge_property(pg, 2)       # Edge property by index
pg[1] = "Alice Updated"                # Property modification

# Dynamic graphs with AdjGraph base
adj_g = build_adj_graph(edges; directed=false)
pg_mut = PropertyGraph(adj_g, vertex_labels, edge_types)

# Efficient mutations (when base graph supports them)
new_vertex = add_vertex!(pg_mut, "David")           # O(1) addition
edge_idx = add_edge!(pg_mut, 1, new_vertex, "buddy")  # O(1) addition

# Combined with weights
weighted_g = build_weighted_graph(edges, weights; directed=false)
pg = PropertyGraph(weighted_g, vertex_labels, edge_types)
# Now has both weights and properties available
```

# Mutation Behavior
```julia
# Mutations work when base graph supports them
adj_pg = PropertyGraph(build_adj_graph(edges), v_props, e_props)
add_edge!(adj_pg, u, v, edge_prop)  # ✅ Works - AdjGraph supports mutations

# Mutations fail gracefully when base graph doesn't support them
core_pg = PropertyGraph(build_core_graph(edges), v_props, e_props)
add_edge!(core_pg, u, v, edge_prop)  # ❌ MethodError - CoreGraph is immutable

# Property arrays are automatically maintained during mutations
original_count = length(adj_pg.edge_properties)
edge_idx = add_edge!(adj_pg, u, v, edge_prop)
@assert length(adj_pg.edge_properties) == original_count + 1
```

# Performance Notes
- **Delegation overhead**: Typically optimized away by compiler
- **Mutation performance**: Same as underlying graph type
- **Property management**: Automatic with minimal overhead
- **Memory**: Base graph memory + property arrays + small wrapper overhead
"""
struct PropertyGraph{G<:GraphInterface,V,E} <: PropertyGraphInterface{V,E}
    core::G                           # Base graph
    vertex_properties::Vector{V}      # Properties for each vertex
    edge_properties::Vector{E}        # Properties for each undirected edge

    function PropertyGraph{G,V,E}(core::G, vertex_properties::Vector{V},
                                  edge_properties::Vector{E}) where {G,V,E}
        @assert length(vertex_properties) == num_vertices(core)
        @assert length(edge_properties) == num_edges(core)
        new{G,V,E}(core, vertex_properties, edge_properties)
    end
end

# Convenience constructor
PropertyGraph(core::G, vertex_properties::Vector{V}, edge_properties::Vector{E}) where {G,V,E} =
    PropertyGraph{G,V,E}(core, vertex_properties, edge_properties)

"""
    build_graph(::Type{PropertyGraph{G,V,E}}, edges; kwargs...) where {G,V,E}

Build a property graph with vertex and edge properties. The underlying graph type G
determines performance characteristics (CoreGraph for CSR, AdjGraph for adjacency list).

# Arguments
- `edges`: Vector of (u,v) tuples/pairs representing graph edges
- `directed=true`: Whether the graph is directed
- `vertex_properties=[]`: Properties for each vertex (type V)
- `edge_properties=[]`: Properties for each edge (type E)
- `weights=[]`: Edge weights (optional)
- `validate=true`: Whether to validate inputs

# Examples
```julia
# Property graph with CoreGraph backend
g = build_graph(PropertyGraph{CoreGraph,String,String}, [(1,2), (2,3)];
                vertex_properties=["A", "B", "C"], edge_properties=["e1", "e2"])

# Property graph with AdjGraph backend for dynamic use
g = build_graph(PropertyGraph{AdjGraph,Int,Symbol}, [(1,2), (2,3)];
                vertex_properties=[1, 2, 3], edge_properties=[:a, :b])
```
"""
function build_graph(::Type{PropertyGraph{G,V,E}},
                     edges;
                     directed::Bool=true,
                     n::Integer=0,
                     weights::AbstractVector{W}=Float64[],
                     vertex_properties::AbstractVector{V}=V[],
                     edge_properties::AbstractVector{E}=E[],
                     validate::Bool=true) where {G<:GraphInterface, V, E, W<:Number}

    # Build the underlying graph first
    base_graph = build_graph(G, edges; directed=directed, n=n, weights=weights, validate=validate)

    # Validate property array lengths
    num_vertices_count = num_vertices(base_graph)
    num_edges_count = num_edges(base_graph)

    if validate
        if !isempty(vertex_properties) && length(vertex_properties) != num_vertices_count
            throw(ArgumentError("vertex_properties length ($(length(vertex_properties))) must match number of vertices ($num_vertices_count)"))
        end
        if !isempty(edge_properties) && length(edge_properties) != num_edges_count
            throw(ArgumentError("edge_properties length ($(length(edge_properties))) must match number of edges ($num_edges_count)"))
        end
    end

    # Fill with default values if empty
    v_props = isempty(vertex_properties) ? fill(zero(V), num_vertices_count) : collect(vertex_properties)
    e_props = isempty(edge_properties) ? fill(zero(E), num_edges_count) : collect(edge_properties)

    # Create PropertyGraph
    return PropertyGraph(base_graph, v_props, e_props)
end

# ==============================================================================
# STRUCTURAL DELEGATION (All graph types)
# ==============================================================================

# Delegate all structural methods to core
@inline num_vertices(g::PropertyGraph) = num_vertices(g.core)
@inline num_edges(g::PropertyGraph) = num_edges(g.core)
@inline num_directed_edges(g::PropertyGraph) = num_directed_edges(g.core)
@inline Base.@propagate_inbounds has_vertex(g::PropertyGraph, v::Integer) = has_vertex(g.core, v)
@inline Base.@propagate_inbounds has_edge(g::PropertyGraph, u::Integer, v::Integer) = has_edge(g.core, u, v)
@inline Base.@propagate_inbounds neighbor_indices(g::PropertyGraph, v::Integer) = neighbor_indices(g.core, v)
@inline is_directed_graph(g::PropertyGraph) = is_directed_graph(g.core)

# Extended neighbor access
@inline Base.@propagate_inbounds neighbor(g::PropertyGraph, v::Integer, k::Integer) = neighbor(g.core, v, k)

# Edge indexing (delegate with proper type constraints)
@inline Base.@propagate_inbounds edge_index(g::PropertyGraph, v::Integer, k::Integer) = edge_index(g.core, v, k)
@inline Base.@propagate_inbounds edge_indices(g::PropertyGraph, v::Integer) = edge_indices(g.core, v)
@inline Base.@propagate_inbounds find_edge_index(g::PropertyGraph, u::Integer, v::Integer) = find_edge_index(g.core, u, v)

# Directed edge indexing
@inline Base.@propagate_inbounds directed_edge_index(g::PropertyGraph, v::Integer, k::Integer) = directed_edge_index(g.core, v, k)
@inline Base.@propagate_inbounds directed_edge_indices(g::PropertyGraph, v::Integer) = directed_edge_indices(g.core, v)
@inline Base.@propagate_inbounds find_directed_edge_index(g::PropertyGraph, u::Integer, v::Integer) = find_directed_edge_index(g.core, u, v)

# ==============================================================================
# WEIGHT DELEGATION (Weighted graph types only)
# ==============================================================================

# Weight methods (only for weighted cores)
@inline Base.@propagate_inbounds edge_weights(g::PropertyGraph{<:WeightedGraphInterface}, v::Integer) = edge_weights(g.core, v)
@inline Base.@propagate_inbounds neighbor_weights(g::PropertyGraph{<:WeightedGraphInterface}, v::Integer) = neighbor_weights(g.core, v)
@inline Base.@propagate_inbounds edge_weight(g::PropertyGraph{<:WeightedGraphInterface}, args...) = edge_weight(g.core, args...)

# ==============================================================================
# PROPERTY ACCESS AND MUTATION
# ==============================================================================

# Property methods
@inline Base.@propagate_inbounds vertex_property(g::PropertyGraph, v::Integer) = g.vertex_properties[v]
@inline Base.@propagate_inbounds edge_property(g::PropertyGraph, edge_idx::Integer) = g.edge_properties[edge_idx]

# Property iterators
@inline vertex_properties(g::PropertyGraph) = g.vertex_properties
@inline edge_properties(g::PropertyGraph) = g.edge_properties

# Property mutation (always available for property arrays)
@inline Base.@propagate_inbounds function set_vertex_property!(g::PropertyGraph, v::Integer, prop)
    g.vertex_properties[v] = prop
    return prop
end

@inline Base.@propagate_inbounds function set_edge_property!(g::PropertyGraph, edge_idx::Integer, prop)
    g.edge_properties[edge_idx] = prop
    return prop
end

# ==============================================================================
# GRAPH MUTATION (Available when base graph supports it)
# ==============================================================================

"""
    add_vertex!(g::PropertyGraph{G,V,E}, vertex_prop::V) -> Int32

Add a new vertex to the property graph with the specified property.
Returns the index of the new vertex.

Only available when the base graph type supports `add_vertex!`.
"""
function add_vertex!(g::PropertyGraph{G,V,E}, vertex_prop::V) where {G,V,E}
    new_vertex = add_vertex!(g.core)
    push!(g.vertex_properties, vertex_prop)
    return new_vertex
end

"""
    add_edge!(g::PropertyGraph{G,V,E}, u::Integer, v::Integer, edge_prop::E) -> Int32

Add an edge from vertex u to vertex v with an edge property and return the edge index.

**PropertyGraph-specific**: Adds edge with type-safe property storage.
Inherits base graph performance plus minimal property overhead.

See [`add_edge!`](@ref) for the general interface documentation.
"""
function add_edge!(g::PropertyGraph{G,V,E}, u::Integer, v::Integer, edge_prop::E) where {G,V,E}
    edge_idx = add_edge!(g.core, u, v)
    if edge_idx > 0
        push!(g.edge_properties, edge_prop)
    end
    return edge_idx
end

"""
    add_edge!(g::PropertyGraph{<:WeightedGraphInterface,V,E}, u::Integer, v::Integer,
              weight::W, edge_prop::E) -> Int32

Add a weighted edge from u to v with the specified weight and edge property.
Returns the edge index of the newly added edge, or 0 if edge already exists.

Only available when the base graph type supports weighted `add_edge!`.
"""
function add_edge!(g::PropertyGraph{<:WeightedGraphInterface{W},V,E}, u::Integer, v::Integer,
                   weight::W, edge_prop::E) where {W,V,E}
    edge_idx = add_edge!(g.core, u, v, weight)
    if edge_idx > 0
        push!(g.edge_properties, edge_prop)
    end
    return edge_idx
end

"""
    remove_vertex!(g::PropertyGraph, v::Integer) -> Bool

Remove vertex v and all its incident edges from the property graph.

**PropertyGraph-specific**: Automatic property management with vertex renumbering.
Inherits base graph performance characteristics.

See [`remove_vertex!`](@ref) for the general interface documentation.
"""
function remove_vertex!(g::PropertyGraph, v::Integer)
    success = remove_vertex!(g.core, v)
    if success
        # Remove vertex property
        deleteat!(g.vertex_properties, v)

        # Remove edge properties for all edges incident to v
        # Note: The base graph's remove_vertex! handles the complex edge removal logic
        # We just need to synchronize our edge_properties array
        new_edge_count = num_edges(g.core)
        if length(g.edge_properties) > new_edge_count
            # Trim edge properties to match new edge count
            # The exact indices removed depend on the base graph implementation
            resize!(g.edge_properties, new_edge_count)
        end
    end
    return success
end

"""
    remove_edge!(g::PropertyGraph, u::Integer, v::Integer) -> Bool

Remove the edge from vertex u to vertex v from the property graph.

**PropertyGraph-specific**: Automatic property management with edge removal.
Inherits base graph performance characteristics.

See [`remove_edge!`](@ref) for the general interface documentation.
"""
function remove_edge!(g::PropertyGraph, u::Integer, v::Integer)
    # Get the edge index before removal for property cleanup
    edge_idx = find_edge_index(g.core, u, v)
    success = remove_edge!(g.core, u, v)

    if success && edge_idx > 0
        # Remove the corresponding edge property
        deleteat!(g.edge_properties, edge_idx)
    end
    return success
end

# ==============================================================================
# WEIGHT MUTATION (Available when base weighted graph supports it)
# ==============================================================================

"""
    set_edge_weight!(g::PropertyGraph{<:WeightedGraphInterface}, directed_edge_idx::Integer, weight) -> Nothing

Set the weight of the directed edge at directed_edge_idx to weight.
Only available when the base graph type supports weight mutation.
"""
function set_edge_weight!(g::PropertyGraph{<:WeightedGraphInterface}, args...)
    return set_edge_weight!(g.core, args...)
end

# ==============================================================================
# DISPLAY METHODS
# ==============================================================================

function Base.show(io::IO, g::PropertyGraph{G,V,E}) where {G,V,E}
    direction = is_directed_graph(g) ? "directed" : "undirected"
    print(io, "PropertyGraph{$(G.name.name),$V,$E} ($direction): ",
          "$(num_vertices(g)) vertices, $(num_edges(g)) edges")
end

# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

"""
    to_property_graph(g::GraphInterface, vertex_props::Vector{V}, edge_props::Vector{E}) -> PropertyGraph{typeof(g),V,E}

Convert any GraphInterface to a PropertyGraph with the given properties.
"""
function to_property_graph(g::GraphInterface, vertex_props::Vector{V}, edge_props::Vector{E}) where {V,E}
    return PropertyGraph{typeof(g),V,E}(g, vertex_props, edge_props)
end

"""
    to_property_graph(g::PropertyGraphInterface{V,E}) -> PropertyGraph{CoreGraph,V,E}

Convert any PropertyGraphInterface to a PropertyGraph wrapping a CoreGraph.
Preserves directedness, vertex properties, and edge properties.
"""
function to_property_graph(g::PropertyGraphInterface{V,E}) where {V,E}
    edge_list = edges(g)
    nv = num_vertices(g)
    vertex_props = collect(vertex_properties(g))
    edge_props = collect(edge_properties(g))

    return build_property_graph(edge_list, vertex_props, edge_props;
                               directed=is_directed_graph(g), n=nv)
end

"""
    to_property_graph(g::PropertyGraphInterface{V,E}, ::Type{AdjGraph}) -> PropertyGraph{AdjGraph,V,E}

Convert any PropertyGraphInterface to a PropertyGraph backed by AdjGraph.
Preserves directedness and all properties.
"""
function to_property_graph(g::PropertyGraphInterface{V,E}, ::Type{AdjGraph}) where {V,E}
    core = to_adj_graph(g)
    vertex_props = collect(vertex_properties(g))
    edge_props = collect(edge_properties(g))
    return PropertyGraph{typeof(core),V,E}(core, vertex_props, edge_props)
end

# ==============================================================================
# CONVENIENCE CONSTRUCTION FUNCTIONS
# ==============================================================================

"""
    build_property_graph(edges, vertex_properties, edge_properties; directed=true, kwargs...)

Build a PropertyGraph with CoreGraph backend.

# Arguments
- `edges`: Vector of (u,v) tuples representing graph edges
- `vertex_properties`: Vector of vertex properties
- `edge_properties`: Vector of edge properties
- `directed=true`: Whether to build a directed graph
- `kwargs...`: Additional arguments passed to underlying graph construction

# Example
```julia
edges = [(1,2), (2,3), (1,3)]
vertex_props = ["Alice", "Bob", "Charlie"]
edge_props = ["friend", "colleague", "family"]
pg = build_property_graph(edges, vertex_props, edge_props; directed=false)
```
"""
function build_property_graph(edges,
                             vertex_properties::AbstractVector{V},
                             edge_properties::AbstractVector{E};
                             directed::Bool=true, kwargs...) where {V,E}
    return build_graph(PropertyGraph{CoreGraph,V,E}, edges;
                      directed=directed,
                      vertex_properties=vertex_properties,
                      edge_properties=edge_properties, kwargs...)
end

"""
    build_property_adj_graph(edges, vertex_properties, edge_properties; directed=true, kwargs...)

Build a PropertyGraph with AdjGraph backend (supports efficient mutations).

# Arguments
- `edges`: Vector of (u,v) tuples representing graph edges
- `vertex_properties`: Vector of vertex properties
- `edge_properties`: Vector of edge properties
- `directed=true`: Whether to build a directed graph
- `kwargs...`: Additional arguments passed to underlying graph construction

# Example
```julia
edges = [(1,2), (2,3)]
vertex_props = [1, 2, 3]
edge_props = [:a, :b]
pg = build_property_adj_graph(edges, vertex_props, edge_props; directed=false)
```
"""
function build_property_adj_graph(edges,
                                  vertex_properties::AbstractVector{V},
                                  edge_properties::AbstractVector{E};
                                  directed::Bool=true, kwargs...) where {V,E}
    adj_graph = build_graph(AdjGraph, edges; directed=directed, kwargs...)
    return PropertyGraph{typeof(adj_graph),V,E}(adj_graph, vertex_properties, edge_properties)
end
