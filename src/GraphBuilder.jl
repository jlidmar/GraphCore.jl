"""
    GraphBuilder{V,E,W}

Builder for constructing graphs incrementally.
Optimized for fast additions during construction phase.
"""
mutable struct GraphBuilder{V,E,W} <: GraphInterface
    # Edge storage - optimized for fast appending
    edges::Vector{Tuple{Int32,Int32}}
    edge_properties::Vector{E}
    edge_weights::Vector{W}

    # Vertex storage
    vertex_properties::Vector{V}

    # Construction settings
    directed::Bool
    max_vertex::Int32  # Track highest vertex index seen

    function GraphBuilder{V,E,W}(; directed::Bool = true) where {V,E,W}
        new{V,E,W}(
            Tuple{Int32,Int32}[],
            E[],
            W[],
            V[],
            directed,
            Int32(0)
        )
    end
end

# Convenience constructors
GraphBuilder(; directed::Bool = true) = GraphBuilder{Nothing,Nothing,Nothing}(directed=directed)
WeightedGraphBuilder(::Type{W}; directed::Bool = true) where {W} = GraphBuilder{Nothing,Nothing,W}(directed=directed)
PropertyGraphBuilder(::Type{V}, ::Type{E}; directed::Bool = true) where {V,E} = GraphBuilder{V,E,Nothing}(directed=directed)
FullGraphBuilder(::Type{V}, ::Type{E}, ::Type{W}; directed::Bool = true) where {V,E,W} = GraphBuilder{V,E,W}(directed=directed)

# Basic queries
num_vertices(builder::GraphBuilder) = Int(builder.max_vertex)
num_edges(builder::GraphBuilder) = length(builder.edges)
is_directed_graph(builder::GraphBuilder) = builder.directed

"""
    add_vertex!(builder::GraphBuilder [, prop]) -> Int32

Add a vertex with optional property. Returns the vertex index.
For non-property builders, prop should be omitted.
"""
function add_vertex!(builder::GraphBuilder{V,E,W}) where {V,E,W}
    builder.max_vertex += 1
    if V !== Nothing
        # This will require the property to be provided
        throw(ArgumentError("Property required for PropertyGraphBuilder"))
    end
    return builder.max_vertex
end

function add_vertex!(builder::GraphBuilder{V,E,W}, prop::V) where {V,E,W}
    builder.max_vertex += 1
    push!(builder.vertex_properties, prop)
    return builder.max_vertex
end

"""
    add_edge!(builder::GraphBuilder, u::Integer, v::Integer;
              edge_property=nothing, weight=nothing) -> Int32

Add an edge with optional properties and weights using keyword arguments.
Returns the edge index (1-based).

# Examples
```julia
# Basic edge
add_edge!(builder, 1, 2)

# Weighted edge
add_edge!(builder, 1, 2; weight=1.5)

# Edge with property
add_edge!(builder, 1, 2; edge_property="connection")

# Both weight and property
add_edge!(builder, 1, 2; edge_property="highway", weight=2.5)
```
"""
function add_edge!(builder::GraphBuilder{V,E,W}, u::Integer, v::Integer;
                   edge_property=nothing, weight=nothing) where {V,E,W}
    u32, v32 = Int32(u), Int32(v)

    # if max(u32, v32) > builder.max_vertex
    #     throw(ArgumentError("Vertex indices of edge ($u,$v) exceeds current max vertex $(num_vertices(builder)). Add vertices first."))
    # end

    builder.max_vertex = max(builder.max_vertex, u32, v32)

    # Add edge
    push!(builder.edges, (u32, v32))

    # Validate and add properties
    if E !== Nothing
        if edge_property === nothing
            throw(ArgumentError("edge_property required for PropertyGraphBuilder"))
        end
        # Type check
        if !(edge_property isa E)
            throw(ArgumentError("edge_property must be of type $E, got $(typeof(edge_property))"))
        end
        push!(builder.edge_properties, edge_property)
    else
        if edge_property !== nothing
            throw(ArgumentError("edge_property not supported for this builder type"))
        end
    end

    # Validate and add weights
    if W !== Nothing
        if weight === nothing
            throw(ArgumentError("weight required for WeightedGraphBuilder"))
        end
        # Type check
        if !(weight isa W)
            throw(ArgumentError("weight must be of type $W, got $(typeof(weight))"))
        end
        push!(builder.edge_weights, weight)
    else
        if weight !== nothing
            throw(ArgumentError("weight not supported for this builder type"))
        end
    end

    return Int32(length(builder.edges))
end

"""
    add_edge!(builder::GraphBuilder{Nothing,Nothing,Nothing}, u::Integer, v::Integer) -> Int32

Convenience method for basic (unweighted, no properties) graph builders.
"""
function add_edge!(builder::GraphBuilder{Nothing,Nothing,Nothing}, u::Integer, v::Integer)
    return add_edge!(builder, u, v; edge_property=nothing, weight=nothing)
end

"""
    build_graph(builder::GraphBuilder{V,E,W}) -> GraphInterface

Convert the builder to an optimized graph representation.
Automatically chooses the most appropriate graph type among:
CoreGraph, WeightedGraph{W}, PropertyGraph{G,V,E}.
"""
function build_graph(builder::GraphBuilder{V,E,W}) where {V,E,W}
    nv = num_vertices(builder)

    # Pad vertex properties if needed
    if V !== Nothing && length(builder.vertex_properties) < nv
        # Fill missing vertex properties with default (requires V to support zero())
        resize!(builder.vertex_properties, nv)
    end

    # Choose appropriate graph type and build
    if V === Nothing && E === Nothing && W === Nothing
        # Pure topology
        return build_core_graph(builder.edges; directed=builder.directed)

    elseif V === Nothing && E === Nothing && W !== Nothing
        # Weighted only
        return build_weighted_graph(builder.edges, builder.edge_weights;
                                   directed=builder.directed)

    else
        # Property graph (with or without weights)
        if W === Nothing
            # Unweighted property graph
            core = build_core_graph(builder.edges; directed=builder.directed)
        else
            # Weighted property graph
            core = build_weighted_graph(builder.edges, builder.edge_weights;
                                       directed=builder.directed)
        end

        # Ensure we have vertex properties
        vertex_props = V === Nothing ? fill(nothing, nv) : builder.vertex_properties
        edge_props = E === Nothing ? fill(nothing, length(builder.edges)) : builder.edge_properties

        return PropertyGraph(core, vertex_props, edge_props)
    end
end

"""
    build_graph(builder::GraphBuilder, ::Type{G}) -> G

Convert builder to a specific graph type G, where G can be one of:
CoreGraph, WeightedGraph{W}, PropertyGraph, AdjGraph, WeightedAdjGraph{W}, PropertyAdjGraph.
Note that the builder must be compatible with the target graph type.
Also note that the order of the arguments, builder, graph type, is opposite to the usual order.
"""
function build_graph(builder::GraphBuilder, ::Type{G}) where {G<:GraphInterface}
    if G <: CoreGraph
        return build_core_graph(builder.edges; directed=builder.directed)
    elseif G <: WeightedGraph{W} where W
        W = eltype(builder.edge_weights)
        return build_weighted_graph(builder.edges, builder.edge_weights;
            directed=builder.directed)
    elseif G <: PropertyGraph
        return PropertyGraph(build_core_graph(builder.edges; directed=builder.directed),
            builder.vertex_properties, builder.edge_properties)
    elseif G <: AdjGraph
        return build_adj_graph(builder.edges; directed=builder.directed)
    elseif G <: WeightedAdjGraph{W} where W
        W = eltype(builder.edge_weights)
        return build_weighted_adj_graph(builder.edges, builder.edge_weights;
            directed=builder.directed)
    elseif G <: PropertyAdjGraph
        return PropertyAdjGraph(build_adj_graph(builder.edges; directed=builder.directed),
            builder.vertex_properties, builder.edge_properties)
    else
        throw(ArgumentError("Unsupported target graph type: $G"))
    end
end

# Convenience functions for common patterns
"""
    build_from_function(vertex_fn::Function, edge_fn::Function, nv::Int; directed=true)

Build a graph by calling vertex_fn(i) for each vertex and edge_fn(u,v) for potential edges.
"""
function build_from_function(vertex_fn::Function, edge_fn::Function, nv::Int;
                            directed::Bool = true)
    V = typeof(vertex_fn(1))
    E = typeof(edge_fn(1, 2))

    builder = PropertyGraphBuilder{V,E}(directed=directed)

    # Add vertices
    for i in 1:nv
        add_vertex!(builder, vertex_fn(i))
    end

    # Add edges
    for u in 1:nv, v in (directed ? (1:nv) : (u:nv))
        if u != v  # No self-loops
            edge_data = edge_fn(u, v)
            if edge_data !== nothing
                add_edge!(builder, u, v, edge_data)
            end
        end
    end

    return build_graph(builder)
end