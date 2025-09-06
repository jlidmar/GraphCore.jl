```@meta
CurrentModule = GraphCore
```

# GraphCore.jl

A high-performance, type-safe graph library for Julia with a focus on efficiency, flexibility, and ease of use.

## Design Philosophy

GraphCore.jl is designed around three key principles:

### **Performance-Oriented Design**
- **CSR Storage**: Compressed Sparse Row format for cache-efficient traversal
- **Type Specialization**: Parametric types enable compiler optimizations and zero-cost abstractions
- **Efficient Indexing**: Direct O(1) access patterns with `Int32` indexing for memory efficiency
- **Minimal Overhead**: Compact memory layouts optimized for common graph operations

### **Flexible Storage Options**
- **Multiple Representations**: CSR format (CoreGraph) for static analysis, adjacency lists (AdjGraph) for dynamic operations
- **Unified Interface**: All graph types implement the same `GraphInterface` for consistent usage
- **Property Integration**: Built-in support for type-safe vertex and edge properties
- **External Array Support**: Stable indexing schemes for user-managed data arrays

### **Type Safety and Reliability**
- **Compile-Time Checking**: Parametric types catch common errors during compilation
- **Clear Data Ownership**: Explicit separation between graph structure, weights, and properties  
- **Stable Indexing**: Consistent edge/vertex indices for reliable external array management
- **Comprehensive Validation**: Input checking and well-defined method contracts

## Architecture Overview

```
GraphCore.jl
├── GraphInterface      # Abstract interface definition
├── CoreGraph           # CSR-based graphs (static, high-performance)
├── AdjGraph            # Adjacency list graphs (dynamic, mutable)
└── GraphConstruction   # Builder patterns for graph construction
```

## Core Types and Submodules

### **Main Module: GraphCore**
```julia
using GraphCore

# Core functionality (always available)
edge_list = [(1,2), (2,3), (3,1)]
weights = [0.5, 1.1, 0.2]
vertex_props = ["A", "B", "C"]
edge_props = [1,2,3]
g = build_core_graph(edge_list; directed=true)
wg = build_weighted_graph(edge_list, weights; directed=false)
pg = build_property_graph(edge_list, vertex_props, edge_props; directed=true)
```

### **Available Submodules**
```julia
using GraphCore.Conversions        # ✅ Type conversions and Graphs.jl interop
using GraphCore.GraphConstruction  # ✅ Advanced construction patterns  
using GraphCore.Lattices           # ✅ Special graph structures
using GraphCore.PowerOfTwoLattices # ✅ Optimized power-of-two lattices

# Conversion utilities (e.g.- to and from Graphs.jl types)
graphs_g = from_graphs_jl(simple_graph)
our_graph = to_graphs_jl(core_graph)

# Graph construction helpers  
builder = WeightedGraphBuilder(Float64; directed=false)
add_edge!(builder, 1, 2, weight=0.9)
graph = build_graph(builder)

# Special graph structures
lattice_3d = HypercubicLattice{3,Int}((5, 5, 5))
lattice_2d = PowerOfTwoLattice(4,5) # size 2^4 x 2^5 = 16 x 32
```

## Core Graph Types

### **CoreGraph{D}** - High-Performance Static Graphs
```julia
# CSR (Compressed Sparse Row) storage with efficient mutations
g = build_graph(CoreGraph, edge_list; directed=true)
add_edge!(g, u, v)          # ✅ Efficient in-place mutations
remove_vertex!(g, v)        # ✅ Dynamic modifications  
# ✅ O(1) neighbor access
# ✅ Cache-efficient iteration  
# ✅ Minimal memory overhead
# ✅ Supports efficient mutations
```

### **WeightedGraph{W,D}** - Weighted Static Graphs
```julia
# CSR with parallel weight arrays, with mutation support
g = build_graph(WeightedGraph{Float64}, edge_list; weights=weights, directed=true)
add_edge!(g, u, v, weight)  # ✅ Add weighted edges efficiently
# ✅ Type-safe weights (W can be any numeric type)
# ✅ Directional weights even for undirected graphs
# ✅ Same performance as CoreGraph
# ✅ Supports efficient mutations
```

### **PropertyGraph{G,V,E}** - Graphs with Properties
```julia
# Wrapper around any base graph type
g = PropertyGraph(core_graph, vertex_props, edge_props)
# ✅ Type-safe properties
# ✅ Zero overhead delegation
# ✅ Works with any underlying graph type
# ⚠️  Mutation performance depends on underlying type
```

### **AdjGraph{D}** - Dynamic Mutable Graphs
```julia
# Vector{Vector{Int32}} storage
g = build_adj_graph(edge_list; directed=true)
add_edge!(g, u, v)      # ✅ O(1) mutations
remove_vertex!(g, v)    # ✅ Dynamic modification
# ✅ Efficient for construction and modification
# ❌ Higher memory overhead than CSR
```

## Key Features

### **Safety-First Performance**
GraphCore implements Julia-idiomatic bounds checking for all vertex and edge access:

```julia
g = build_core_graph([(1,2), (2,3), (1,3)]; directed=false)

# Safe by default - throws BoundsError for invalid vertices
neighbor_indices(g, 10)      # BoundsError: vertex 10 out of bounds [1,3]
has_edge(g, 1, 10)          # BoundsError: vertex 10 out of bounds [1,3]
find_directed_edge_index(g, 10, 1)  # BoundsError: vertex 10 out of bounds [1,3]

# Optimizable for performance-critical code
function hot_path_algorithm(g, vertices)
    for v in vertices
        # @inbounds disables bounds checking when you know access is safe
        neighbors = @inbounds neighbor_indices(g, v)
        edge_idx = @inbounds find_directed_edge_index(g, v, first(neighbors))
        # ... fast inner loop operations
    end
end

# Global bounds checking control
# --check-bounds=no    # Disable all bounds checking (unsafe but maximum speed)
# --check-bounds=yes   # Enable all bounds checking (safe, default)
```

**Safety guarantees:**
- ✅ All vertex/edge access is bounds-checked by default
- ✅ Invalid access throws descriptive `BoundsError` exceptions
- ✅ `@inbounds` provides escape hatch for performance-critical sections
- ✅ No performance penalty when bounds checking is disabled globally

### **Dual Indexing System**
GraphCore provides two complementary indexing schemes for maximum flexibility:

```julia
# Undirected edge indices (1:num_edges)
edge_capacities = Vector{Float64}(undef, num_edges(g))
edge_idx = find_edge_index(g, u, v)  # Same result for (u,v) and (v,u)

# Directed edge indices (1:num_directed_edges)  
edge_flows = Vector{Float64}(undef, num_directed_edges(g))
flow_uv = edge_flows[find_directed_edge_index(g, u, v)]
flow_vu = edge_flows[find_directed_edge_index(g, v, u)]  # Different index
```

### **Informative Edge Iteration**
GraphCore provides helpful display for edge iterators to make debugging easier:

```julia
g = build_graph(CoreGraph, [(1,2), (2,3), (1,3)]; directed=false)

# Edge iterators show helpful information
println(edges(g))
# Output: "EdgeIterator over 3 edges from CoreGraph (undirected): (1, 2), (1, 3), (2, 3)"

println(all_directed_edges(g))  
# Output: "EdgeIterator over 6 directed edges from CoreGraph (undirected): (1, 2), (1, 3), (2, 1), ..."

# Empty graphs show clearly
empty_g = build_graph(CoreGraph, Tuple{Int,Int}[]; n=3, directed=false) 
println(edges(empty_g))
# Output: "EdgeIterator over 0 edges from CoreGraph (undirected)"
```

The display shows:
- **Edge count**: Total number of edges the iterator will yield
- **Graph type**: Whether it's CoreGraph, WeightedGraph, AdjGraph, etc.
- **Directedness**: Whether the underlying graph is directed or undirected  
- **Preview**: First few edges to give you a sense of the data
- **Iterator type**: Clear distinction between `edges()` and `all_directed_edges()`

### **Seamless Conversions**
```julia
# Idiomatic Julia constructors for easy conversion
core_g = CoreGraph(adj_graph)           # Any graph → CSR format
weighted_g = WeightedGraph(core_g)      # Add weight support  
adj_g = AdjGraph(weighted_g)            # Convert to adjacency lists

# Type-safe conversions with directedness enforcement
directed_g = CoreGraph{true}(undirected_g)   # Throws error - prevents mistakes
undirected_g = CoreGraph{false}(directed_g)  # Throws error - type safety

# Construction workflow
using GraphCore.GraphConstruction
builder = WeightedGraphBuilder(Float64; directed=false)
# ... build graph incrementally ...
analysis_graph = build_graph(builder, CoreGraph)    # → CSR for analysis
mutable_graph = build_graph(builder, AdjGraph)      # → Adj lists for mutations

# Graphs.jl interop
using GraphCore.Conversions
our_graph = from_graphs_jl(simple_graph)
graphs_jl_g = to_graphs_jl(core_graph)
```

## Usage Examples

### Basic Graph Construction
```julia
using GraphCore

# Simple unweighted graph - unified interface
edge_list = [(1,2), (2,3), (1,3)]
g = build_graph(CoreGraph, edge_list; directed=false)

# Weighted graph - type-safe construction
weights = [1.0, 2.0, 1.5]
wg = build_graph(WeightedGraph{Float64}, edge_list; weights=weights, directed=false)

# Property graph
vertex_names = ["Alice", "Bob", "Charlie"]  
edge_types = ["friend", "colleague", "family"]
pg = build_graph(PropertyGraph{CoreGraph,String,String}, edge_list; 
                 vertex_properties=vertex_names, edge_properties=edge_types, directed=false)

# Easy conversions between types
adj_g = AdjGraph(g)          # Convert to adjacency list
core_g = CoreGraph(adj_g)    # Convert back to CSR
```

### Incremental Construction
```julia
# Using GraphBuilder for dynamic construction
using GraphCore.GraphConstruction

builder = WeightedGraphBuilder(Float64; directed=true)

add_vertex!(builder)  # Returns vertex index
add_edge!(builder, 1, 2; weight=1.5)
add_edge!(builder, 2, 3; weight=2.0)

# Convert to optimized storage when done
graph = build_graph(builder)
```

### Efficient Algorithms
```julia
# O(1) neighbor and weight access
function dijkstra_step(g, distances, v)
    for (neighbor, weight) in neighbor_weights(g, v)
        new_dist = distances[v] + weight
        if new_dist < distances[neighbor]
            distances[neighbor] = new_dist
        end
    end
end

# O(1) index access during neighbor iteration
for (i, neighbor) in enumerate(neighbor_indices(g, v))
    edge_idx = edge_index(g, v, i)           # O(1) - no search!
    weight = edge_weights(g, v)[i]           # O(1) - direct access
end

# External property arrays with stable indexing
edge_flows = zeros(num_directed_edges(g))
for (u, v) in edges(g)
    flow_idx = find_directed_edge_index(g, u, v)
    edge_flows[flow_idx] = compute_flow(u, v)
end
```

### Mutable Graph Operations
```julia
# All graph types support efficient mutations
g = build_graph(CoreGraph, initial_edges; directed=false)
wg = build_graph(WeightedGraph{Float64}, initial_edges; weights=weights, directed=false)

# Efficient mutations on all types
new_vertex = add_vertex!(g)               # O(1) vertex addition
edge_added = add_edge!(g, 1, new_vertex)  # Efficient edge addition
add_edge!(wg, 1, 2, 3.5)                 # Add weighted edge

# Inspect your changes with informative edge iterators
println(edges(g))  # Shows: "EdgeIterator over 4 edges from CoreGraph (undirected): (1,2), (1,3), ..."

# For property graphs - choose the right base type
pg_core = build_graph(PropertyGraph{CoreGraph,String,String}, edges; 
                      vertex_properties=vprops, edge_properties=eprops)
add_edge!(pg_core, u, v, "new_edge")     # Efficient with CoreGraph base

# Convert if you need different mutation characteristics  
pg_adj = PropertyGraph(AdjGraph(pg_core.graph), pg_core.vertex_properties, pg_core.edge_properties)
```

## Performance Best Practices

### **When to Use `@inbounds`**
Use `@inbounds` in performance-critical inner loops where you can guarantee safety:

```julia
# ✅ SAFE: After explicit validation
function safe_fast_algorithm(g, valid_vertices)
    # Pre-validate all vertices are in bounds
    @assert all(has_vertex(g, v) for v in valid_vertices)
    
    for v in valid_vertices
        # Safe to use @inbounds - we validated vertices above
        neighbors = @inbounds neighbor_indices(g, v)
        for (i, neighbor) in enumerate(neighbors)
            edge_idx = @inbounds directed_edge_index(g, v, i)
            # Fast operations...
        end
    end
end

# ✅ SAFE: With explicit bounds checking
function process_edge_if_exists(g, u, v)
    @boundscheck begin
        has_vertex(g, u) || return nothing
        has_vertex(g, v) || return nothing
    end
    # Safe to use @inbounds after validation
    return @inbounds find_directed_edge_index(g, u, v)
end

# ❌ UNSAFE: Don't use @inbounds on user input
function unsafe_example(g, user_vertex)
    # DON'T DO THIS - user_vertex might be out of bounds!
    neighbors = @inbounds neighbor_indices(g, user_vertex)  # Potential crash
end
```

### **Bounds Checking Control**
```julia
# For maximum performance in production (after thorough testing):
# julia --check-bounds=no script.jl

# For development and debugging (default):
# julia --check-bounds=yes script.jl

# Selective optimization in functions:
function hot_path(g, vertices)
    # Bounds checking here for user input
    for v in vertices
        has_vertex(g, v) || continue
        
        # @inbounds in inner loop for speed
        neighbors = @inbounds neighbor_indices(g, v)
        process_neighbors(neighbors)
    end
end
```

### **Memory and Cache Optimization**
```julia
# ✅ GOOD: Use CoreGraph for analysis workloads
analysis_graph = build_core_graph(edges; directed=false)

# ✅ GOOD: Batch operations for cache efficiency
function efficient_traversal(g)
    for v in vertices(g)
        neighbors = neighbor_indices(g, v)  # O(1) view, cache-friendly
        for neighbor in neighbors
            # Process all neighbors together
        end
    end
end

# ❌ AVOID: Repeated edge lookups
function inefficient_example(g, pairs)
    for (u, v) in pairs
        if has_edge(g, u, v)  # O(degree) lookup each time
            # This is slow for large graphs
        end
    end
end

# ✅ BETTER: Pre-compute or batch process
function efficient_edge_checking(g, pairs)
    edge_set = Set{Tuple{Int,Int}}()
    for (u, v) in edges(g)
        push!(edge_set, (u, v))
        push!(edge_set, (v, u))  # For undirected graphs
    end
    
    for (u, v) in pairs
        if (u, v) in edge_set  # O(1) lookup
            # Fast processing
        end
    end
end
```

## Performance Characteristics

| Operation | CoreGraph | WeightedGraph | AdjGraph | PropertyGraph | PropertyAdjGraph |
|-----------|-----------|---------------|----------|---------------|------------------|
| Neighbor Access | O(1), ~2ns | O(1), ~2ns | O(1), ~2ns | O(1), ~2ns | O(1), ~2ns |
| Edge Lookup | O(degree), ~3ns | O(degree), ~3ns | O(degree), ~3ns | O(degree), ~3ns | O(degree), ~3ns |
| Bounds Checking | ✅ `@boundscheck` | ✅ `@boundscheck` | ✅ `@boundscheck` | ✅ `@boundscheck` | ✅ `@boundscheck` |
| @inbounds Safe | ✅ Performance | ✅ Performance | ✅ Performance | ✅ Performance | ✅ Performance |
| Add Edge | ✅ Efficient | ✅ Efficient | O(1) | ✅ Efficient* | O(1) |
| Remove Edge | ✅ Efficient | ✅ Efficient | O(degree) | ✅ Efficient* | O(degree) |
| Add Vertex | ✅ O(1) | ✅ O(1) | O(1) | ✅ O(1)* | O(1) |
| Remove Vertex | ✅ Efficient | ✅ Efficient | O(V+E) | ✅ Efficient* | O(V+E) |
| Input Validation | ✅ Comprehensive | ✅ Comprehensive | ✅ Basic | ✅ Comprehensive | ✅ Basic |
| Memory Overhead | Minimal | +weights | +pointers | +properties | +properties+pointers |
| Cache Efficiency | Excellent | Excellent | Good | Excellent** | Good** |

*PropertyGraph inherits mutation performance from its underlying graph type.
**PropertyGraph inherits the performance characteristics of its underlying graph type.

**Performance notes:**
- Timings are median benchmarks on typical graphs (Petersen graph: 10 vertices, 15 edges)
- All operations benefit from `@inbounds` optimizations in performance-critical loops
- Bounds checking can be disabled globally with `--check-bounds=no` for maximum speed
- Edge lookup time depends on vertex degree but benefits from cache-efficient CSR layout

## Graphs.jl Compatibility

GraphCore implements the `AbstractGraph` interface for seamless integration:

```julia
using Graphs
using GraphCore
using GraphCore.Conversions

g = build_core_graph(edge_list; directed=false)

# Standard Graphs.jl functions work
nv(g)                    # Number of vertices  
ne(g)                    # Number of edges
outneighbors(g, v)       # Neighbor list
is_directed(g)           # Check directedness

# Algorithm compatibility
shortest_paths = dijkstra_shortest_paths(g, source)

# Convert to/from Graphs.jl types
simple_g = to_graphs_jl(g)
our_g = from_graphs_jl(simple_g)
```

## Design Decisions & Trade-offs

### Why Julia-Idiomatic Bounds Checking?
- **Safety First**: All vertex/edge access is safe by default with clear error messages
- **Performance When Needed**: `@inbounds` provides zero-cost optimization for validated access
- **Familiar Pattern**: Follows Julia's array indexing conventions that users already know
- **Trade-off**: Small overhead in tight loops, but eliminates silent corruption bugs

### Why CSR for CoreGraph?
- **Cache Efficiency**: Neighbors stored contiguously in memory
- **Space Efficiency**: No pointer overhead compared to adjacency lists  
- **Index Stability**: External arrays remain valid during graph analysis
- **Efficient Mutations**: Direct array manipulation preserves CSR benefits
- **Trade-off**: More complex mutation algorithms, but maintains performance characteristics

### Why Dual Indexing?
- **Flexibility**: Support both shared and directional edge properties
- **Performance**: O(1) access during iteration via `edge_index(g, v, i)`
- **Correctness**: Clear separation between undirected and directed semantics
- **Trade-off**: Slightly more complex API, but with clear documentation

### Why Multiple Graph Types?
- **Specialization**: Each type optimized for its use case
- **Composability**: PropertyGraph wraps any base type
- **Migration Path**: Easy conversion between representations
- **Trade-off**: More types to learn, but unified interface
