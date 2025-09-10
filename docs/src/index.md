```@meta
CurrentModule = GraphCore
```

# GraphCore.jl User Guide

A high-performance, type-safe graph library for Julia with a focus on efficiency, flexibility, and ease of use.

GraphCore.jl provides multiple specialized graph data structures optimized for different use cases - from static analysis with CoreGraph's compressed storage to dynamic construction with AdjGraph's adjacency lists. All graph types support efficient mutations, type-safe properties, and seamless integration with the Graphs.jl ecosystem, allowing you to use existing algorithms while benefiting from GraphCore's performance optimizations.

## Quick Start

```julia
using GraphCore

# Create your first graph
edges = [(1,2), (2,3), (1,3)]
g = build_core_graph(edges; directed=false)

# Query the graph
println("Graph has $(num_vertices(g)) vertices and $(num_edges(g)) edges")
println("Neighbors of vertex 1: $(collect(neighbor_indices(g, 1)))")
```

## Installation

```julia
using Pkg
Pkg.add("GraphCore")
```

For optional features:
```julia
# For benchmarking examples
Pkg.add("BenchmarkTools")

# For plotting examples  
Pkg.add(["Plots", "GraphRecipes"])
```

## Graph Types Overview

GraphCore provides four main graph types, each optimized for different use cases:

### **CoreGraph** - High-Performance Static Analysis
```julia
# Best for: Fast queries, analysis algorithms, memory efficiency
g = build_core_graph([(1,2), (2,3), (1,3)]; directed=false)

# Fast neighbor access
for neighbor in neighbor_indices(g, 1)
    println("Neighbor: $neighbor")
end

# Efficient mutations
add_edge!(g, 1, 4)
add_vertex!(g)
```

### **WeightedGraph** - Graphs with Edge Weights
```julia
# Best for: Algorithms needing edge weights (shortest paths, MST, etc.)
edges = [(1,2), (2,3), (1,3)]
weights = [1.0, 2.5, 1.2]
wg = build_weighted_graph(edges, weights; directed=false)

# Access weights efficiently
for (neighbor, weight) in neighbor_weights(wg, 1)
    println("Edge to $neighbor has weight $weight")
end

# Add weighted edges
add_edge!(wg, 1, 4, 3.7)
```

### **PropertyGraph** - Graphs with Custom Data
```julia
# Best for: Attaching custom data to vertices and edges
vertices = ["Alice", "Bob", "Charlie"]
edges = [(1,2), (2,3), (1,3)]
edge_types = ["friend", "colleague", "family"]

pg = build_property_graph(edges, vertices, edge_types; directed=false)

# Access properties
println("Vertex 1 is: $(vertex_property(pg, 1))")
println("Edge (1,2) type: $(edge_property(pg, 1, 2))")
```

### **AdjGraph** - Dynamic Modification
```julia
# Best for: Frequent edge additions/removals during construction
g = build_adj_graph([(1,2)]; directed=false)

# Very fast mutations
add_edge!(g, 2, 3)  # O(1)
add_edge!(g, 3, 4)  # O(1)
remove_edge!(g, 1, 2)  # O(degree)
```

## Common Workflows

### Building Graphs

**From edge lists:**
```julia
# Simple unweighted graph
edges = [(1,2), (2,3), (3,4), (4,1)]
g = build_core_graph(edges; directed=true)

# With weights
weights = [1.0, 2.0, 1.5, 0.8]
wg = build_weighted_graph(edges, weights; directed=true)
```

**Incrementally with GraphBuilder:**
```julia
using GraphCore.Builders

builder = WeightedGraphBuilder(Float64; directed=false)
add_edge!(builder, 1, 2; weight=1.5)
add_edge!(builder, 2, 3; weight=2.0)
add_edge!(builder, 1, 3; weight=1.0)

# Convert to your preferred graph type
core_g = build_graph(builder, CoreGraph)      # For analysis
adj_g = build_graph(builder, AdjGraph)        # For further modifications
```

**From other graph libraries:**
```julia
using GraphCore.Conversions
using Graphs

# From Graphs.jl
simple_g = SimpleGraph(5)
add_edge!(simple_g, 1, 2)
our_g = from_graphs_jl(simple_g)

# To Graphs.jl  
graphs_g = to_graphs_jl(our_g)
```

### Graph Analysis

**Basic queries:**
```julia
g = build_core_graph([(1,2), (2,3), (1,3)]; directed=false)

# Graph properties
println("Vertices: $(num_vertices(g))")
println("Edges: $(num_edges(g))")
println("Directed: $(is_directed(g))")

# Vertex queries
println("Degree of vertex 1: $(degree(g, 1))")
println("Neighbors: $(collect(neighbor_indices(g, 1)))")

# Edge queries  
println("Has edge (1,2): $(has_edge(g, 1, 2))")
```

**Iteration patterns:**
```julia
# Iterate over all vertices
for v in vertices(g)
    println("Processing vertex $v")
end

# Iterate over all edges
for (u, v) in edges(g)
    println("Edge: $u → $v")
end

# Iterate over neighbors efficiently
for v in vertices(g)
    for neighbor in neighbor_indices(g, v)
        println("$v is connected to $neighbor")
    end
end
```

### Unified Edge Weight Interface

GraphCore provides a **unified interface** for edge weights that works seamlessly with both weighted and unweighted graphs:

```julia
# Works on ANY graph type!
unweighted_g = build_core_graph([(1,2), (1,3)]; directed=false)
weighted_g = build_weighted_graph([(1,2), (1,3)], [1.5, 2.0]; directed=false)

# edge_weight() returns 1 for unweighted graphs, actual weight for weighted graphs
println(edge_weight(unweighted_g, 1))  # → 1 (Int32)
println(edge_weight(weighted_g, 1))    # → 1.5 (Float64)

# neighbor_weights() works universally - perfect for generic algorithms!
for (neighbor, weight) in neighbor_weights(unweighted_g, 1)
    println("Edge to $neighbor has weight $weight")  # weight = 1 for unweighted
end

for (neighbor, weight) in neighbor_weights(weighted_g, 1)
    println("Edge to $neighbor has weight $weight")  # actual weights
end

# Runtime detection for algorithm optimization
if is_weighted_graph(g)
    dijkstra_shortest_paths(g, source)  # Use actual weights
else
    bfs_shortest_paths(g, source)       # More efficient for unweighted
end
```

### Graph Modification

**Adding elements:**
```julia
g = build_core_graph([(1,2)]; directed=false)

# Add vertices and edges
new_vertex = add_vertex!(g)  # Returns vertex index
add_edge!(g, 1, new_vertex)
add_edge!(g, 2, new_vertex)

# For weighted graphs
wg = build_weighted_graph([(1,2)], [1.0]; directed=false)
add_edge!(wg, 1, 3, 2.5)  # vertex 3, weight 2.5
```

**Removing elements:**
```julia
# Remove edges
remove_edge!(g, 1, 2)

# Remove vertices (removes all connected edges)
remove_vertex!(g, 3)
```

### Converting Between Types

**Easy conversions:**
```julia
# Start with one type
adj_g = build_adj_graph([(1,2), (2,3)]; directed=false)

# Convert as needed
core_g = CoreGraph(adj_g)           # For fast analysis
weighted_g = WeightedGraph(core_g)  # Add weight support
property_g = PropertyGraph(core_g, ["A", "B", "C"], ["edge1", "edge2"])
```

**Type-safe conversions:**
```julia
directed_g = build_core_graph([(1,2)]; directed=true)

# This prevents mistakes:
# undirected_g = CoreGraph{false}(directed_g)  # Would throw error!

# Explicit conversion when you know what you're doing:
undirected_g = CoreGraph{false}(edges(directed_g))  # OK
```

## Working with Properties

**Vertex properties:**
```julia
names = ["Alice", "Bob", "Charlie"]
g = build_property_graph([(1,2), (2,3)], names, String[]; directed=false)

# Access properties
println(vertex_property(g, 1))  # "Alice"
# or equivalently
println(g[1])

# Modify properties
set_vertex_property!(g, 1, "Alice Smith") # of g[1] = ...
```

**Edge properties:**
```julia
edges = [(1,2), (2,3)]
edge_labels = ["friend", "colleague"]
g = build_property_graph(edges, String[], edge_labels; directed=false)

# Access edge properties
println(edge_property(g, 1, 2))  # "friend"
# or equivalently
println(g[1 => 2])
```

## Integration with Graphs.jl

GraphCore graphs work seamlessly with the Graphs.jl ecosystem:

```julia
using Graphs, GraphCore

g = build_core_graph([(1,2), (2,3), (1,3)]; directed=false)

# Standard Graphs.jl functions work
println("Vertices: $(nv(g))")
println("Edges: $(ne(g))")
println("Neighbors of 1: $(outneighbors(g, 1))")

# Algorithms work too
wg = build_weighted_graph([(1,2), (2,3)], [1.0, 2.0]; directed=false)
paths = dijkstra_shortest_paths(wg, 1)
```

## Safety and Performance

### Bounds Checking
GraphCore provides safe access by default with optimizations available:

```julia
g = build_core_graph([(1,2), (2,3)]; directed=false)

# Safe by default
try
    neighbor_indices(g, 10)  # BoundsError: vertex 10 out of bounds
catch e
    println("Caught: $e")
end

# Optimize when you know access is safe
function fast_algorithm(g, valid_vertices)
    for v in valid_vertices
        # @inbounds skips bounds checking for speed
        neighbors = @inbounds neighbor_indices(g, v)
        # ... fast inner loop
    end
end
```

### Edge Iterator Information
GraphCore provides helpful edge iterator displays:

```julia
g = build_core_graph([(1,2), (2,3), (1,3)]; directed=false)

println(edges(g))
# Output: "EdgeIterator over 3 edges from CoreGraph (undirected): (1, 2), (1, 3), (2, 3)"

# Shows: edge count, graph type, directedness, and preview
```

## Next Steps

- **[Design Philosophy](design.md)** - Architecture and performance details  
- **[API Reference](api.md)** - Complete function documentation
- **Examples** - Check the `examples/` directory for:
  - Performance benchmarks
  - Algorithm implementations
  - Plotting examples
  - Graphs.jl integration

## Need Help?

- Check the examples in the `examples/` directory
- Look at the comprehensive test suite in `test/`
- See the API reference for detailed function documentation
