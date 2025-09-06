# GraphCore.jl

A high-performance, type-safe graph library for Julia with a focus on efficiency, flexibility, and ease of use.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jlidmar.github.io/GraphCore.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jlidmar.github.io/GraphCore.jl/dev/)
[![Build Status](https://github.com/jlidmar/GraphCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jlidmar/GraphCore.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jlidmar/GraphCore.jl")
```

**Package Status**: Research software under active development. API may change between versions.

## Quick Start

```julia
using GraphCore

# Simple unweighted graph
edge_list = [(1,2), (2,3), (1,3)]
g = build_graph(CoreGraph, edge_list; directed=false)

# Weighted graph  
weights = [1.0, 2.0, 1.5]
wg = build_graph(WeightedGraph{Float64}, edge_list; weights=weights, directed=false)

# Convert between graph types
adj_g = AdjGraph(g)           # Convert to adjacency list for mutations
core_g = CoreGraph(adj_g)     # Convert back to CSR format

# Graph mutations (supported on CoreGraph and WeightedGraph!)
add_vertex!(g)                # Add isolated vertex
add_edge!(g, 1, 4)           # Add new edge
remove_edge!(g, 2, 3)        # Remove existing edge

# Inspect edge iterators
println(edges(g))            # Shows: "EdgeIterator over 3 edges from CoreGraph (undirected): (1,2), (1,3), (1,4)"

# Safe iteration (bounds checked by default)
for neighbor in neighbor_indices(g, 1)
    println("Connected to vertex $neighbor")
end

# High-performance iteration (bounds checks disabled)
if has_vertex(g, v)
    @inbounds for neighbor in neighbor_indices(g, v)
        # Process at ~2-3ns per access
    end
end

# Weighted neighbor iteration  
for (neighbor, weight) in neighbor_weights(wg, 1)
    println("Edge to $neighbor with weight $weight")
end
```

## Key Features

- **High Performance**: CSR storage for cache-efficient traversal (~2-3ns core operations)
- **Safety First**: Julia-idiomatic bounds checking with `@inbounds` optimization escape hatch
- **Type Safety**: Parametric types catch errors at compile time
- **Flexible Storage**: Choose between static (CoreGraph) and mutable (AdjGraph) representations  
- **In-Place Mutations**: All graph types support efficient mutations
- **Easy Conversions**: Idiomatic constructors for switching between graph types
- **Property Support**: Built-in vertex and edge properties
- **Graphs.jl Compatible**: Implements AbstractGraph interface

## Graph Types

| Type | Use Case | Mutations | Performance |
|------|----------|-----------|-------------|
| `CoreGraph` | Static analysis | ✅ Efficient | Excellent |
| `AdjGraph` | Dynamic operations | ✅ O(1) add | Good |
| `WeightedGraph` | Weighted analysis | ✅ Efficient | Excellent |
| `PropertyGraph` | With metadata | Depends on base | Varies |

## Graphs.jl Compatibility

```julia
using Graphs, GraphCore.Conversions
g = build_core_graph(edge_list; directed=false)
shortest_paths = dijkstra_shortest_paths(g, 1)  # Works directly!
```

## Documentation

See the [full documentation](docs/src/index.md) for:
- Design philosophy and architecture
- Detailed usage examples  
- Performance characteristics
- Complete API reference

## License

MIT License - See [LICENSE](LICENSE) for details.