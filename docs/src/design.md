# Design Philosophy & Architecture

This page covers the design principles, architecture, and performance characteristics of GraphCore.jl.

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
└── Builders            # Builder patterns for graph construction
```

## Performance Characteristics

| Operation | CoreGraph | WeightedGraph | AdjGraph | PropertyGraph |
|-----------|-----------|---------------|----------|---------------|
| Neighbor Access | O(1) view | O(1) view | O(1) direct | O(1) + property* |
| Edge Lookup | O(degree) | O(degree) | O(degree) | O(degree) + property* |
| Bounds Checking | ✅ `@boundscheck` | ✅ `@boundscheck` | ✅ `@boundscheck` | ✅ `@boundscheck` |
| @inbounds Safe | ✅ Full support | ✅ Full support | ✅ Full support | ✅ Full support |
| Add Edge | O(1) amortized | O(1) amortized | O(1) + has_edge check | Inherits base type* |
| Remove Edge | O(degree) | O(degree) | O(degree) | Inherits base type* |
| Add Vertex | O(1) | O(1) | O(1) | O(1) + property* |
| Remove Vertex | O(V+E) rebuild | O(V+E) rebuild | O(V+E) | Inherits base type* |
| Input Validation | ✅ Comprehensive | ✅ Comprehensive | ✅ Basic | ✅ Comprehensive |
| Memory Layout | CSR (cache-friendly) | CSR + weights | Adjacency lists | Base + properties |
| Memory Overhead | Minimal | +weights array | +pointer arrays | +property arrays |

*PropertyGraph wraps any base graph type and inherits its performance characteristics, plus minimal property access overhead.

**Real-world performance notes:**
- Neighbor access: ~10-50ns per vertex depending on graph size and memory layout
- Edge operations: Times vary significantly with vertex degree and graph structure  
- All operations benefit from `@inbounds` when bounds are pre-validated
- Use `--check-bounds=no` for maximum performance in production after testing
- CSR layout provides better cache efficiency for traversal-heavy algorithms
- Property access adds minimal overhead when types are concrete

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

## When to Use Which Graph Type

### **CoreGraph{D}** - Choose When:
- ✅ You need maximum performance for queries and analysis
- ✅ Graph structure is mostly static after initial construction  
- ✅ Memory efficiency is important
- ✅ You'll be doing many neighbor traversals

### **WeightedGraph{W,D}** - Choose When:
- ✅ You need type-safe edge weights
- ✅ Performance is critical (same as CoreGraph)
- ✅ You want built-in weight management
- ✅ You need directional weights in undirected graphs

### **AdjGraph{D}** - Choose When:
- ✅ You'll be frequently adding/removing edges
- ✅ Graph construction is dynamic and unpredictable
- ✅ Flexibility is more important than memory efficiency

### **PropertyGraph{G,V,E}** - Choose When:
- ✅ You need to attach custom data to vertices/edges
- ✅ Type safety for properties is important
- ✅ You want to compose with any underlying graph type
- ✅ Zero-overhead property access is required

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
```
