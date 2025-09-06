"""
Graph Utility Functions
=======================

Utility functions for edge format conversions and graph preprocessing.
These functions are marked as `public` but not exported, meaning they can be
accessed as `GraphCore.canonicalize_edges()` etc. without polluting the namespace.

# Edge Format Conversions

## Canonical Format
For undirected graphs, canonical format means each edge (u,v) appears once with u ≤ v.
This is the standard input format for most graph construction functions.

## Symmetric Format
Symmetric format includes both directions (u,v) and (v,u) for undirected edges.
This format is sometimes used in datasets or when converting from directed representations.

# Usage Examples

```julia
# Convert symmetric to canonical
edges = [(1,2), (2,1), (2,3), (3,2)]
canonical = GraphCore.canonicalize_edges(edges)  # [(1,2), (2,3)]

# Convert canonical to symmetric
symmetric = GraphCore.symmetrize_edges(canonical)  # [(1,2), (2,1), (2,3), (3,2)]

# With weights
weights = [1.0, 1.0, 2.0, 2.0]
canonical_edges, canonical_weights = GraphCore.canonicalize_edges(edges, weights)
```
"""

# ==============================================================================
# EDGE FORMAT CONVERSION UTILITIES
# ==============================================================================

# Public API declarations (accessible via GraphCore.function_name but not exported)
public canonicalize_edges, symmetrize_edges

"""
    canonicalize_edges(edges) -> Vector{Tuple{Int,Int}}

Convert symmetric edge format to canonical format for undirected graphs.

# Arguments
- `edges`: Vector of (u,v) tuples representing edges

# Returns
- Vector of canonical edges where u ≤ v for each edge

# Format Conversion
- **Input**: `[(1,2), (2,1), (2,3), (3,2)]` (both directions)
- **Output**: `[(1,2), (2,3)]` (canonical: u ≤ v)

Use when your input has both directions listed for undirected edges.
This removes duplicates and ensures a consistent canonical representation.

# Example
```julia
edges = [(1,2), (2,1), (2,3), (3,2), (1,3), (3,1)]
canonical = GraphCore.canonicalize_edges(edges)
# Result: [(1,2), (1,3), (2,3)]
```
"""
function canonicalize_edges(edges)
    canonical = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()

    for (u, v) in edges
        edge = u ≤ v ? (u, v) : (v, u)
        if edge ∉ seen
            push!(canonical, edge)
            push!(seen, edge)
        end
    end

    return canonical
end

"""
    canonicalize_edges(edges, weights::AbstractVector{W}) -> (canonical_edges, canonical_weights)

Convert symmetric format to canonical, keeping weights for canonical edges only.

# Arguments
- `edges`: Vector of (u,v) tuples representing edges
- `weights`: Vector of weights corresponding to edges

# Returns
- Tuple of (canonical_edges, canonical_weights)

# Behavior
When multiple weights exist for the same undirected edge, keeps the first encountered weight.
This is useful when processing datasets that list both directions with potentially different weights.

# Example
```julia
edges = [(1,2), (2,1), (2,3), (3,2)]
weights = [1.5, 1.5, 2.0, 2.1]  # Note: slight difference in last weight
canonical_edges, canonical_weights = GraphCore.canonicalize_edges(edges, weights)
# Result: edges = [(1,2), (2,3)], weights = [1.5, 2.0]
```
"""
function canonicalize_edges(edges, weights::AbstractVector{W}) where W
    if length(edges) != length(weights)
        throw(ArgumentError("edges and weights must have same length"))
    end

    canonical_edges = Tuple{Int,Int}[]
    canonical_weights = W[]
    seen = Set{Tuple{Int,Int}}()

    for (idx, (u, v)) in enumerate(edges)
        edge = u ≤ v ? (u, v) : (v, u)
        if edge ∉ seen
            push!(canonical_edges, edge)
            push!(canonical_weights, weights[idx])
            push!(seen, edge)
        end
    end

    return canonical_edges, canonical_weights
end

"""
    symmetrize_edges(edges) -> Vector{Tuple{Int,Int}}

Convert canonical format to symmetric format (both directions).

# Arguments
- `edges`: Vector of (u,v) tuples in canonical format

# Returns
- Vector of edges with both directions included

# Format Conversion
- **Input**: `[(1,2), (2,3)]` (canonical)
- **Output**: `[(1,2), (2,1), (2,3), (3,2)]` (both directions)

Use when you need to create a directed graph from undirected edges, or when
working with algorithms that expect symmetric adjacency representations.

# Example
```julia
canonical = [(1,2), (2,3), (1,3)]
symmetric = GraphCore.symmetrize_edges(canonical)
# Result: [(1,2), (2,1), (2,3), (3,2), (1,3), (3,1)]
```

# Note
Self-loops (u,u) are not duplicated to avoid redundancy.
"""
function symmetrize_edges(edges)
    symmetric = Tuple{Int,Int}[]

    for (u, v) in edges
        push!(symmetric, (u, v))
        if u != v  # Avoid duplicate self-loops
            push!(symmetric, (v, u))
        end
    end

    return symmetric
end

"""
    symmetrize_edges(edges, weights::AbstractVector{W}) -> (symmetric_edges, symmetric_weights)

Convert canonical format to symmetric, duplicating weights for both directions.

# Arguments
- `edges`: Vector of (u,v) tuples in canonical format
- `weights`: Vector of weights corresponding to edges

# Returns
- Tuple of (symmetric_edges, symmetric_weights)

# Behavior
Each weight is duplicated for both directions of the edge. This creates a symmetric
weight matrix suitable for undirected graph algorithms.

# Example
```julia
edges = [(1,2), (2,3)]
weights = [1.5, 2.0]
symmetric_edges, symmetric_weights = GraphCore.symmetrize_edges(edges, weights)
# Result:
# edges = [(1,2), (2,1), (2,3), (3,2)]
# weights = [1.5, 1.5, 2.0, 2.0]
```
"""
function symmetrize_edges(edges, weights::AbstractVector{W}) where W
    if length(edges) != length(weights)
        throw(ArgumentError("edges and weights must have same length"))
    end

    symmetric_edges = Tuple{Int,Int}[]
    symmetric_weights = W[]

    for (idx, (u, v)) in enumerate(edges)
        weight = weights[idx]

        push!(symmetric_edges, (u, v))
        push!(symmetric_weights, weight)

        if u != v
            push!(symmetric_edges, (v, u))
            push!(symmetric_weights, weight)
        end
    end

    return symmetric_edges, symmetric_weights
end
