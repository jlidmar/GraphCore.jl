"""
Lattice Graph Structures
=======================

Specialized graph implementations for regular lattice structures.
Optimized for spatial/grid-based computations with mathematical neighbor lookup.
"""
module Lattices

using StaticArrays: MVector, SVector

import ..GraphCore: GraphInterface, num_vertices, num_edges, num_directed_edges, has_vertex, has_edge, degree
import ..GraphCore: neighbor_indices, edge_indices, directed_edge_indices, neighbor, edge_index, directed_edge_index
import ..GraphCore: is_directed_graph, find_edge_index, find_directed_edge_index

export HypercubicLattice
export is_periodic
export lattice_size, lattice_dimension, coord_to_vertex, vertex_to_coord
export lattice_neighbors, lattice_distance

# ==============================================================================
# HYPERCUBIC LATTICE GRAPH
# ==============================================================================

"""
    HypercubicLattice{D,T} <: GraphInterface

A D-dimensional hypercubic lattice graph with side length of type T.

# Type Parameters
- `D::Int`: Dimension of the lattice (1D=line, 2D=grid, 3D=cube, etc.)
- `T<:Integer`: Type for lattice size/coordinates

# Storage
Uses mathematical coordinate mapping instead of explicit edge storage.
Memory usage: O(1) regardless of lattice size!

# Coordinate System
- Vertices are numbered 1 to prod(sizes)
- Coordinates are 0-indexed: (0,0,...,0) to (size₁-1, size₂-1, ..., sizeD-1)
- Periodic boundary conditions optional

# Examples
```julia
# 2D grid: 10×10
lattice_2d = HypercubicLattice{2,Int}((10, 10))

# 3D cube: 5×5×5
lattice_3d = HypercubicLattice{3,Int}((5, 5, 5))

# 1D chain: 100 vertices
lattice_1d = HypercubicLattice{1,Int}((100,))
```
"""
struct HypercubicLattice{D,T<:Integer} <: GraphInterface
    sizes::NTuple{D,T}           # Size in each dimension
    periodic::NTuple{D,Bool}     # Periodic boundary conditions per dimension
    total_vertices::Int32        # Cache for num_vertices
    total_edges::Int32          # Cache for num_edges

    function HypercubicLattice{D,T}(sizes::NTuple{D,T};
                                   periodic::NTuple{D,Bool} = ntuple(i -> false, D)) where {D,T<:Integer}

        if D < 1
            throw(ArgumentError("Dimension must be positive, got $D"))
        end

        if any(s ≤ 0 for s in sizes)
            throw(ArgumentError("All sizes must be positive, got $sizes"))
        end

        total_vertices = Int32(prod(sizes))

        # Calculate total edges
        total_edges = Int32(0)
        for dim in 1:D
            edges_in_dim = prod(sizes) ÷ sizes[dim]  # Number of "slices" perpendicular to dim
            if periodic[dim]
                edges_in_dim *= sizes[dim]  # All connections in periodic case
            else
                edges_in_dim *= (sizes[dim] - 1)  # One less connection per slice
            end
            total_edges += edges_in_dim
        end

        new{D,T}(sizes, periodic, total_vertices, total_edges)
    end
end

# Convenience constructors
HypercubicLattice(sizes::NTuple{D,T}; kwargs...) where {D,T} = HypercubicLattice{D,T}(sizes; kwargs...)
HypercubicLattice(sizes::Vararg{T,D}; kwargs...) where {D,T} = HypercubicLattice{D,T}(sizes; kwargs...)

# Convenience constructors for common cases
"""
    Grid2D(width, height; periodic=(false, false)) -> HypercubicLattice{2,Int}

Create a 2D grid lattice.
"""
Grid2D(width::Integer, height::Integer; periodic=(false, false)) =
    HypercubicLattice{2,Int}((width, height); periodic=periodic)

"""
    Grid3D(width, height, depth; periodic=(false, false, false)) -> HypercubicLattice{3,Int}

Create a 3D cubic lattice.
"""
Grid3D(width::Integer, height::Integer, depth::Integer; periodic=(false, false, false)) =
    HypercubicLattice{3,Int}((width, height, depth); periodic=periodic)

"""
    Chain1D(length; periodic=false) -> HypercubicLattice{1,Int}

Create a 1D chain lattice.
"""
Chain1D(length::Integer; periodic=false) =
    HypercubicLattice{1,Int}((length,); periodic=(periodic,))

# ==============================================================================
# CORE INTERFACE IMPLEMENTATION
# ==============================================================================

"""Number of vertices in the lattice."""
num_vertices(g::HypercubicLattice) = g.total_vertices

"""Number of edges in the lattice."""
num_edges(g::HypercubicLattice) = g.total_edges

"""Number of directed edges (twice the number of undirected for lattices)."""
num_directed_edges(g::HypercubicLattice) = 2 * g.total_edges

"""Lattices are always undirected."""
is_directed_graph(g::HypercubicLattice) = false

"""Check if vertex exists."""
function has_vertex(g::HypercubicLattice, v::Integer)
    return 1 ≤ v ≤ num_vertices(g)
end

"""Check if edge exists between two vertices."""
function has_edge(g::HypercubicLattice{D,T}, u::Integer, v::Integer) where {D,T}
    if !has_vertex(g, u) || !has_vertex(g, v)
        return false
    end

    coord_u = vertex_to_coord(g, u)
    coord_v = vertex_to_coord(g, v)

    # Check if they differ by exactly 1 in exactly one dimension
    diff_count = 0
    diff_dim = 0

    for dim in 1:D
        diff = abs(coord_u[dim] - coord_v[dim])

        if g.periodic[dim]
            # Handle periodic boundary: distance is min(diff, size - diff)
            diff = min(diff, g.sizes[dim] - diff)
        end

        if diff == 1
            diff_count += 1
            diff_dim = dim
        elseif diff > 1
            return false  # Too far apart
        end
    end

    return diff_count == 1
end

#===============================================================================

"""Get neighbors of a vertex using mathematical computation."""
function neighbor_indices(g::HypercubicLattice{D,T}, v::Integer) where {D,T}
    if !has_vertex(g, v)
        throw(BoundsError("Vertex $v out of bounds for lattice with $(num_vertices(g)) vertices"))
    end

    coord = vertex_to_coord(g, v)
    neighbors = Int32[]

    # Check each dimension
    for dim in 1:D
        # Forward neighbor
        new_coord_forward = ntuple(i -> i == dim ? coord[i] + 1 : coord[i], D)
        if coord[dim] < g.sizes[dim] - 1
            # Normal case: within bounds
            push!(neighbors, coord_to_vertex(g, new_coord_forward))
        elseif g.periodic[dim]
            # Periodic case: wrap around
            new_coord_wrap = ntuple(i -> i == dim ? T(0) : coord[i], D)
            push!(neighbors, coord_to_vertex(g, new_coord_wrap))
        end

        # Backward neighbor
        new_coord_backward = ntuple(i -> i == dim ? coord[i] - 1 : coord[i], D)
        if coord[dim] > 0
            # Normal case: within bounds
            push!(neighbors, coord_to_vertex(g, new_coord_backward))
        elseif g.periodic[dim]
            # Periodic case: wrap around
            new_coord_wrap = ntuple(i -> i == dim ? g.sizes[i] - 1 : coord[i], D)
            push!(neighbors, coord_to_vertex(g, new_coord_wrap))
        end
    end

    return neighbors
end

"""Find edge index (undirected)."""
function find_edge_index(g::HypercubicLattice{D,T}, u::Integer, v::Integer) where {D,T}
    if !has_edge(g, u, v)
        return Int32(0)  # Edge doesn't exist
    end

    # For lattices, we can compute edge index mathematically
    # This is complex but deterministic - simpler to use a lookup table
    # For now, we'll use a simple sequential numbering
    min_vertex, max_vertex = minmax(u, v)
    coord_min = vertex_to_coord(g, min_vertex)
    coord_max = vertex_to_coord(g, max_vertex)

    # Find which dimension differs
    diff_dim = findfirst(i -> coord_min[i] != coord_max[i], 1:D)

    # Compute edge index based on position and dimension
    edge_idx = Int32(1)
    for dim in 1:diff_dim-1
        edge_idx += _edges_in_dimension(g, dim)
    end

    # Add offset within this dimension
    linear_coord = _coord_to_linear_in_slice(g, coord_min, diff_dim)
    edge_idx += linear_coord

    return edge_idx
end

"""Find directed edge index (same as undirected for lattices)."""
find_directed_edge_index(g::HypercubicLattice, u::Integer, v::Integer) = find_edge_index(g, u, v)

===============================================================================#
#===============================================================================

# ==============================================================================
# COORDINATE CONVERSION UTILITIES
# ==============================================================================

"""
    vertex_to_coord(g::HypercubicLattice{D,T}, v::Integer) -> NTuple{D,T}

Convert vertex index to D-dimensional coordinates.
Uses row-major ordering (last dimension varies fastest).
"""
function vertex_to_coord(g::HypercubicLattice{D,T}, v::Integer) where {D,T}
    if !has_vertex(g, v)
        throw(BoundsError("Vertex $v out of bounds"))
    end

    # Convert to 0-based indexing
    idx = v - 1
    coords = Vector{T}(undef, D)

    # Row-major order: last dimension varies fastest
    for dim in D:-1:1
        coords[dim] = T(idx % g.sizes[dim])
        idx ÷= g.sizes[dim]
    end

    return NTuple{D,T}(coords)
end

"""
    coord_to_vertex(g::HypercubicLattice{D,T}, coord::NTuple{D,T}) -> Int32

Convert D-dimensional coordinates to vertex index.
"""
function coord_to_vertex(g::HypercubicLattice{D,T}, coord::NTuple{D,T}) where {D,T}
    # Validate coordinates
    for dim in 1:D
        if !(0 ≤ coord[dim] < g.sizes[dim])
            throw(BoundsError("Coordinate $coord out of bounds for lattice with sizes $(g.sizes)"))
        end
    end

    # Row-major conversion
    idx = 0
    for dim in 1:D
        idx = idx * g.sizes[dim] + coord[dim]
    end

    return Int32(idx + 1)  # Convert to 1-based indexing
end

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

"""Get the number of edges in a specific dimension."""
function _edges_in_dimension(g::HypercubicLattice{D,T}, dim::Int) where {D,T}
    slices = prod(g.sizes) ÷ g.sizes[dim]
    if g.periodic[dim]
        return slices * g.sizes[dim]
    else
        return slices * (g.sizes[dim] - 1)
    end
end

"""Convert coordinate to linear index within a slice."""
function _coord_to_linear_in_slice(g::HypercubicLattice{D,T}, coord::NTuple{D,T}, varying_dim::Int) where {D,T}
    # This is a simplified version - full implementation would be more complex
    linear = 0
    multiplier = 1

    for dim in D:-1:1
        if dim != varying_dim
            linear += coord[dim] * multiplier
            multiplier *= g.sizes[dim]
        end
    end

    return Int32(linear + 1)
end

===============================================================================#


# ==============================================================================
# LATTICE-SPECIFIC UTILITIES
# ==============================================================================

"""Get the dimensions of the lattice."""
lattice_dimension(g::HypercubicLattice{D,T}) where {D,T} = D

"""Get the size tuple of the lattice."""
lattice_size(g::HypercubicLattice) = g.sizes

"""Check if the lattice has periodic boundary conditions."""
is_periodic(g::HypercubicLattice) = g.periodic

"""
    lattice_distance(g::HypercubicLattice, u::Integer, v::Integer) -> Float64

Compute the Manhattan distance between two vertices on the lattice.
Accounts for periodic boundary conditions.
"""
function lattice_distance(g::HypercubicLattice{D,T}, u::Integer, v::Integer) where {D,T}
    coord_u = vertex_to_coord(g, u)
    coord_v = vertex_to_coord(g, v)

    total_distance = 0.0
    for dim in 1:D
        diff = abs(coord_u[dim] - coord_v[dim])
        if g.periodic[dim]
            diff = min(diff, g.sizes[dim] - diff)
        end
        total_distance += diff
    end

    return total_distance
end

"""
    lattice_neighbors(g::HypercubicLattice, v::Integer) -> Vector{Int32}

Alias for neighbor_indices with better name for lattice context.
"""
lattice_neighbors(g::HypercubicLattice, v::Integer) = collect(neighbor_indices(g, v))

# ==============================================================================
# ULTRA-OPTIMIZED COORDINATE CONVERSION
# ==============================================================================

"""
    vertex_to_coord(g::HypercubicLattice{D,T}, v::Integer) -> NTuple{D,T}

Convert vertex index to coordinates using optimized integer arithmetic.
Fully inlined and branch-free for maximum performance.
"""
@inline function vertex_to_coord(g::HypercubicLattice{D,T}, v::Integer) where {D,T}
    # Use unsafe indexing for performance - bounds checking done at higher level
    idx = v - 1  # Convert to 0-based

    # Unroll the loop for small dimensions (most common cases)
    if D == 1
        return (T(idx),)
    elseif D == 2
        y = T(idx % g.sizes[2])
        x = T(idx ÷ g.sizes[2])
        return (x, y)
    elseif D == 3
        z = T(idx % g.sizes[3])
        idx ÷= g.sizes[3]
        y = T(idx % g.sizes[2])
        x = T(idx ÷ g.sizes[2])
        return (x, y, z)
    else
        # General case for higher dimensions
        coords = Vector{T}(undef, D)
        @inbounds for dim in D:-1:1
            coords[dim] = T(idx % g.sizes[dim])
            idx = idx ÷ g.sizes[dim]
        end
        return NTuple{D,T}(coords)
    end
end

"""
    coord_to_vertex(g::HypercubicLattice{D,T}, coord::NTuple{D,T}) -> Int32

Convert coordinates to vertex index using optimized arithmetic.
Specialized for common dimensions with loop unrolling.
"""
@inline function coord_to_vertex(g::HypercubicLattice{D,T}, coord::NTuple{D,T}) where {D,T}
    # Specialized implementations for common dimensions
    if D == 1
        return Int32(coord[1] + 1)
    elseif D == 2
        return Int32(coord[1] * g.sizes[2] + coord[2] + 1)
    elseif D == 3
        return Int32((coord[1] * g.sizes[2] + coord[2]) * g.sizes[3] + coord[3] + 1)
    else
        # General case using Horner's method for efficiency
        idx = coord[1]
        @inbounds for dim in 2:D
            idx = idx * g.sizes[dim] + coord[dim]
        end
        return Int32(idx + 1)
    end
end

# ==============================================================================
# ULTRA-OPTIMIZED NEIGHBOR COMPUTATION
# ==============================================================================

"""
    neighbor_indices(g::HypercubicLattice{D,T}, v::Integer) -> StaticVector{2D,Int32}

Compute neighbors using optimized coordinate arithmetic.
Returns a stack-allocated vector for maximum performance.
"""
@inline function neighbor_indices(g::HypercubicLattice{D,T}, v::Integer) where {D,T}
    coord = vertex_to_coord(g, v)

    # Pre-allocate with maximum possible neighbors (2D)
    neighbors = MVector{2*D,Int32}(undef)
    count = 0

    # Unroll for common dimensions
    if D == 1
        count = _compute_neighbors_1d!(neighbors, g, coord, count)
    elseif D == 2
        count = _compute_neighbors_2d!(neighbors, g, coord, count)
    elseif D == 3
        count = _compute_neighbors_3d!(neighbors, g, coord, count)
    else
        count = _compute_neighbors_nd!(neighbors, g, coord, count)
    end

    # Return only the valid neighbors
    return SVector{count,Int32}(neighbors[1:count])
end

# Specialized neighbor computation for 1D
@inline function _compute_neighbors_1d!(neighbors, g::HypercubicLattice{1,T}, coord, count) where T
    x = coord[1]
    count = 0

    # Left neighbor
    if x > 0
        count += 1
        neighbors[count] = Int32(x)  # coord_to_vertex((x-1,)) = x-1+1 = x
    elseif g.periodic[1]
        count += 1
        neighbors[count] = Int32(g.sizes[1])  # Wrap to end
    end

    # Right neighbor
    if x < g.sizes[1] - 1
        count += 1
        neighbors[count] = Int32(x + 2)  # coord_to_vertex((x+1,)) = x+1+1 = x+2
    elseif g.periodic[1]
        count += 1
        neighbors[count] = Int32(1)  # Wrap to beginning
    end

    return count
end

# Specialized neighbor computation for 2D
@inline function _compute_neighbors_2d!(neighbors, g::HypercubicLattice{2,T}, coord, count) where T
    x, y = coord
    width, height = g.sizes
    count = 0

    # Left neighbor (x-1, y)
    if x > 0
        count += 1
        neighbors[count] = Int32((x - 1) * height + y + 1)
    elseif g.periodic[1]
        count += 1
        neighbors[count] = Int32((width - 1) * height + y + 1)
    end

    # Right neighbor (x+1, y)
    if x < width - 1
        count += 1
        neighbors[count] = Int32((x + 1) * height + y + 1)
    elseif g.periodic[1]
        count += 1
        neighbors[count] = Int32(y + 1)
    end

    # Down neighbor (x, y-1)
    if y > 0
        count += 1
        neighbors[count] = Int32(x * height + (y - 1) + 1)
    elseif g.periodic[2]
        count += 1
        neighbors[count] = Int32(x * height + (height - 1) + 1)
    end

    # Up neighbor (x, y+1)
    if y < height - 1
        count += 1
        neighbors[count] = Int32(x * height + (y + 1) + 1)
    elseif g.periodic[2]
        count += 1
        neighbors[count] = Int32(x * height + 1)
    end

    return count
end

# Specialized neighbor computation for 3D
@inline function _compute_neighbors_3d!(neighbors, g::HypercubicLattice{3,T}, coord, count) where T
    x, y, z = coord
    width, height, depth = g.sizes
    count = 0

    # Helper for 3D coordinate to vertex conversion
    @inline coord_to_v(x, y, z) = Int32((x * height + y) * depth + z + 1)

    # X-direction neighbors
    if x > 0
        count += 1
        neighbors[count] = coord_to_v(x - 1, y, z)
    elseif g.periodic[1]
        count += 1
        neighbors[count] = coord_to_v(width - 1, y, z)
    end

    if x < width - 1
        count += 1
        neighbors[count] = coord_to_v(x + 1, y, z)
    elseif g.periodic[1]
        count += 1
        neighbors[count] = coord_to_v(0, y, z)
    end

    # Y-direction neighbors
    if y > 0
        count += 1
        neighbors[count] = coord_to_v(x, y - 1, z)
    elseif g.periodic[2]
        count += 1
        neighbors[count] = coord_to_v(x, height - 1, z)
    end

    if y < height - 1
        count += 1
        neighbors[count] = coord_to_v(x, y + 1, z)
    elseif g.periodic[2]
        count += 1
        neighbors[count] = coord_to_v(x, 0, z)
    end

    # Z-direction neighbors
    if z > 0
        count += 1
        neighbors[count] = coord_to_v(x, y, z - 1)
    elseif g.periodic[3]
        count += 1
        neighbors[count] = coord_to_v(x, y, depth - 1)
    end

    if z < depth - 1
        count += 1
        neighbors[count] = coord_to_v(x, y, z + 1)
    elseif g.periodic[3]
        count += 1
        neighbors[count] = coord_to_v(x, y, 0)
    end

    return count
end

# General case for higher dimensions
@inline function _compute_neighbors_nd!(neighbors, g::HypercubicLattice{D,T}, coord, count) where {D,T}
    count = 0

    # For each dimension, check both directions
    @inbounds for dim in 1:D
        # Negative direction
        if coord[dim] > 0
            new_coord = ntuple(i -> i == dim ? coord[i] - 1 : coord[i], D)
            count += 1
            neighbors[count] = coord_to_vertex(g, new_coord)
        elseif g.periodic[dim]
            new_coord = ntuple(i -> i == dim ? g.sizes[i] - 1 : coord[i], D)
            count += 1
            neighbors[count] = coord_to_vertex(g, new_coord)
        end

        # Positive direction
        if coord[dim] < g.sizes[dim] - 1
            new_coord = ntuple(i -> i == dim ? coord[i] + 1 : coord[i], D)
            count += 1
            neighbors[count] = coord_to_vertex(g, new_coord)
        elseif g.periodic[dim]
            new_coord = ntuple(i -> i == dim ? T(0) : coord[i], D)
            count += 1
            neighbors[count] = coord_to_vertex(g, new_coord)
        end
    end

    return count
end

# ==============================================================================
# OPTIMIZED INDEXED NEIGHBOR ACCESS
# ==============================================================================

"""
    neighbor(g::HypercubicLattice{D,T}, v::Integer, k::Integer) -> Int32

Get the k-th neighbor of vertex v without allocating the full neighbor list.
Optimized for accessing specific neighbors in computational loops.
"""
@inline function neighbor(g::HypercubicLattice{D,T}, v::Integer, k::Integer) where {D,T}
    coord = vertex_to_coord(g, v)

    # Specialized implementations for common dimensions
    if D == 1
        return _neighbor_1d(g, coord, k)
    elseif D == 2
        return _neighbor_2d(g, coord, k)
    elseif D == 3
        return _neighbor_3d(g, coord, k)
    else
        # General case - compute all neighbors and index
        neighbors = neighbor_indices(g, v)
        if k > length(neighbors)
            throw(BoundsError("Neighbor index $k out of bounds (vertex $v has $(length(neighbors)) neighbors)"))
        end
        return neighbors[k]
    end
end

@inline function _neighbor_1d(g::HypercubicLattice{1,T}, coord, k) where T
    x = coord[1]

    if k == 1
        # First neighbor: left direction
        if x > 0
            return Int32(x)
        elseif g.periodic[1]
            return Int32(g.sizes[1])
        else
            throw(BoundsError("Vertex at coordinate $coord has no neighbor $k"))
        end
    elseif k == 2
        # Second neighbor: right direction
        if x < g.sizes[1] - 1
            return Int32(x + 2)
        elseif g.periodic[1]
            return Int32(1)
        else
            throw(BoundsError("Vertex at coordinate $coord has no neighbor $k"))
        end
    else
        throw(BoundsError("1D lattice vertices have at most 2 neighbors, requested neighbor $k"))
    end
end

@inline function _neighbor_2d(g::HypercubicLattice{2,T}, coord, k) where T
    x, y = coord
    width, height = g.sizes

    # Order: left, right, down, up (consistent with neighbor_indices)
    if k == 1  # Left
        if x > 0 || g.periodic[1]
            nx = x > 0 ? x - 1 : width - 1
            return Int32(nx * height + y + 1)
        end
    elseif k == 2  # Right
        if x < width - 1 || g.periodic[1]
            nx = x < width - 1 ? x + 1 : 0
            return Int32(nx * height + y + 1)
        end
    elseif k == 3  # Down
        if y > 0 || g.periodic[2]
            ny = y > 0 ? y - 1 : height - 1
            return Int32(x * height + ny + 1)
        end
    elseif k == 4  # Up
        if y < height - 1 || g.periodic[2]
            ny = y < height - 1 ? y + 1 : 0
            return Int32(x * height + ny + 1)
        end
    end

    throw(BoundsError("Invalid neighbor index $k for vertex at coordinate $coord"))
end

@inline function _neighbor_3d(g::HypercubicLattice{3,T}, coord, k) where T
    x, y, z = coord
    width, height, depth = g.sizes

    @inline coord_to_v(x, y, z) = Int32((x * height + y) * depth + z + 1)

    # Order: -x, +x, -y, +y, -z, +z
    if k == 1  # -x
        if x > 0 || g.periodic[1]
            nx = x > 0 ? x - 1 : width - 1
            return coord_to_v(nx, y, z)
        end
    elseif k == 2  # +x
        if x < width - 1 || g.periodic[1]
            nx = x < width - 1 ? x + 1 : 0
            return coord_to_v(nx, y, z)
        end
    elseif k == 3  # -y
        if y > 0 || g.periodic[2]
            ny = y > 0 ? y - 1 : height - 1
            return coord_to_v(x, ny, z)
        end
    elseif k == 4  # +y
        if y < height - 1 || g.periodic[2]
            ny = y < height - 1 ? y + 1 : 0
            return coord_to_v(x, ny, z)
        end
    elseif k == 5  # -z
        if z > 0 || g.periodic[3]
            nz = z > 0 ? z - 1 : depth - 1
            return coord_to_v(x, y, nz)
        end
    elseif k == 6  # +z
        if z < depth - 1 || g.periodic[3]
            nz = z < depth - 1 ? z + 1 : 0
            return coord_to_v(x, y, nz)
        end
    end

    throw(BoundsError("Invalid neighbor index $k for vertex at coordinate $coord"))
end

# ==============================================================================
# DEGREE COMPUTATION (OPTIMIZED)
# ==============================================================================

"""
    degree(g::HypercubicLattice{D,T}, v::Integer) -> Int32

Compute degree without allocating neighbor list.
"""
@inline function degree(g::HypercubicLattice{D,T}, v::Integer) where {D,T}
    coord = vertex_to_coord(g, v)
    deg = 0

    @inbounds for dim in 1:D
        # Check negative direction
        if coord[dim] > 0 || g.periodic[dim]
            deg += 1
        end
        # Check positive direction
        if coord[dim] < g.sizes[dim] - 1 || g.periodic[dim]
            deg += 1
        end
    end

    return Int32(deg)
end
end # module Lattices