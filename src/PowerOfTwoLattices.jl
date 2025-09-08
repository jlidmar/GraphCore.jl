"""
Power-of-Two Hypercubic Lattices
================================

Ultra-fast lattice implementation for dimensions that are powers of 2.
Uses bit operations instead of division/modulo for maximum performance.

Restrictions:
- All dimensions must be powers of 2
- Maximum dimension D ≤ 5 (to keep neighbor tuples manageable)
- Periodic boundary conditions only
"""
module PowerOfTwoLattices

import ..GraphCore: GraphInterface, num_vertices, num_edges, num_directed_edges, has_vertex, has_edge, degree
import ..GraphCore: neighbor_indices, edge_indices, directed_edge_indices, neighbor, edge_index, directed_edge_index
import ..GraphCore: is_directed_graph, find_edge_index, find_directed_edge_index

export PowerOfTwoLattice
export P2Grid2D, P2Grid3D, P2Chain1D
export lattice_size, lattice_dimension
export lattice_neighbors, lattice_distance
public coord_to_vertex, vertex_to_coord

# ==============================================================================
# POWER-OF-TWO LATTICE STRUCTURE
# ==============================================================================

"""
    PowerOfTwoLattice{D} <: GraphInterface

Ultra-fast D-dimensional hypercubic lattice where all sizes are powers of 2.
Uses bit operations for coordinate conversion and neighbor lookup.

# Restrictions
- D ≤ 5 (keeps neighbor tuples reasonable)
- All sizes must be powers of 2: 2, 4, 8, 16, 32, 64, 128, ...
- Periodic boundary conditions (makes bit operations clean)

# Type Parameters
- D::Int: Dimension (1 ≤ D ≤ 5)

# Examples
```julia
# 2D: 16×32 grid (2^4 × 2^5)
g2d = P2Grid2D(4, 5)  # log₂ sizes

# 3D: 8×8×16 cube (2^3 × 2^3 × 2^4)
g3d = P2Grid3D(3, 3, 4)

# 1D: 64-element chain (2^6)
g1d = P2Chain1D(6)
```
"""
struct PowerOfTwoLattice{D} <: GraphInterface
    log_sizes::NTuple{D,UInt8}    # log₂ of each dimension size
    sizes::NTuple{D,Int32}        # Actual sizes (for convenience)
    masks::NTuple{D,Int32}        # Bit masks for each dimension
    total_vertices::Int32         # Total number of vertices
    total_edges::Int32           # Total number of edges

    function PowerOfTwoLattice{D}(log_sizes::NTuple{D,Integer}) where D
        if D < 1 || D > 5
            throw(ArgumentError("Dimension must be 1 ≤ D ≤ 5, got $D"))
        end

        if any(ls < 1 || ls > 20 for ls in log_sizes)  # Reasonable size limits
            throw(ArgumentError("Log sizes must be 1 ≤ log_size ≤ 20, got $log_sizes"))
        end

        log_sizes_u8 = NTuple{D,UInt8}(log_sizes)
        sizes = ntuple(i -> Int32(1 << log_sizes[i]), D)
        masks = ntuple(i -> Int32((1 << log_sizes[i]) - 1), D)

        total_vertices = Int32(prod(sizes))
        total_edges = Int32(D * total_vertices)  # Each vertex has 2 neighbors per dimension

        new{D}(log_sizes_u8, sizes, masks, total_vertices, total_edges)
    end
end

# Convenience constructors
PowerOfTwoLattice(log_sizes::NTuple{D,Integer}) where D = PowerOfTwoLattice{D}(log_sizes)
PowerOfTwoLattice(log_sizes::Vararg{Integer,D}) where D = PowerOfTwoLattice{D}(log_sizes)

"""1D chain: size = 2^log_size"""
P2Chain1D(log_size::Integer) = PowerOfTwoLattice{1}((log_size,))

"""2D grid: sizes = 2^log_width × 2^log_height"""
P2Grid2D(log_width::Integer, log_height::Integer) = PowerOfTwoLattice{2}((log_width, log_height))

"""3D cube: sizes = 2^log_width × 2^log_height × 2^log_depth"""
P2Grid3D(log_width::Integer, log_height::Integer, log_depth::Integer) =
    PowerOfTwoLattice{3}((log_width, log_height, log_depth))

# ==============================================================================
# CORE INTERFACE (MINIMAL IMPLEMENTATION)
# ==============================================================================

num_vertices(g::PowerOfTwoLattice) = g.total_vertices
num_edges(g::PowerOfTwoLattice) = g.total_edges
num_directed_edges(g::PowerOfTwoLattice) = g.total_edges
is_directed_graph(g::PowerOfTwoLattice) = false

@inline has_vertex(g::PowerOfTwoLattice, v::Integer) = 1 ≤ v ≤ g.total_vertices

@inline function has_edge(g::PowerOfTwoLattice{D}, u::Integer, v::Integer) where D
    if !has_vertex(g, u) || !has_vertex(g, v)
        return false
    end

    coord_u = vertex_to_coord(g, u)
    coord_v = vertex_to_coord(g, v)

    # Check if they differ by ±1 in exactly one dimension (with wraparound)
    diff_count = 0
    for dim in 1:D
        diff = (coord_u[dim] - coord_v[dim]) & g.masks[dim]
        if diff == 1 || diff == g.masks[dim]  # ±1 with wraparound
            diff_count += 1
        elseif diff != 0
            return false
        end
    end

    return diff_count == 1
end

# ==============================================================================
# ULTRA-FAST COORDINATE CONVERSION
# ==============================================================================

"""Convert vertex to coordinates using bit operations."""
@inline function vertex_to_coord(g::PowerOfTwoLattice{D}, v::Integer) where D
    idx = v - 1  # Convert to 0-based

    if D == 1
        return (Int32(idx),)
    elseif D == 2
        y = Int32(idx & g.masks[2])
        x = Int32(idx >> g.log_sizes[2])
        return (x, y)
    elseif D == 3
        z = Int32(idx & g.masks[3])
        idx >>= g.log_sizes[3]
        y = Int32(idx & g.masks[2])
        x = Int32(idx >> g.log_sizes[2])
        return (x, y, z)
    else
        # General case for D=4,5
        coords = Vector{Int32}(undef, D)
        @inbounds for dim in D:-1:1
            coords[dim] = Int32(idx & g.masks[dim])
            idx >>= g.log_sizes[dim]
        end
        return NTuple{D,Int32}(coords)
    end
end

"""Convert coordinates to vertex using bit operations."""
@inline function coord_to_vertex(g::PowerOfTwoLattice{D}, coord::NTuple{D,Int32}) where D
    if D == 1
        return coord[1] + 1
    elseif D == 2
        return (coord[1] << g.log_sizes[2]) | coord[2] + 1
    elseif D == 3
        return ((coord[1] << g.log_sizes[2]) | coord[2]) << g.log_sizes[3] | coord[3] + 1
    else
        # General case
        idx = coord[1]
        @inbounds for dim in 2:D
            idx = (idx << g.log_sizes[dim]) | coord[dim]
        end
        return idx + 1
    end
end

# ==============================================================================
# NEIGHBOR COMPUTATION RETURNING TUPLES
# ==============================================================================

"""
Return neighbors as a statically-sized tuple for maximum performance.
Each dimension contributes exactly 2 neighbors (periodic boundaries).
"""
@inline function neighbor_indices(g::PowerOfTwoLattice{D}, v::Integer) where D
    coord = vertex_to_coord(g, v)

    if D == 1
        return _neighbors_1d(g, coord)
    elseif D == 2
        return _neighbors_2d(g, coord)
    elseif D == 3
        return _neighbors_3d(g, coord)
    elseif D == 4
        return _neighbors_4d(g, coord)
    elseif D == 5
        return _neighbors_5d(g, coord)
    end
end

# 1D: exactly 2 neighbors
@inline function _neighbors_1d(g::PowerOfTwoLattice{1}, coord)
    x = coord[1]
    left = ((x - 1) & g.masks[1]) + 1
    right = ((x + 1) & g.masks[1]) + 1
    return (left, right)
end

# 2D: exactly 4 neighbors
@inline function _neighbors_2d(g::PowerOfTwoLattice{2}, coord)
    x, y = coord

    # Use bit operations for wraparound
    left_x = (x - 1) & g.masks[1]
    right_x = (x + 1) & g.masks[1]
    down_y = (y - 1) & g.masks[2]
    up_y = (y + 1) & g.masks[2]

    # Convert back to vertex indices using bit shifts
    left = (left_x << g.log_sizes[2]) | y + 1
    right = (right_x << g.log_sizes[2]) | y + 1
    down = (x << g.log_sizes[2]) | down_y + 1
    up = (x << g.log_sizes[2]) | up_y + 1

    return (left, right, down, up)
end

# 3D: exactly 6 neighbors
@inline function _neighbors_3d(g::PowerOfTwoLattice{3}, coord)
    x, y, z = coord

    # Bit operations for all directions
    left_x = (x - 1) & g.masks[1]
    right_x = (x + 1) & g.masks[1]
    down_y = (y - 1) & g.masks[2]
    up_y = (y + 1) & g.masks[2]
    back_z = (z - 1) & g.masks[3]
    front_z = (z + 1) & g.masks[3]

    # Convert to vertices
    base = (x << g.log_sizes[2]) | y
    current_slice = (base << g.log_sizes[3]) | z + 1

    left = ((left_x << g.log_sizes[2]) | y) << g.log_sizes[3] | z + 1
    right = ((right_x << g.log_sizes[2]) | y) << g.log_sizes[3] | z + 1
    down = ((x << g.log_sizes[2]) | down_y) << g.log_sizes[3] | z + 1
    up = ((x << g.log_sizes[2]) | up_y) << g.log_sizes[3] | z + 1
    back = (base << g.log_sizes[3]) | back_z + 1
    front = (base << g.log_sizes[3]) | front_z + 1

    return (left, right, down, up, back, front)
end

# 4D: exactly 8 neighbors
@inline function _neighbors_4d(g::PowerOfTwoLattice{4}, coord)
    # Implementation for 4D case
    neighbors = Vector{Int32}(undef, 8)
    idx = 1

    @inbounds for dim in 1:4
        # Negative direction
        new_coord = ntuple(i -> i == dim ? (coord[i] - 1) & g.masks[i] : coord[i], 4)
        neighbors[idx] = coord_to_vertex(g, new_coord)
        idx += 1

        # Positive direction
        new_coord = ntuple(i -> i == dim ? (coord[i] + 1) & g.masks[i] : coord[i], 4)
        neighbors[idx] = coord_to_vertex(g, new_coord)
        idx += 1
    end

    return (neighbors[1], neighbors[2], neighbors[3], neighbors[4],
            neighbors[5], neighbors[6], neighbors[7], neighbors[8])
end

# 5D: exactly 10 neighbors
@inline function _neighbors_5d(g::PowerOfTwoLattice{5}, coord)
    # Implementation for 5D case
    neighbors = Vector{Int32}(undef, 10)
    idx = 1

    @inbounds for dim in 1:5
        # Negative direction
        new_coord = ntuple(i -> i == dim ? (coord[i] - 1) & g.masks[i] : coord[i], 5)
        neighbors[idx] = coord_to_vertex(g, new_coord)
        idx += 1

        # Positive direction
        new_coord = ntuple(i -> i == dim ? (coord[i] + 1) & g.masks[i] : coord[i], 5)
        neighbors[idx] = coord_to_vertex(g, new_coord)
        idx += 1
    end

    return (neighbors[1], neighbors[2], neighbors[3], neighbors[4],
            neighbors[5], neighbors[6], neighbors[7], neighbors[8],
            neighbors[9], neighbors[10])
end

# ==============================================================================
# OPTIMIZED SINGLE NEIGHBOR ACCESS
# ==============================================================================

"""
Get k-th neighbor directly without computing all neighbors.
Order: dim1-, dim1+, dim2-, dim2+, ...
"""
@inline function neighbor(g::PowerOfTwoLattice{D}, v::Integer, k::Integer) where D
    if k < 1 || k > 2*D
        throw(BoundsError("Neighbor index $k out of bounds (vertices have $(2*D) neighbors)"))
    end

    coord = vertex_to_coord(g, v)
    dim = (k + 1) ÷ 2  # Which dimension
    direction = k % 2  # 1 = negative, 0 = positive

    if direction == 1  # Negative direction
        new_coord = ntuple(i -> i == dim ? (coord[i] - 1) & g.masks[i] : coord[i], D)
    else  # Positive direction
        new_coord = ntuple(i -> i == dim ? (coord[i] + 1) & g.masks[i] : coord[i], D)
    end

    return coord_to_vertex(g, new_coord)
end

# ==============================================================================
# EDGE INDEXING FOR POWER-OF-TWO LATTICES
# ==============================================================================

"""
    edge_index(g::PowerOfTwoLattice{D}, v::Integer, i::Integer) -> Int32

Get the i-th edge index from vertex v.
For power-of-2 lattices, edges are ordered by dimension: dim1-, dim1+, dim2-, dim2+, ...
"""
@inline function edge_index(g::PowerOfTwoLattice{D}, v::Integer, i::Integer) where D
    if i < 1 || i > 2*D
        throw(BoundsError("Edge index $i out of bounds (vertex $v has $(2*D) edges)"))
    end

    # Simple formula: each vertex contributes 2*D edges, ordered by local edge index
    return Int32((v - 1) * (2*D) + i)
end

"""
    edge_indices(g::PowerOfTwoLattice{D}, v::Integer) -> NTuple{2D,Int32}

Get all edge indices from vertex v as a tuple.
Returns edges in order: dim1-, dim1+, dim2-, dim2+, dim3-, dim3+, ...
"""
@inline function edge_indices(g::PowerOfTwoLattice{D}, v::Integer) where D
    base = Int32((v - 1) * (2*D))

    if D == 1
        return (base + 1, base + 2)
    elseif D == 2
        return (base + 1, base + 2, base + 3, base + 4)
    elseif D == 3
        return (base + 1, base + 2, base + 3, base + 4, base + 5, base + 6)
    elseif D == 4
        return (base + 1, base + 2, base + 3, base + 4,
                base + 5, base + 6, base + 7, base + 8)
    elseif D == 5
        return (base + 1, base + 2, base + 3, base + 4, base + 5,
                base + 6, base + 7, base + 8, base + 9, base + 10)
    end
end

# ==============================================================================
# DIRECTED EDGE INDEXING (SAME AS UNDIRECTED FOR LATTICES)
# ==============================================================================

"""
    directed_edge_index(g::PowerOfTwoLattice{D}, v::Integer, i::Integer) -> Int32

Get the i-th directed edge index from vertex v.
For undirected lattices, this is the same as edge_index.
"""
@inline directed_edge_index(g::PowerOfTwoLattice{D}, v::Integer, i::Integer) where D =
    edge_index(g, v, i)

"""
    directed_edge_indices(g::PowerOfTwoLattice{D}, v::Integer) -> NTuple{2D,Int32}

Get all directed edge indices from vertex v.
For undirected lattices, this is the same as edge_indices.
"""
@inline directed_edge_indices(g::PowerOfTwoLattice{D}, v::Integer) where D =
    edge_indices(g, v)

# ==============================================================================
# ENHANCED EDGE LOOKUP
# ==============================================================================

"""
    find_edge_index(g::PowerOfTwoLattice{D}, u::Integer, v::Integer) -> Int32

Find the edge index for edge (u,v).
Uses the smaller vertex as the base and computes which local edge it is.
"""
@inline function find_edge_index(g::PowerOfTwoLattice{D}, u::Integer, v::Integer) where D
    if !has_edge(g, u, v)
        return Int32(0)
    end

    # Use smaller vertex as base for consistent edge indexing
    min_vertex, max_vertex = minmax(u, v)

    # Find which local edge this is from min_vertex
    coord_min = vertex_to_coord(g, min_vertex)
    coord_max = vertex_to_coord(g, max_vertex)

    # Find the dimension and direction
    local_edge_idx = _find_local_edge_index(g, coord_min, coord_max)

    return edge_index(g, min_vertex, local_edge_idx)
end

@inline find_directed_edge_index(g::PowerOfTwoLattice, u::Integer, v::Integer) = find_edge_index(g, u, v)

"""
Helper function to find which local edge connects two adjacent coordinates.
"""
@inline function _find_local_edge_index(g::PowerOfTwoLattice{D}, coord_min, coord_max) where D
    @inbounds for dim in 1:D
        diff = (coord_max[dim] - coord_min[dim]) & g.masks[dim]
        if diff == 1
            return 2*dim  # Positive direction edge
        elseif diff == g.masks[dim]  # Wraparound: max_coord is actually smaller
            return 2*dim - 1  # Negative direction edge
        end
    end

    # Should never reach here for valid adjacent vertices
    throw(ArgumentError("Vertices are not adjacent"))
end

# ==============================================================================
# REVERSE EDGE LOOKUP
# ==============================================================================

"""
    edge_to_vertices(g::PowerOfTwoLattice{D}, edge_idx::Integer) -> Tuple{Int32,Int32}

Convert edge index back to the two vertices it connects.
Returns (smaller_vertex, larger_vertex) for consistency.
"""
@inline function edge_to_vertices(g::PowerOfTwoLattice{D}, edge_idx::Integer) where D
    if edge_idx < 1 || edge_idx > num_edges(g)
        throw(BoundsError("Edge index $edge_idx out of bounds"))
    end

    # Decode which vertex and local edge
    vertex_offset = (edge_idx - 1) ÷ (2*D)
    local_edge = (edge_idx - 1) % (2*D) + 1

    base_vertex = Int32(vertex_offset + 1)
    neighbor_vertex = neighbor(g, base_vertex, local_edge)

    return minmax(base_vertex, neighbor_vertex)
end

# ==============================================================================
# ENHANCED UTILITIES
# ==============================================================================

"""
    local_edge_to_direction(local_edge::Integer, D::Integer) -> (dim, direction)

Convert local edge index to dimension and direction.
Returns (dimension, direction) where direction is :negative or :positive.
"""
@inline function local_edge_to_direction(local_edge::Integer, D::Integer)
    if local_edge < 1 || local_edge > 2*D
        throw(BoundsError("Local edge index $local_edge out of bounds"))
    end

    dim = (local_edge + 1) ÷ 2
    direction = local_edge % 2 == 1 ? :negative : :positive

    return (dim, direction)
end

"""
    direction_to_local_edge(dim::Integer, direction::Symbol, D::Integer) -> Int32

Convert dimension and direction to local edge index.
"""
@inline function direction_to_local_edge(dim::Integer, direction::Symbol, D::Integer)
    if dim < 1 || dim > D
        throw(BoundsError("Dimension $dim out of bounds"))
    end

    if direction == :negative
        return Int32(2*dim - 1)
    elseif direction == :positive
        return Int32(2*dim)
    else
        throw(ArgumentError("Direction must be :negative or :positive, got $direction"))
    end
end

# ==============================================================================
# SIMPLE UTILITIES
# ==============================================================================

"""Degree is always 2*D for power-of-2 periodic lattices."""
@inline degree(g::PowerOfTwoLattice{D}, v::Integer) where D = 2*D

"""Get lattice dimensions."""
lattice_dimension(g::PowerOfTwoLattice{D}) where D = D

"""Get actual sizes."""
lattice_size(g::PowerOfTwoLattice) = g.sizes

"""Get log₂ sizes."""
lattice_log_size(g::PowerOfTwoLattice) = g.log_sizes

end # module PowerOfTwoLattices