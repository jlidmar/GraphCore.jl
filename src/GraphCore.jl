# GraphCore.jl

# Copyright (c) 2025 Jack Lidmar
# All rights reserved.

# This software is licensed under the MIT License. See the LICENSE file for details.

# SPDX-FileCopyrightText: 2025 Jack Lidmar <jlidmar@kth.se>
# SPDX-License-Identifier: MIT

"""
    GraphCore
A module for graph-based computations and algorithms.
"""
module GraphCore

# ==============================================================================
# CORE TYPES AND INTERFACES
# ==============================================================================
export GraphInterface, WeightedGraphInterface, PropertyGraphInterface
export CoreGraph, WeightedGraph, PropertyGraph
export AdjGraph, WeightedAdjGraph, PropertyAdjGraph

# ==============================================================================
# GRAPH CONSTRUCTION (Primary API)
# ==============================================================================
export build_graph, build_core_graph, build_weighted_graph, build_property_graph
export build_adj_graph, build_weighted_adj_graph, build_property_adj_graph

# ==============================================================================
# BASIC GRAPH QUERIES
# ==============================================================================
export num_vertices, num_edges, num_directed_edges, is_directed_graph
export has_vertex, has_edge, degree
export vertices, edges, all_edges, all_directed_edges, neighbor, neighbor_indices
export edge_indices, directed_edge_indices
export edge_index, directed_edge_index
export find_edge_index, find_directed_edge_index

# ==============================================================================
# WEIGHTED/PROPERTY OPERATIONS
# ==============================================================================
export edge_weight, edge_weights, neighbor_weights
export vertex_property, edge_property
export set_vertex_property!, set_edge_property!
export vertex_properties, edge_properties

# ==============================================================================
# MUTABLE OPERATIONS
# ==============================================================================
export add_vertex!, add_edge!, remove_vertex!, remove_edge!

# Include core source files
include("GraphInterface.jl")
include("CoreGraph.jl")
include("AdjGraph.jl")
include("PropertyGraph.jl")
include("utils.jl")

# Make utility functions public but not exported
public canonicalize_edges, symmetrize_edges

# ==============================================================================
# SUBMODULES
# ==============================================================================

# Graph Construction and Builders
include("GraphConstruction.jl")
using .GraphConstruction
export GraphBuilder, WeightedGraphBuilder, PropertyGraphBuilder, FullGraphBuilder
export build_from_function

# Conversion utilities
include("Conversions.jl")
using .Conversions
export from_graphs_jl, from_weighted_graphs_jl
# Re-export commonly used conversions
export to_adj_graph, to_core_graph, to_weighted_graph

# Other graph types
include("Lattices.jl")
include("PowerOfTwoLattices.jl")
using .Lattices, .PowerOfTwoLattices
export HypercubicLattice, Grid2D, Grid3D, Chain1D
export PowerOfTwoLattice, P2Grid2D, P2Grid3D, P2Chain1D
export lattice_size, lattice_dimension, coord_to_vertex, vertex_to_coord
export lattice_neighbors, lattice_distance

end # module GraphCore