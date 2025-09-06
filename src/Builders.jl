"""
Graph Construction Builder
=========================

Efficient builder for constructing graphs with incremental vertex/edge addition.
Optimized for construction speed, not query performance.
"""
module Builders

using ..GraphCore: GraphInterface
using ..GraphCore: CoreGraph, WeightedGraph, PropertyGraph
using ..GraphCore: AdjGraph, WeightedAdjGraph
using ..GraphCore: build_core_graph, build_weighted_graph, build_adj_graph
import ..GraphCore: add_vertex!, add_edge!, remove_vertex!, remove_edge!
import ..GraphCore: num_vertices, num_edges, is_directed_graph, build_graph

# Export the main types and functions
export GraphBuilder, WeightedGraphBuilder, PropertyGraphBuilder, FullGraphBuilder
export add_vertex!, add_edge!, build_graph
export build_from_function

# Include the implementation
include("GraphBuilder.jl")

# Generators
export empty_graph, complete_graph, path_graph, cycle_graph
export star_graph, grid_graph, random_graph, lattice_graph
export erdos_renyi_graph, barabasi_albert_graph, wheel_graph, hypercubic_graph

include("Generators.jl")

end # module Builders
