# Changelog

## [0.4.4]

### Changed
- **Conversion functions**: Improve `to_core_graph`, etc.
- **Documentation Improvements**: Streamlined and centralized function docstrings for better maintainability

## [0.4.0]

### Changed
- **Module Restructuring**: Renamed `GraphConstruction` module to `Builders` for improved clarity and consistency
  - Updated all internal references and documentation
  - Simplified imports: use `GraphCore.Builders` instead of `GraphCore.GraphConstruction`

## [0.3.1]

### Added
- **Plotting Examples**: New visualization capabilities using Plots and GraphRecipes plotting tools
  - `examples/plotting_example.jl` - Simple example
  - Integration with Plots.jl ecosystem via GraphRecipes.jl

## [0.3.0]

### Added
- **Graph Mutation Support**: You can now modify graphs in-place without rebuilding
  - `add_vertex!(g)` - Add isolated vertices to existing graphs
  - `add_edge!(g, u, v)` - Add edges dynamically (with weights for WeightedGraph)
  - `remove_edge!(g, u, v)` - Remove specific edges
  - `remove_vertex!(g, v)` - Remove vertices and all connected edges
- **Easy Graph Conversion**: Convert between graph types using intuitive constructors
  - `CoreGraph(adj_graph)` - Convert any graph to CSR format
  - `WeightedGraph(unweighted_graph)` - Add weight support to existing graphs
  - `AdjGraph(core_graph)` - Convert to adjacency list format for mutations
  - Type-safe conversions: `CoreGraph{false}(directed_graph)` enforces directedness
- **Better Edge Inspection**: Edge iterators now show helpful information
  - `edges(g)` displays edge count, graph type, and preview: "EdgeIterator over 5 edges from CoreGraph (undirected): (1,2), (1,3), (2,3), ..."
  - Clear distinction between `edges(g)` and `all_directed_edges(g)`

### Changed
- **Mutable CoreGraph and WeightedGraph**: These graph types now support in-place modifications
- **Cleaner Documentation**: Simplified and organized help text for `build_graph` functions
  - Each graph type's docs now focus on when to use it
  - Removed redundant information across different graph types

### Performance & Reliability
- All mutations maintain optimal performance characteristics
- Type safety prevents common conversion errors (e.g., mixing directed/undirected)
- Comprehensive test coverage ensures reliability (6190+ tests passing)

## [0.2.2] - 2025-09-04

### Initial Release
- **Multiple Graph Storage Options**: Choose the right format for your use case
  - `CoreGraph` - Compressed storage for static graphs and fast queries
  - `AdjGraph` - Flexible storage for dynamic graphs requiring frequent modifications
  - `WeightedGraph` - Built-in edge weight support
  - `PropertyGraph` - Store custom data on vertices and edges
- **Graphs.jl Compatibility**: Drop-in replacement for most Graphs.jl functionality
- **Unified Interface**: All graph types work with the same functions and syntax

[0.4.4]: https://github.com/jlidmar/GraphCore.jl/releases/tag/v0.4.3

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
