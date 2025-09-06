# GraphCore.jl Examples

This folder contains examples and benchmarks demonstrating the performance characteristics and usage patterns of GraphCore.jl.

## Benchmark Suite Overview

1. `basic_usage.jl` - Simple usage example
2. `quick_benchmark.jl` - Quick performance demo
3. `coregraph_vs_adjgraph.jl` - Comparison for graph type selection
4. `dijkstra_benchmark.jl` - Real algorithm performance and interoperability
5. `micro_benchmark.jl` - Performance analysis and optimization techniques
6. `plotting_example.jl` - Graph visualization examples

## Files

### `basic_usage.jl`
Simple usage example showing how to create and query graphs with properties.

**Includes:**
- Basic graph creation and queries
- Property management
- Simple traversal patterns

### `quick_benchmark.jl`
A comprehensive but fast-running benchmark that demonstrates GraphCore's key performance characteristics.

**What it shows:**
- Graph construction performance (CoreGraph vs AdjGraph)
- Neighbor access patterns and optimization benefits
- Dynamic operations (additions/removals)
- PropertyGraph usage with real-world-style data
- Memory efficiency comparisons

**Perfect for:** Quick evaluation, demos, CI testing

**Run with:** `julia --project examples/quick_benchmark.jl`

### `coregraph_vs_adjgraph.jl`
Focused comparison between the two main graph representations to help users choose the right type.

**What it shows:**
- **CoreGraph**: CSR (Compressed Sparse Row) format, best for static analysis
- **AdjGraph**: Adjacency list format, best for dynamic modifications
- Detailed timing comparisons across different graph sizes and structures
- When to use each representation

**Perfect for:** Architecture decisions, understanding trade-offs

**Run with:** `julia --project examples/coregraph_vs_adjgraph.jl`

### `dijkstra_benchmark.jl`
Demonstrating real algorithm performance and ecosystem interoperability.

**What it shows:**
- Dijkstra's shortest path algorithm performance across graph types
- **GraphCore.jl types**: CoreGraph, AdjGraph, WeightedGraph
- **Graphs.jl interoperability**: How GraphCore graphs work with existing algorithms
- **Correctness verification**: Ensures algorithms produce correct results
- **Memory analysis**: Construction and algorithm overhead
- **Performance comparison**: Direct comparison with Graphs.jl's SimpleGraph

**Perfect for:** Validating real-world performance, algorithm development, ecosystem integration

**Run with:** `julia --project examples/dijkstra_benchmark.jl`

### `micro_benchmark.jl`
Detailed analysis of implementation-level performance characteristics and optimization techniques.

**What it shows:**
- Why performance can vary between graph types in different settings
- **Bounds checking impact**: How `--check-bounds=no` affects relative performance
- **Memory layout effects**: Cache behavior across different graph sizes  
- **Implementation details**: Raw neighbor access overhead analysis
- **Optimization techniques**: When and how to use `@inbounds` effectively

**Perfect for:** Performance optimization, understanding internals, advanced users

**Run with:** `julia --project examples/micro_benchmark.jl`

### `plotting_example.jl`
Graph visualization examples using modern plotting tools.

**What it shows:**
- **Social network visualization**: Pretty graphs with node labels
- **Integration with Plots.jl ecosystem**: Using GraphRecipes for graph plotting
- **Export capabilities**: Saving plots as PNG files

**Perfect for:** Creating figures for papers, presentations, or documentation

**Run with:** 
```bash
# First install plotting dependencies
julia --project -e "using Pkg; Pkg.add([\"Plots\", \"GraphRecipes\"])"

# Then run the examples
julia --project examples/plotting_example.jl
```


### Optimization Techniques

**Safe use of `@inbounds` in performance-critical loops**:
   ```julia
   # After validating inputs
   for v in vertices(g)
       neighbors = @inbounds neighbor_indices(g, v)
       # Fast inner loop
   end
   ```

**Global bounds check elimination for production**:
   ```bash
   # For maximum performance
   julia --check-bounds=no --project your_script.jl
   ```

## Running the Benchmarks

**Note:** The benchmark examples require BenchmarkTools.jl. Install it first:

```bash
# Install BenchmarkTools for running benchmarks
julia --project -e "using Pkg; Pkg.add(\"BenchmarkTools\")"
```

**Note:** The plotting example requires Plots.jl and GraphRecipes.jl:

```bash
# Install plotting packages for visualization examples
julia --project -e "using Pkg; Pkg.add([\"Plots\", \"GraphRecipes\"])"
```

Then run the examples:

```bash
cd GraphCore

# 1. Quick performance overview
julia --project examples/quick_benchmark.jl

# 2. Understand the trade-offs between graph types  
julia --project examples/coregraph_vs_adjgraph.jl

# 3. See real algorithm performance (flagship benchmark)
julia --project examples/dijkstra_benchmark.jl

# 4. Deep dive into optimization (advanced)
julia --project examples/micro_benchmark.jl

# 5. Create visualizations (requires plotting packages)
julia --project examples/plotting_example.jl
```
