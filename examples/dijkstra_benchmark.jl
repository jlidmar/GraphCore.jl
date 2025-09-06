#!/usr/bin/env julia

"""
Dijkstra's Algorithm Performance Benchmark

Compares Dijkstra's shortest path performance on different GraphCore 
graph types using Graphs.jl directly on GraphCore implementations.
"""

using GraphCore
using Graphs
using SimpleWeightedGraphs
using BenchmarkTools
using Random
using Printf

Random.seed!(42)

function to_simple_weighted_graph(g::WeightedGraph)
    weights = zeros(Float64, num_vertices(g), num_vertices(g))
    for u in 1:num_vertices(g)
        for (v, w) in zip(neighbor_indices(g, u), edge_weights(g, u))
            weights[u, v] = w
            weights[v, u] = w
        end
    end
    return SimpleWeightedGraph(weights)
end

function benchmark_dijkstra()
    println("Dijkstra Performance Benchmark")
    println("="^30)
    
    # Test graph: 1000 vertices, sparse
    edges = [(rand(1:1000), rand(1:1000)) for _ in 1:3000]
    edges = unique([(min(u,v), max(u,v)) for (u,v) in edges if u != v])
    weights = [rand(1.0:10.0) for _ in edges]
    
    println("\nTest graph: 1000 vertices, $(length(edges)) edges")
    
    # Build graphs
    core_g = build_graph(CoreGraph, edges; directed=false)
    adj_g = build_graph(AdjGraph, edges; directed=false)
    weighted_g = build_graph(WeightedGraph, edges; directed=false, weights=weights)
    simple_weighted_g = to_simple_weighted_graph(weighted_g)
    
    simple_g = SimpleGraph(1000)
    for (u, v) in edges
        Graphs.add_edge!(simple_g, u, v)
    end
    
    test_sources = rand(1:1000, 5)
    
    println("\nDijkstra Performance (5 queries):")
    
    # Benchmark each graph type
    b_simple = @benchmark begin
        for src in $test_sources
            dijkstra_shortest_paths($simple_g, src)
        end
    end
    time_simple = median(b_simple).time / length(test_sources)
    println("  SimpleGraph:   $(BenchmarkTools.prettytime(time_simple))/query")
    
    b_core = @benchmark begin
        for src in $test_sources
            dijkstra_shortest_paths($core_g, src)
        end
    end
    time_core = median(b_core).time / length(test_sources)
    println("  CoreGraph:     $(BenchmarkTools.prettytime(time_core))/query")
    
    b_adj = @benchmark begin
        for src in $test_sources
            dijkstra_shortest_paths($adj_g, src)
        end
    end
    time_adj = median(b_adj).time / length(test_sources)
    println("  AdjGraph:      $(BenchmarkTools.prettytime(time_adj))/query")
    
    b_weighted = @benchmark begin
        for src in $test_sources
            dijkstra_shortest_paths($simple_weighted_g, src)
        end
    end
    time_weighted = median(b_weighted).time / length(test_sources)
    println("  WeightedGraph: $(BenchmarkTools.prettytime(time_weighted))/query")
    
    # Performance ratios
    println("\nPerformance vs SimpleGraph:")
    @printf "  CoreGraph:     %.2fx %s\n" (time_core/time_simple) (time_core > time_simple ? "slower" : "faster")
    @printf "  AdjGraph:      %.2fx %s\n" (time_adj/time_simple) (time_adj > time_simple ? "slower" : "faster")
    @printf "  WeightedGraph: %.2fx %s\n" (time_weighted/time_simple) (time_weighted > time_simple ? "slower" : "faster")
end

function verify_correctness()
    println("\nCorrectness Verification")
    println("="^24)
    
    edges = [(1, 2), (2, 3), (3, 4), (1, 4), (2, 4)]
    weights = [1.0, 2.0, 1.0, 4.0, 1.0]
    
    core_g = build_graph(CoreGraph, edges; directed=false)
    weighted_g = build_graph(WeightedGraph, edges; directed=false, weights=weights)
    simple_weighted_g = to_simple_weighted_graph(weighted_g)
    
    result_core = dijkstra_shortest_paths(core_g, 1)
    result_weighted = dijkstra_shortest_paths(simple_weighted_g, 1)
    
    println("Distances from vertex 1:")
    println("  CoreGraph (unweighted): ", result_core.dists)
    println("  WeightedGraph (weighted):", result_weighted.dists)
    
    expected_unweighted = [0, 1, 2, 1]
    expected_weighted = [0.0, 1.0, 3.0, 2.0]
    
    println("  Correctness: ", result_core.dists == expected_unweighted && result_weighted.dists ≈ expected_weighted)
end

function main()
    verify_correctness()
    benchmark_dijkstra()
    
    println("\nSummary:")
    println("- CoreGraph: Optimized for static analysis")  
    println("- WeightedGraph: For weighted shortest paths")
    println("- AdjGraph: For dynamic graph modifications")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
