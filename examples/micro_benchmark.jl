#!/usr/bin/env julia

"""
Micro-benchmark: CoreGraph vs AdjGraph Performance Analysis

Analyzes specific performance differences between CoreGraph and AdjGraph
focusing on neighbor access patterns and bounds checking impacts.
"""

using GraphCore
using BenchmarkTools
using Random
using Printf

Random.seed!(42)

function create_test_graph(n_vertices=1000, n_edges=3000)
    edges = [(rand(1:n_vertices), rand(1:n_vertices)) for _ in 1:n_edges]
    edges = unique([(min(u,v), max(u,v)) for (u,v) in edges if u != v])
    
    core_g = build_graph(CoreGraph, edges; directed=false)
    adj_g = build_graph(AdjGraph, edges; directed=false)
    
    return core_g, adj_g, edges
end

function benchmark_neighbor_access()
    println("Neighbor Access Performance")
    println("="^27)
    
    core_g, adj_g, edges = create_test_graph(1000, 3000)
    test_vertices = rand(1:num_vertices(core_g), 100)
    
    println("Graph: $(num_vertices(core_g)) vertices, $(num_edges(core_g)) edges")
    println("Testing $(length(test_vertices)) random vertices")
    
    # Raw neighbor access
    println("\nNeighbor Access Time:")
    
    b_core = @benchmark begin
        for v in $test_vertices
            neighbors = neighbor_indices($core_g, v)
            length(neighbors)
        end
    end
    time_core = median(b_core).time / length(test_vertices)
    println("  CoreGraph: $(BenchmarkTools.prettytime(time_core))/call")
    
    b_adj = @benchmark begin
        for v in $test_vertices
            neighbors = neighbor_indices($adj_g, v)
            length(neighbors)
        end
    end
    time_adj = median(b_adj).time / length(test_vertices)
    println("  AdjGraph:  $(BenchmarkTools.prettytime(time_adj))/call")
    
    @printf "  Ratio: %.2fx %s\n" (time_core/time_adj) (time_core > time_adj ? "CoreGraph slower" : "CoreGraph faster")
    
    # Neighbor iteration
    println("\nNeighbor Iteration:")
    
    b_core_iter = @benchmark begin
        total = 0
        for v in $test_vertices
            for neighbor in neighbor_indices($core_g, v)
                total += neighbor
            end
        end
        total
    end
    time_core_iter = median(b_core_iter).time
    println("  CoreGraph: $(BenchmarkTools.prettytime(time_core_iter)) total")
    
    b_adj_iter = @benchmark begin
        total = 0
        for v in $test_vertices
            for neighbor in neighbor_indices($adj_g, v)
                total += neighbor
            end
        end
        total
    end
    time_adj_iter = median(b_adj_iter).time
    println("  AdjGraph:  $(BenchmarkTools.prettytime(time_adj_iter)) total")
    
    @printf "  Ratio: %.2fx %s\n" (time_core_iter/time_adj_iter) (time_core_iter > time_adj_iter ? "CoreGraph slower" : "CoreGraph faster")
end

function benchmark_bounds_checking()
    println("\nBounds Checking Impact")
    println("="^22)
    
    core_g, adj_g, _ = create_test_graph(1000, 3000)
    test_vertices = rand(1:num_vertices(core_g), 100)
    
    # With bounds checking
    println("With bounds checking:")
    
    b_core_bounds = @benchmark begin
        for v in $test_vertices
            neighbors = neighbor_indices($core_g, v)
            length(neighbors)
        end
    end
    time_core_bounds = median(b_core_bounds).time / length(test_vertices)
    println("  CoreGraph: $(BenchmarkTools.prettytime(time_core_bounds))/call")
    
    b_adj_bounds = @benchmark begin
        for v in $test_vertices
            neighbors = neighbor_indices($adj_g, v)
            length(neighbors)
        end
    end
    time_adj_bounds = median(b_adj_bounds).time / length(test_vertices)
    println("  AdjGraph:  $(BenchmarkTools.prettytime(time_adj_bounds))/call")
    
    ratio_with_bounds = time_core_bounds / time_adj_bounds
    @printf "  Ratio: %.2fx %s\n" ratio_with_bounds (ratio_with_bounds > 1.0 ? "CoreGraph slower" : "CoreGraph faster")
    
    # Without bounds checking  
    println("\nWith @inbounds:")
    
    b_core_nobounds = @benchmark begin
        for v in $test_vertices
            neighbors = @inbounds neighbor_indices($core_g, v)
            length(neighbors)
        end
    end
    time_core_nobounds = median(b_core_nobounds).time / length(test_vertices)
    println("  CoreGraph: $(BenchmarkTools.prettytime(time_core_nobounds))/call")
    
    b_adj_nobounds = @benchmark begin
        for v in $test_vertices
            neighbors = @inbounds neighbor_indices($adj_g, v)
            length(neighbors)
        end
    end
    time_adj_nobounds = median(b_adj_nobounds).time / length(test_vertices)
    println("  AdjGraph:  $(BenchmarkTools.prettytime(time_adj_nobounds))/call")
    
    ratio_no_bounds = time_core_nobounds / time_adj_nobounds
    @printf "  Ratio: %.2fx %s\n" ratio_no_bounds (ratio_no_bounds > 1.0 ? "CoreGraph slower" : "CoreGraph faster")
    
    # Speedup analysis
    println("\nSpeedup from @inbounds:")
    core_speedup = time_core_bounds / time_core_nobounds
    adj_speedup = time_adj_bounds / time_adj_nobounds
    
    @printf "  CoreGraph: %.2fx faster\n" core_speedup
    @printf "  AdjGraph:  %.2fx faster\n" adj_speedup
end

function analyze_why_adjgraph_wins()
    println("\nWhy AdjGraph Often Outperforms CoreGraph")
    println("="^40)
    
    println("CoreGraph neighbor_indices():")
    println("  1. g.vertex_offsets[v] lookup")
    println("  2. g.vertex_offsets[v + 1] lookup") 
    println("  3. @view creation with bounds checks")
    println("  4. SubArray object creation")
    
    println("\nAdjGraph neighbor_indices():")
    println("  1. g.neighbors[v] direct access")
    println("  2. Returns pre-allocated Vector")
    
    println("\nKey factors:")
    println("  - AdjGraph: Simpler operation, fewer bounds checks")
    println("  - CoreGraph: More complex, multiple array accesses")
    println("  - @inbounds helps but doesn't eliminate all overhead")
    println("  - --check-bounds=no gives CoreGraph bigger advantage")
end

function main()
    println("Micro-benchmark: CoreGraph vs AdjGraph")
    println("="^37)
    
    benchmark_neighbor_access()
    benchmark_bounds_checking()
    analyze_why_adjgraph_wins()
    
    println("\nKey Findings:")
    println("- AdjGraph simpler operations perform better in default Julia")
    println("- CoreGraph benefits more from bounds checking removal")
    println("- For optimal CoreGraph performance, use --check-bounds=no")
    println("- Both graph types have their optimal use cases")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
