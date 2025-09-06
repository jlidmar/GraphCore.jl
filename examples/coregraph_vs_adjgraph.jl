# examples/coregraph_vs_adjgraph.jl
# Focused comparison between CoreGraph (CSR) and AdjGraph (adjacency list) performance

using GraphCore
using BenchmarkTools
using Random
using Printf

"""
Demonstrate the performance trade-offs between CoreGraph and AdjGraph.
"""
function compare_graph_types()
    println("CoreGraph vs AdjGraph Performance Comparison")
    println("="^50)
    
    Random.seed!(42)
    
    # Test with different graph structures
    test_cases = [
        ("Small Dense", [(i, j) for i in 1:50 for j in i+1:50 if rand() < 0.3]),
        ("Medium Sparse", [(rand(1:1000), rand(1:1000)) for _ in 1:2000]),
        ("Large Sparse", [(rand(1:10000), rand(1:10000)) for _ in 1:15000])
    ]
    
    for (desc, raw_edges) in test_cases
        # Clean edges (remove duplicates and self-loops)
        edges = unique([(min(u,v), max(u,v)) for (u,v) in raw_edges if u != v])
        
        println("\n$desc Graph: $(length(edges)) edges")
        println("-" * "=" * "-"^(length(desc) + 20))
        
        # Create both graph types
        core_g = build_graph(CoreGraph, edges; directed=false)
        adj_g = build_graph(AdjGraph, edges; directed=false)
        
        nv = num_vertices(core_g)
        
        # 1. Construction time
        println("\n1. Construction Performance:")
        b_core = @benchmark build_graph(CoreGraph, $edges; directed=false)
        b_adj = @benchmark build_graph(AdjGraph, $edges; directed=false)
        
        t_core = median(b_core).time / 1e6  # Convert to ms
        t_adj = median(b_adj).time / 1e6
        
        @printf "   CoreGraph: %.2f ms\n" t_core
        @printf "   AdjGraph:  %.2f ms (%.1fx %s)\n" t_adj (t_adj/t_core) (t_adj > t_core ? "slower" : "faster")
        
        # 2. Neighbor access
        println("\n2. Neighbor Access (per vertex):")
        test_vertices = rand(1:nv, min(100, nv))
        
        b_core = @benchmark begin
            for v in $test_vertices
                neighbors = neighbor_indices($core_g, v)
                length(neighbors)
            end
        end
        
        b_adj = @benchmark begin
            for v in $test_vertices  
                neighbors = neighbor_indices($adj_g, v)
                length(neighbors)
            end
        end
        
        t_core = median(b_core).time / length(test_vertices)
        t_adj = median(b_adj).time / length(test_vertices)
        
        @printf "   CoreGraph: %.1f ns/vertex\n" t_core
        @printf "   AdjGraph:  %.1f ns/vertex (%.1fx %s)\n" t_adj (t_adj/t_core) (t_adj > t_core ? "slower" : "faster")
        
        # 3. Edge existence checking
        println("\n3. Edge Existence Queries:")
        test_edges = [(rand(1:nv), rand(1:nv)) for _ in 1:1000]
        
        b_core = @benchmark begin
            for (u, v) in $test_edges
                has_edge($core_g, u, v)
            end
        end
        
        b_adj = @benchmark begin
            for (u, v) in $test_edges
                has_edge($adj_g, u, v)
            end
        end
        
        t_core = median(b_core).time / length(test_edges)
        t_adj = median(b_adj).time / length(test_edges)
        
        @printf "   CoreGraph: %.1f ns/query\n" t_core
        @printf "   AdjGraph:  %.1f ns/query (%.1fx %s)\n" t_adj (t_adj/t_core) (t_adj > t_core ? "slower" : "faster")
        
        # 4. Full graph traversal (cache performance test)
        println("\n4. Full Graph Traversal:")
        
        b_core = @benchmark begin
            total = 0
            for v in 1:num_vertices($core_g)
                for neighbor in neighbor_indices($core_g, v)
                    total += neighbor
                end
            end
            total
        end
        
        b_adj = @benchmark begin
            total = 0
            for v in 1:num_vertices($adj_g)
                for neighbor in neighbor_indices($adj_g, v)
                    total += neighbor
                end
            end
            total
        end
        
        t_core = median(b_core).time / 1e6  # Convert to ms
        t_adj = median(b_adj).time / 1e6
        
        @printf "   CoreGraph: %.2f ms\n" t_core
        @printf "   AdjGraph:  %.2f ms (%.1fx %s)\n" t_adj (t_adj/t_core) (t_adj > t_core ? "slower" : "faster")
        
        # 5. Memory usage comparison
        println("\n5. Memory Characteristics:")
        @printf "   CoreGraph: %d vertices, %d edges\n" num_vertices(core_g) num_edges(core_g)
        @printf "   AdjGraph:  %d vertices, %d edges\n" num_vertices(adj_g) num_edges(adj_g)
        println("   (Actual memory usage depends on Julia's memory layout)")
    end
    
    println("\n" * "="^50)
    println("SUMMARY")
    println("="^50)
    println("CoreGraph (CSR format):")
    println("  ✅ Excellent cache performance for traversals")
    println("  ✅ Memory efficient, predictable layout")
    println("  ✅ Fast neighbor access")
    println("  ❌ Static structure - no efficient mutations")
    println("  ❌ Edge queries are O(degree)")
    
    println("\nAdjGraph (Adjacency Lists):")
    println("  ✅ Dynamic structure - O(1) add/remove operations")
    println("  ✅ Familiar adjacency list semantics")
    println("  ❌ Potentially less cache-friendly")
    println("  ❌ Edge queries are O(degree)")
    println("  ❌ More memory overhead from vectors")
    
    println("\nRecommendations:")
    println("  • Use CoreGraph for analysis of static graphs")
    println("  • Use AdjGraph for algorithms that modify graph structure")
    println("  • Both benefit from @inbounds in performance-critical loops")
end

# Demonstrate dynamic operations (where AdjGraph shines)
function demonstrate_dynamic_operations()
    println("\n" * "="^50)
    println("DYNAMIC OPERATIONS DEMO")
    println("="^50)
    
    # Start with small graphs for clear timing
    initial_edges = [(1, 2), (2, 3), (3, 4)]
    
    # Can't easily demonstrate CoreGraph mutations since they're not implemented
    # But we can show AdjGraph's efficiency
    println("\nAdjGraph Dynamic Operations:")
    adj_g = build_graph(AdjGraph, initial_edges; directed=false)
    
    # Add many edges
    println("Adding 1000 edges...")
    b = @benchmark begin
        g_copy = deepcopy($adj_g)
        nv = num_vertices(g_copy)
        for i in 1:1000
            u, v = rand(1:nv), rand(1:nv)
            if u != v && !has_edge(g_copy, u, v)
                add_edge!(g_copy, u, v)
            end
        end
    end
    println("  Time: $(BenchmarkTools.prettytime(median(b).time)) ($(BenchmarkTools.prettytime(median(b).time ÷ 1000)) per edge)")
    
    # Add many vertices
    println("Adding 1000 vertices...")
    b = @benchmark begin
        g_copy = deepcopy($adj_g)
        for i in 1:1000
            add_vertex!(g_copy)
        end
    end
    println("  Time: $(BenchmarkTools.prettytime(median(b).time)) ($(BenchmarkTools.prettytime(median(b).time ÷ 1000)) per vertex)")
    
    println("\nNote: CoreGraph mutations would require full reconstruction")
    println("(O(V + E) time), making AdjGraph much faster for dynamic workloads.")
end

# Run the comparison
if abspath(PROGRAM_FILE) == @__FILE__
    compare_graph_types()
    demonstrate_dynamic_operations()
end
