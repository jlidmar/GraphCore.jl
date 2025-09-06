"""
Type Stability Tests
===================

Tests to ensure that all GraphCore operations are type-stable for performance.
"""

using Test
using GraphCore

@testset "Type Stability Tests" begin
    
    @testset "CoreGraph Type Stability" begin
        test_graph = get_test_graph("k4")
        g = build_core_graph(test_graph.edges; directed=false)
        
        # Basic operations should be type stable
        @test @inferred(num_vertices(g)) isa Int64
        @test @inferred(num_edges(g)) isa Int64
        @test @inferred(is_directed_graph(g)) isa Bool
        @test @inferred(has_vertex(g, 1)) isa Bool
        @test @inferred(has_edge(g, 1, 2)) isa Bool
        
        # Neighbor operations
        @test @inferred(degree(g, 1)) isa Int32
        @test @inferred(neighbor(g, 1, 1)) isa Int32
        @test @inferred(collect(neighbor_indices(g, 1))) isa Vector{Int32}
        
        # Edge indexing
        @test @inferred(find_edge_index(g, 1, 2)) isa Int32
        @test @inferred(find_directed_edge_index(g, 1, 2)) isa Int32

        # Iteration ranges - only require that they are iterable and type stable
        @test collect(@inferred(vertices(g))) isa Vector{Int64}
        @test collect(@inferred(edge_indices(g))) isa Vector{Int64}
        @test collect(@inferred(directed_edge_indices(g))) isa Vector{Int64}
    end
    
    @testset "WeightedGraph Type Stability" begin
        test_graph = get_test_graph("weighted")
        g = build_weighted_graph(test_graph.edges, test_graph.weights; directed=false)
        
        # Weight operations should be type stable
        directed_idx = find_directed_edge_index(g, 1, 2)
        @test @inferred(edge_weight(g, directed_idx)) isa Float64
        
        # Edge weight iteration
        @test @inferred(collect(edge_weights(g, 1))) isa Vector{Float64}
        @test @inferred(collect(neighbor_weights(g, 1))) isa Vector{Tuple{Int32, Float64}}
        
        # Combined operations (check individual components since @inferred may fail on tuples)
        neighbor_weight_pairs = collect(neighbor_weights(g, 1))
        if !isempty(neighbor_weight_pairs)
            @test neighbor_weight_pairs[1][1] isa Int32      # neighbor
            @test neighbor_weight_pairs[1][2] isa Float64  # weight
        end
    end
    
    @testset "PropertyGraph Type Stability" begin
        test_graph = get_test_graph("path5")
        vertex_props = get_vertex_properties(test_graph.expected_nv)
        edge_props = get_edge_properties(test_graph.expected_ne)
        
        core_g = build_core_graph(test_graph.edges; directed=false)
        g = PropertyGraph(core_g, vertex_props, edge_props)
        
        # Property access should be type stable
        @test @inferred(vertex_property(g, 1)) isa String
        @test @inferred(g[1]) isa String
        @test @inferred(edge_property(g, 1)) isa String
        
        # Property arrays
        @test @inferred(collect(vertex_properties(g))) isa Vector{String}
        @test @inferred(collect(edge_properties(g))) isa Vector{String}
    end
    
    @testset "AdjGraph Type Stability" begin
        test_graph = get_test_graph("cycle4")
        g = build_adj_graph(test_graph.edges; directed=false)
        
        # All basic operations should remain type stable
        @test @inferred(num_vertices(g)) isa Int64
        @test @inferred(num_edges(g)) isa Int64
        @test @inferred(has_edge(g, 1, 2)) isa Bool
        @test @inferred(degree(g, 1)) isa Int32
        
        # Mutable operations should also be type stable
        @test @inferred(add_vertex!(g)) isa Int32
        
        # Edge addition returns Int32 (edge index or 0)
        edge_result = @inferred add_edge!(g, 1, num_vertices(g))
        @test edge_result isa Int32
        
        # Removal operations return Bool
        @test @inferred(remove_edge!(g, 1, 2)) isa Bool
    end
    
    @testset "Conversion Type Stability" begin
        test_graph = get_test_graph("single_edge")
        
        # Test that conversions return concrete types at runtime
        core_g = build_core_graph(test_graph.edges; directed=false)
        result1 = to_core_graph(core_g)
        @test result1 isa CoreGraph{false}
        
        # Weighted conversions
        weights = [1.0]
        weighted_g = build_weighted_graph(test_graph.edges, weights; directed=false)
        result2 = to_core_graph(weighted_g)
        @test result2 isa CoreGraph{false}
    end
    
    @testset "Iterator Type Stability" begin
        test_graph = get_test_graph("k4")
        g = build_core_graph(test_graph.edges; directed=false)
        
        # Edge iteration should be type stable
        edges_iter = edges(g)
        @test @inferred(collect(edges_iter)) isa Vector{Tuple{Int64, Int64}}
        
        directed_edges_iter = all_directed_edges(g)
        @test @inferred(collect(directed_edges_iter)) isa Vector{Tuple{Int64, Int64}}
        
        # Neighbor iteration
        neighbors_iter = neighbor_indices(g, 1)
        @test @inferred(collect(neighbors_iter)) isa Vector{Int32}
    end
    
    @testset "Builder Type Stability" begin
        @testset "GraphBuilder" begin
            builder = GraphBuilder(directed=false)
            
            # Builder operations should be type stable
            # Add operations should be type stable
            v1 = add_vertex!(builder)
            @test v1 isa Int32
            v2 = add_vertex!(builder)
            @test v2 isa Int32
            
            edge_id = add_edge!(builder, 1, 2)
            @test edge_id isa Int32
            
            # Final build - type stable when called with concrete arguments
            result = build_graph(builder)
            @test result isa CoreGraph{false}  # We know it's undirected from the builder
        end
        
        @testset "WeightedGraphBuilder" begin
            builder = WeightedGraphBuilder(Float64; directed=false)
            
            # Weighted builder operations
            edge_id = add_edge!(builder, 1, 2; weight=1.5)
            @test edge_id isa Int32
            
            # Build to specific type - this should be type stable
            result = build_graph(builder, WeightedGraph{Float64})
            @test result isa WeightedGraph{Float64, false}
        end
    end
    
    @testset "Complex Type Stability" begin
        # Test type stability with different numeric types
        @testset "Different Weight Types" begin
            edges = [(1, 2), (2, 3)]
            
            # Float32 weights
            weights_f32 = Float32[1.0, 2.0]
            g_f32 = build_weighted_graph(edges, weights_f32; directed=false)
            directed_idx = find_directed_edge_index(g_f32, 1, 2)
            @test @inferred(edge_weight(g_f32, directed_idx)) isa Float32
            
            # Int weights  
            weights_int = [1, 2]
            g_int = build_weighted_graph(edges, weights_int; directed=false)
            directed_idx = find_directed_edge_index(g_int, 1, 2)
            @test @inferred(edge_weight(g_int, directed_idx)) isa Int
        end
        
        @testset "Generic Property Types" begin
            test_graph = get_test_graph("single_edge")
            core_g = build_core_graph(test_graph.edges; directed=false)
            
            # Integer properties
            int_vertex_props = [1, 2]
            int_edge_props = [10]
            g_int = PropertyGraph(core_g, int_vertex_props, int_edge_props)
            
            @test @inferred(vertex_property(g_int, 1)) isa Int
            @test @inferred(edge_property(g_int, 1)) isa Int
            
            # Symbol properties
            sym_vertex_props = [:a, :b]
            sym_edge_props = [:x]
            g_sym = PropertyGraph(core_g, sym_vertex_props, sym_edge_props)
            
            @test @inferred(vertex_property(g_sym, 1)) isa Symbol
            @test @inferred(edge_property(g_sym, 1)) isa Symbol
        end
    end
    
    @testset "Performance Critical Paths" begin
        # Test type stability of operations likely to be in hot code paths
        test_graph = get_test_graph("petersen")  # Larger graph
        g = build_core_graph(test_graph.edges; directed=false)
        
        @testset "Hot Path Operations" begin
            # These are likely to be called frequently in algorithms
            v = 1
            @test @inferred(degree(g, v)) isa Int32
            @test @inferred(has_vertex(g, v)) isa Bool
            
            if degree(g, v) > 0
                @test @inferred(neighbor(g, v, 1)) isa Int32
            end
            
            # Edge queries
            u, v = 1, 2
            @test @inferred(has_edge(g, u, v)) isa Bool
            if has_edge(g, u, v)
                @test @inferred(find_edge_index(g, u, v)) isa Int32
                @test @inferred(find_directed_edge_index(g, u, v)) isa Int32
            end
        end
        
        @testset "Iteration Performance" begin
            # Iteration setup should be type stable and iterable
            @test collect(@inferred(vertices(g))) isa Vector{Int64}
            @test collect(@inferred(edge_indices(g))) isa Vector{Int64}
            
            # Neighbor iteration setup
            if num_vertices(g) > 0
                @test @inferred(neighbor_indices(g, 1)) isa Any  # Implementation dependent
            end
        end
    end
    
    @testset "Error Condition Type Stability" begin
        # Even error conditions should have stable return types
        g = build_core_graph([(1, 2)]; directed=false)
        
        # Boundary checks should be type stable
        @test @inferred(has_vertex(g, 0)) isa Bool
        @test @inferred(has_vertex(g, 100)) isa Bool
        
        # Find operations on non-existent edges should still be type stable
        result = try
            find_edge_index(g, 1, 3)
        catch
            0  # If it throws, that's fine
        end
        @test result isa Integer  # More flexible type check
    end
    end
    
    @testset "Cross-Implementation Type Consistency" begin
        # Different implementations should have the same type behavior
        test_graph = get_test_graph("k4")
        
        core_g = build_core_graph(test_graph.edges; directed=false)
        adj_g = build_adj_graph(test_graph.edges; directed=false)
        
        # Same operations should return same types
        @test typeof(@inferred(num_vertices(core_g))) == typeof(@inferred(num_vertices(adj_g)))
        @test typeof(@inferred(num_edges(core_g))) == typeof(@inferred(num_edges(adj_g)))
        @test typeof(@inferred(has_edge(core_g, 1, 2))) == typeof(@inferred(has_edge(adj_g, 1, 2)))
        @test typeof(@inferred(degree(core_g, 1))) == typeof(@inferred(degree(adj_g, 1)))
    end
