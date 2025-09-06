"""
Graph Construction Tests
========================

Test graph construction methods and GraphBuilder functionality.
"""

using Test
using GraphCore
using GraphCore.GraphConstruction

@testset "Graph Construction Tests" begin
    
    @testset "build_core_graph" begin
        test_graph = get_test_graph("k4")  # Use complete graph for testing
        
        @testset "Undirected Construction" begin
            g = build_core_graph(test_graph.edges; directed=false)
            
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            @test !is_directed_graph(g)
            
            # Test all edges exist and are bidirectional
            for (u, v) in test_graph.edges
                @test has_edge(g, u, v)
                @test has_edge(g, v, u)
            end
            
            # Test expected neighbor structure
            for (v, expected_neighbors) in test_graph.expected_neighbors
                actual_neighbors = sort(collect(neighbor_indices(g, v)))
                @test actual_neighbors == sort(expected_neighbors)
            end
        end
        
        @testset "Directed Construction" begin
            g = build_core_graph(test_graph.edges; directed=true)
            
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            @test is_directed_graph(g)
            
            # Test edges exist only in specified direction
            for (u, v) in test_graph.edges
                @test has_edge(g, u, v)
                @test !has_edge(g, v, u) || u == v  # No reverse unless self-loop
            end
        end
        
        @testset "Auto-Detection" begin
            # Test vertex count auto-detection
            edges = [(1, 3), (3, 7), (2, 5)]  # Max vertex = 7
            g = build_core_graph(edges; directed=false)
            @test num_vertices(g) == 7
            @test num_edges(g) == 3
            
            # Test with single edge
            single_edge = [(2, 5)]
            g_single = build_core_graph(single_edge; directed=false)
            @test num_vertices(g_single) == 5
            @test num_edges(g_single) == 1
        end
    end
    
    @testset "build_weighted_graph" begin
        test_graph = get_test_graph("weighted")
        
        @testset "Undirected Weighted" begin
            g = build_weighted_graph(test_graph.edges, test_graph.weights; directed=false)
            
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            @test !is_directed_graph(g)
            @test g isa WeightedGraphInterface
            
            # Test that all edges exist
            for (u, v) in test_graph.edges
                @test has_edge(g, u, v)
                @test has_edge(g, v, u)
            end
            
            # Test weight access
            for (i, (u, v)) in enumerate(test_graph.edges)
                directed_idx_uv = find_directed_edge_index(g, u, v)
                directed_idx_vu = find_directed_edge_index(g, v, u)
                
                # Both directions should have same weight
                @test edge_weight(g, directed_idx_uv) ≈ test_graph.weights[i]
                @test edge_weight(g, directed_idx_vu) ≈ test_graph.weights[i]
            end
            
            # Test neighbor weights iteration
            for v in vertices(g)
                neighbor_weight_pairs = collect(neighbor_weights(g, v))
                @test length(neighbor_weight_pairs) == degree(g, v)
                
                for (neighbor, weight) in neighbor_weight_pairs
                    @test neighbor in neighbor_indices(g, v)
                    @test weight isa Number
                    @test weight > 0  # Our test weights are positive
                end
            end
        end
        
        @testset "Directed Weighted" begin
            g = build_weighted_graph(test_graph.edges, test_graph.weights; directed=true)
            
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            @test is_directed_graph(g)
            @test g isa WeightedGraphInterface
            
            # Test directed edges only exist in one direction
            for (u, v) in test_graph.edges
                @test has_edge(g, u, v)
                @test !has_edge(g, v, u) || u == v
            end
        end
        
        @testset "Default Weights" begin
            # Test construction with unit weights
            edges = [(1, 2), (2, 3)]
            weights = [1.0, 1.0]  # Default unit weights
            g = build_weighted_graph(edges, weights; directed=false)
            
            @test num_vertices(g) == 3
            @test num_edges(g) == 2
            
            # Default weights should be 1.0
            for v in vertices(g)
                for weight in edge_weights(g, v)
                    @test weight == 1.0
                end
            end
        end
    end
    
    @testset "build_property_graph" begin
        test_graph = get_test_graph("path5")
        vertex_props = get_vertex_properties(test_graph.expected_nv)
        edge_props = get_edge_properties(test_graph.expected_ne)
        
        @testset "Basic Property Graph" begin
            g = build_property_graph(test_graph.edges, vertex_props, edge_props; directed=false)
            
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            @test !is_directed_graph(g)
            @test g isa PropertyGraphInterface
            
            # Test vertex property access
            for (v, expected_prop) in enumerate(vertex_props)
                @test vertex_property(g, v) == expected_prop
                @test g[v] == expected_prop  # Bracket notation
            end
            
            # Test edge property access
            for (edge_idx, expected_prop) in enumerate(edge_props)
                @test edge_property(g, edge_idx) == expected_prop
            end
            
            # Test property arrays
            @test collect(vertex_properties(g)) == vertex_props
            @test collect(edge_properties(g)) == edge_props
        end
        
        @testset "Property Graph with Different Types" begin
            # Test with different property types
            int_vertex_props = collect(1:test_graph.expected_nv)
            symbol_edge_props = [Symbol("edge_$i") for i in 1:test_graph.expected_ne]
            
            g = build_property_graph(test_graph.edges, int_vertex_props, symbol_edge_props; directed=false)
            
            @test g isa PropertyGraph{CoreGraph{false}, Int, Symbol}
            @test vertex_property(g, 1) isa Int
            @test edge_property(g, 1) isa Symbol
        end
    end
    
    @testset "build_adj_graph" begin
        test_graph = get_test_graph("cycle4")
        
        @testset "Basic AdjGraph" begin
            g = build_adj_graph(test_graph.edges; directed=false)
            
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            @test !is_directed_graph(g)
            @test g isa AdjGraph{false}
            
            # Test structure matches expected
            for (v, expected_neighbors) in test_graph.expected_neighbors
                actual_neighbors = sort(collect(neighbor_indices(g, v)))
                @test actual_neighbors == sort(expected_neighbors)
            end
        end
        
        @testset "Mutable Operations" begin
            g = build_adj_graph(test_graph.edges; directed=false)
            original_nv = num_vertices(g)
            original_ne = num_edges(g)
            
            # Test add_vertex!
            new_vertex = add_vertex!(g)
            @test new_vertex == original_nv + 1
            @test num_vertices(g) == original_nv + 1
            @test degree(g, new_vertex) == 0
            
            # Test add_edge!
            edge_idx = add_edge!(g, 1, new_vertex)
            @test edge_idx > 0
            @test num_edges(g) == original_ne + 1
            @test has_edge(g, 1, new_vertex)
            @test has_edge(g, new_vertex, 1)  # Undirected
        end
    end
    
    @testset "build_weighted_adj_graph" begin
        test_graph = get_test_graph("weighted")
        
        g = build_weighted_adj_graph(test_graph.edges, test_graph.weights; directed=false)
        
        @test num_vertices(g) == test_graph.expected_nv
        @test num_edges(g) == test_graph.expected_ne
        @test !is_directed_graph(g)
        @test g isa WeightedAdjGraph{Float64, false}
        
        # Test that it supports both weight and mutation interfaces
        @test g isa WeightedGraphInterface
        
        # Test mutable weighted operations
        original_ne = num_edges(g)
        add_vertex!(g)
        edge_idx = add_edge!(g, 1, num_vertices(g), 10.0)
        @test edge_idx > 0
        @test num_edges(g) == original_ne + 1
        
        # Test the new edge has correct weight
        directed_idx = find_directed_edge_index(g, 1, num_vertices(g))
        @test edge_weight(g, directed_idx) ≈ 10.0
    end
    
    @testset "Error Handling" begin
        @testset "Self-loops" begin
            edges_with_loop = [(1, 2), (2, 2), (2, 3)]
            @test_throws ArgumentError build_core_graph(edges_with_loop; directed=false)
            @test_throws ArgumentError build_adj_graph(edges_with_loop; directed=false)
        end
        
        @testset "Weight Length Mismatch" begin
            edges = [(1, 2), (2, 3)]
            weights = [1.0]  # Wrong length
            @test_throws ArgumentError build_weighted_graph(edges, weights; directed=false)
            @test_throws ArgumentError build_weighted_adj_graph(edges, weights; directed=false)
        end
        
        @testset "Property Length Mismatch" begin
            edges = [(1, 2), (2, 3)]
            vertex_props = ["A", "B", "C"]  # Correct: 3 vertices
            edge_props = ["X"]              # Wrong: 2 edges need 2 properties
            
            @test_throws ArgumentError build_property_graph(edges, vertex_props, edge_props; directed=false)
            
            # Wrong vertex property count
            bad_vertex_props = ["A", "B"]   # Wrong: need 3 for vertices 1,2,3
            good_edge_props = ["X", "Y"]    # Correct: 2 edges
            @test_throws ArgumentError build_property_graph(edges, bad_vertex_props, good_edge_props; directed=false)
        end
        
        @testset "Invalid Edge Types" begin
            bad_edges = ["not", "edges"]
            @test_throws ArgumentError build_core_graph(bad_edges; directed=false)
            
            # Non-tuple edges
            bad_edges2 = [1, 2, 3]
            @test_throws ArgumentError build_core_graph(bad_edges2; directed=false)
        end
        
        @testset "Invalid Vertices" begin
            # Zero or negative vertex indices
            bad_edges = [(0, 1), (1, 2)]
            @test_throws ArgumentError build_core_graph(bad_edges; directed=false)
            
            bad_edges2 = [(1, -1), (1, 2)]
            @test_throws ArgumentError build_core_graph(bad_edges2; directed=false)
        end
    end
    
    @testset "GraphBuilder Tests" begin
        @testset "Basic GraphBuilder" begin
            builder = GraphBuilder(directed=false)
            
            # Add vertices explicitly
            add_vertex!(builder)  # vertex 1
            add_vertex!(builder)  # vertex 2 
            add_vertex!(builder)  # vertex 3
            
            # Add edges
            add_edge!(builder, 1, 2)
            add_edge!(builder, 2, 3)
            add_edge!(builder, 1, 3)
            
            # Build final graph
            graph = build_graph(builder)
            @test num_vertices(graph) == 3
            @test num_edges(graph) == 3
            @test !is_directed_graph(graph)
            
            # Test connectivity
            @test has_edge(graph, 1, 2)
            @test has_edge(graph, 2, 3)
            @test has_edge(graph, 1, 3)
        end
        
        @testset "WeightedGraphBuilder" begin
            builder = WeightedGraphBuilder(Float64; directed=false)
            
            # Add edges with weights (vertices auto-created)
            add_edge!(builder, 1, 2; weight=1.5)
            add_edge!(builder, 2, 3; weight=2.0)
            add_edge!(builder, 1, 3; weight=1.0)
            
            # Build weighted graph
            graph = build_graph(builder, WeightedGraph{Float64})
            @test num_vertices(graph) == 3
            @test num_edges(graph) == 3
            @test graph isa WeightedGraph{Float64, false}
            
            # Test weights
            directed_idx = find_directed_edge_index(graph, 1, 2)
            @test edge_weight(graph, directed_idx) ≈ 1.5
        end
        
        @testset "PropertyGraphBuilder" begin
            builder = PropertyGraphBuilder(String, Symbol; directed=false)
            
            # Add vertices with properties
            add_vertex!(builder, "Alice")
            add_vertex!(builder, "Bob")
            add_vertex!(builder, "Charlie")
            
            # Add edges with properties  
            add_edge!(builder, 1, 2; edge_property=:friend)
            add_edge!(builder, 2, 3; edge_property=:colleague)
            
            # Build property graph
            graph = build_graph(builder, PropertyGraph{CoreGraph, String, Symbol})
            @test num_vertices(graph) == 3
            @test num_edges(graph) == 2
            
            # Test properties
            @test graph[1] == "Alice"
            @test graph[2] == "Bob"
            @test graph[3] == "Charlie"
            @test edge_property(graph, 1) == :friend
            @test edge_property(graph, 2) == :colleague
        end
    end
    
    @testset "Validation Toggle" begin
        edges = [(1, 2), (2, 3)]
        
        @testset "Validation Enabled (Default)" begin
            # Should work fine with valid input
            g = build_core_graph(edges; directed=false, validate=true)
            @test num_vertices(g) == 3
            @test num_edges(g) == 2
            
            # Should catch invalid input
            bad_edges = [(1, 1)]  # Self-loop
            @test_throws ArgumentError build_core_graph(bad_edges; directed=false, validate=true)
        end
        
        @testset "Validation Disabled" begin
            # Should work fine with valid input
            g = build_core_graph(edges; directed=false, validate=false)
            @test num_vertices(g) == 3
            @test num_edges(g) == 2
            
            # Invalid input should not throw (though results are undefined)
            # We can't safely test this without potentially crashing
        end
    end
    
    @testset "Empty and Minimal Cases" begin
        @testset "Empty Edge List" begin
            g = build_core_graph(Tuple{Int,Int}[]; directed=false)
            @test num_vertices(g) == 0  # Truly empty graph
            @test num_edges(g) == 0
        end
        
        @testset "Single Edge" begin
            edges = [(1, 2)]
            g = build_core_graph(edges; directed=false)
            @test num_vertices(g) == 2
            @test num_edges(g) == 1
            @test has_edge(g, 1, 2)
            @test has_edge(g, 2, 1)
        end
        
        @testset "Disconnected Components" begin
            edges = [(1, 2), (3, 4)]  # Two disconnected edges
            g = build_core_graph(edges; directed=false)
            @test num_vertices(g) == 4
            @test num_edges(g) == 2
            @test has_edge(g, 1, 2)
            @test has_edge(g, 3, 4)
            @test !has_edge(g, 1, 3)
            @test !has_edge(g, 2, 4)
        end
    end
end
