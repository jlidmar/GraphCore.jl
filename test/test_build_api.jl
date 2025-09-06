using Test
using GraphCore
using .GraphCore.GraphConstruction

# Add a simple unified API test
@testset "Unified build_graph API" begin
    edges = [(1, 2), (2, 3), (1, 3)]
    
    # Test that all graph types can be built with the same function
    g1 = build_graph(CoreGraph, edges; directed=false)
    g2 = build_graph(AdjGraph, edges; directed=false)
    g3 = build_graph(WeightedGraph{Float64}, edges; weights=[1.0, 2.0, 1.5], directed=false)
    g4 = build_graph(WeightedAdjGraph{Float64}, edges; weights=[1.0, 2.0, 1.5], directed=false)
    
    # All should have same structure
    for g in [g1, g2, g3, g4]
        @test num_vertices(g) == 3
        @test num_edges(g) == 3
        @test !is_directed_graph(g)
    end
    
    # Test type-driven dispatch works
    @test g1 isa CoreGraph{false}
    @test g2 isa AdjGraph{false}
    @test g3 isa WeightedGraph{Float64,false}
    @test g4 isa WeightedAdjGraph{Float64,false}
end

@testset "Unified API Tests" begin
    
    @testset "Basic Auto-Detection" begin
        # Test vertex count auto-detection from edges
        edges = [(1, 2), (2, 3), (1, 3), (3, 5)]  # Max vertex = 5
        
        @testset "CoreGraph Auto-Detection" begin
            g = build_core_graph(edges; directed=false)
            @test num_vertices(g) == 5
            @test num_edges(g) == 4
            @test has_vertex(g, 5)
            @test has_edge(g, 3, 5)
        end
        
        @testset "AdjGraph Auto-Detection" begin
            g = build_adj_graph(edges; directed=true)
            @test num_vertices(g) == 5
            @test num_edges(g) == 4
            @test has_vertex(g, 5)
            @test has_edge(g, 3, 5)
        end
        
        @testset "WeightedGraph Auto-Detection" begin
            weights = [1.0, 2.0, 1.5, 2.5]
            g = build_weighted_graph(edges, weights; directed=false)
            @test num_vertices(g) == 5
            @test num_edges(g) == 4
            @test has_vertex(g, 5)
        end
    end
    
    @testset "Empty Graph Cases" begin
        
        @testset "Completely empty graph" begin
            g = build_core_graph(Tuple{Int,Int}[]; directed=false)
            @test num_vertices(g) == 0  # Truly empty graph
            @test num_edges(g) == 0
        end
    end
    
    @testset "Error Handling" begin
        edges = [(1, 2), (2, 3), (1, 3)]
        
        @testset "Weight length mismatch" begin
            bad_weights = [1.0, 2.0]  # Wrong length
            @test_throws ArgumentError build_weighted_graph(edges, bad_weights; directed=false)
        end
        
        @testset "Edge property length mismatch" begin
            vertex_props = ["A", "B", "C"]
            bad_edge_props = ["X", "Y"]  # Wrong length for 3 edges
            @test_throws ArgumentError build_property_graph(edges, vertex_props, bad_edge_props; directed=false)
        end
        
        @testset "Vertex property length mismatch" begin
            vertex_props = ["A", "B"]  # Length 2, but edges need 3 vertices
            edge_props = ["X", "Y", "Z"]
            @test_throws ArgumentError build_property_graph(edges, vertex_props, edge_props; directed=false)
        end
        
        @testset "Self-loops not supported" begin
            self_loop_edges = [(1, 1), (2, 3)]
            @test_throws ArgumentError build_core_graph(self_loop_edges; directed=false)
        end
        
        @testset "Invalid edge types" begin
            bad_edges = ["not", "edges"]
            @test_throws ArgumentError build_core_graph(bad_edges; directed=false)
        end
    end
    
    @testset "Validation Toggle" begin
        edges = [(1, 2), (2, 3)]
        
        @testset "Validation enabled (default)" begin
            # Should work fine with valid input
            g = build_core_graph(edges; directed=false, validate=true)
            @test num_vertices(g) == 3
            @test num_edges(g) == 2
        end
        
        @testset "Validation disabled" begin
            # Should work fine with valid input
            g = build_core_graph(edges; directed=false, validate=false)
            @test num_vertices(g) == 3
            @test num_edges(g) == 2
            
            # Invalid input should not throw when validation is off
            # (though results are undefined)
            # We can't easily test this without potentially crashing
        end
    end
    
    @testset "Unified build_graph Interface" begin
        edges = [(1, 2), (2, 3), (1, 3)]
        weights = [1.0, 2.0, 1.5]
        vertex_props = ["A", "B", "C"]
        edge_props = ["X", "Y", "Z"]
        
        @testset "CoreGraph via build_graph" begin
            g = build_graph(CoreGraph, edges; directed=false)
            @test g isa CoreGraph{false}
            @test num_vertices(g) == 3
            @test num_edges(g) == 3
        end
        
        @testset "WeightedGraph via build_graph" begin
            g = build_graph(WeightedGraph{Float64}, edges; weights=weights, directed=false)
            @test g isa WeightedGraph{Float64,false}
            @test num_vertices(g) == 3
            @test num_edges(g) == 3
        end
        
        @testset "PropertyGraph via build_graph" begin
            g = build_graph(PropertyGraph{CoreGraph,String,String}, edges; 
                           vertex_properties=vertex_props, edge_properties=edge_props, directed=false)
            @test g isa PropertyGraph{CoreGraph{false},String,String}
            @test num_vertices(g) == 3
            @test num_edges(g) == 3
        end
        
        @testset "AdjGraph via build_graph" begin
            g = build_graph(AdjGraph, edges; directed=true)
            @test g isa AdjGraph{true}
            @test num_vertices(g) == 3
            @test num_edges(g) == 3
        end
    end
    
    @testset "Cross-Type Compatibility" begin
        edges = [(1, 2), (2, 3), (1, 3)]
        weights = [1.0, 2.0, 1.5]
        
        # Build same graph with different types
        core_g = build_core_graph(edges; directed=false)
        adj_g = build_adj_graph(edges; directed=false)
        weighted_g = build_weighted_graph(edges, weights; directed=false)
        
        # All should have same structure
        @test num_vertices(core_g) == num_vertices(adj_g) == num_vertices(weighted_g)
        @test num_edges(core_g) == num_edges(adj_g) == num_edges(weighted_g)
        
        # All should have same connectivity
        for (u, v) in edges
            @test has_edge(core_g, u, v) == has_edge(adj_g, u, v) == has_edge(weighted_g, u, v)
            if !is_directed_graph(core_g)
                @test has_edge(core_g, v, u) == has_edge(adj_g, v, u) == has_edge(weighted_g, v, u)
            end
        end
    end
    
    @testset "Property Graph Edge Cases" begin
        @testset "Consistent property lengths" begin
            edges = [(1, 2), (2, 3)]
            vertex_props = ["A", "B", "C"]  # 3 vertices from edge max
            edge_props = ["X", "Y"]         # 2 edges
            
            pg = build_property_graph(edges, vertex_props, edge_props; directed=false)
            @test length(vertex_properties(pg)) == 3
            @test length(edge_properties(pg)) == 2
        end
        
        @testset "Property access" begin
            edges = [(1, 2), (2, 3)]
            vertex_props = ["Alice", "Bob", "Charlie"]
            edge_props = ["friend", "colleague"]
            
            pg = build_property_graph(edges, vertex_props, edge_props; directed=false)
            
            @test pg[1] == "Alice"
            @test pg[2] == "Bob"
            @test pg[3] == "Charlie"
            @test edge_property(pg, 1) == "friend"
            @test edge_property(pg, 2) == "colleague"
        end
    end
    
    @testset "Weighted Graph Edge Cases" begin
        @testset "Default weights for weighted types" begin
            edges = [(1, 2), (2, 3)]
            # No weights provided - should default to ones
            g = build_graph(WeightedGraph{Float64}, edges; directed=false)
            
            @test num_vertices(g) == 3
            @test num_edges(g) == 2
            
            # Check that default weights are 1.0
            for v in vertices(g)
                for weight in edge_weights(g, v)
                    @test weight == 1.0
                end
            end
        end
    end
end
