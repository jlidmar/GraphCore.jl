"""
Property Handling Tests
========================

Comprehensive tests for vertex and edge property access, modification, and type safety.

Tests cover:
- Basic property access via `vertex_property()` and `edge_property()` functions
- Convenient indexing syntax via `g[vertex]` and `g[edge_pair]`
- Property modification for mutable graph types
- Type safety and compile-time guarantees
- Property array length validation
- Combined property and weight functionality
"""

using Test
using GraphCore

@testset "Property Handling Tests" begin
    
    @testset "Vertex Properties" begin
        test_graph = get_test_graph("k4")  # Complete graph K4
        vertex_props = get_vertex_properties(test_graph.expected_nv)
        edge_props = get_edge_properties(test_graph.expected_ne)
        
        core_g = build_core_graph(test_graph.edges; directed=false)
        prop_g = PropertyGraph(core_g, vertex_props, edge_props)
        
        @testset "Property Access" begin
            # Test direct property function access
            for (v, expected_prop) in enumerate(vertex_props)
                @test vertex_property(prop_g, v) == expected_prop
                @test prop_g[v] == expected_prop  # Bracket notation
            end
            
            # Test property array iteration
            props = vertex_properties(prop_g)
            @test collect(props) == vertex_props
            @test length(props) == num_vertices(prop_g)
        end
        
        @testset "Property Modification" begin
            # Create mutable copies to test modification
            test_vertex_props = copy(vertex_props)
            test_edge_props = copy(edge_props)
            test_g = PropertyGraph(core_g, test_vertex_props, test_edge_props)
            
            # Test bracket assignment syntax
            original_prop = test_g[1]
            test_g[1] = "MODIFIED"
            @test test_g[1] == "MODIFIED"
            @test test_g[1] != original_prop
            
            # Test explicit property setter function
            set_vertex_property!(test_g, 2, "CHANGED")
            @test test_g[2] == "CHANGED"
            
            # Verify original graph is unchanged (immutability test)
            @test prop_g[1] == vertex_props[1]
            @test prop_g[2] == vertex_props[2]
        end
        
        @testset "Different Property Types" begin
            # Test with integer properties
            int_props = collect(1:test_graph.expected_nv)
            int_prop_g = PropertyGraph(core_g, int_props, edge_props)
            
            @test int_prop_g isa PropertyGraph{CoreGraph{false}, Int, String}
            @test vertex_property(int_prop_g, 1) isa Int
            @test int_prop_g[1] == 1
            
            # Test with symbol properties
            symbol_props = [Symbol("v$i") for i in 1:test_graph.expected_nv]
            symbol_prop_g = PropertyGraph(core_g, symbol_props, edge_props)
            
            @test symbol_prop_g isa PropertyGraph{CoreGraph{false}, Symbol, String}
            @test vertex_property(symbol_prop_g, 1) isa Symbol
            @test symbol_prop_g[1] == :v1
        end
    end
    
    @testset "Edge Properties" begin
        test_graph = get_test_graph("path5")  # Path graph for simple edge structure
        vertex_props = collect(1:test_graph.expected_nv)  # Integer vertex properties
        edge_props = [Symbol("edge_$i") for i in 1:test_graph.expected_ne]  # Symbol edge properties
        
        core_g = build_core_graph(test_graph.edges; directed=false)
        prop_g = PropertyGraph(core_g, vertex_props, edge_props)
        
        @testset "Edge Property Access" begin
            # Test direct edge property access by index
            for (edge_idx, expected_prop) in enumerate(edge_props)
                @test edge_property(prop_g, edge_idx) == expected_prop
            end
            
            # Test edge property array iteration
            props = edge_properties(prop_g)
            @test collect(props) == edge_props
            @test length(props) == num_edges(prop_g)
        end
        
        @testset "Edge Property by Vertex Pair" begin
            # Test accessing edge properties via vertex pairs
            for (i, (u, v)) in enumerate(test_graph.edges)
                edge_idx = find_edge_index(prop_g, u, v)
                edge_prop = edge_property(prop_g, edge_idx)
                @test edge_prop == edge_props[edge_idx]
                
                # For undirected graphs, both directions should give same result
                edge_idx_rev = find_edge_index(prop_g, v, u)
                @test edge_idx == edge_idx_rev
            end
        end
        
        @testset "Edge Property Modification" begin
            # Test edge property modification on mutable copies
            test_vertex_props = copy(vertex_props)
            test_edge_props = copy(edge_props)
            test_g = PropertyGraph(core_g, test_vertex_props, test_edge_props)
            
            original_prop = edge_property(test_g, 1)
            set_edge_property!(test_g, 1, :modified_edge)
            @test edge_property(test_g, 1) == :modified_edge
            @test edge_property(test_g, 1) != original_prop
            
            # Verify original unchanged
            @test edge_property(prop_g, 1) == edge_props[1]
        end
    end
    
    @testset "Weighted Graph Properties" begin
        test_graph = get_test_graph("weighted")
        vertex_props = get_vertex_properties(test_graph.expected_nv)
        edge_props = get_edge_properties(test_graph.expected_ne)
        
        weighted_g = build_weighted_graph(test_graph.edges, test_graph.weights; directed=false)
        prop_weighted_g = PropertyGraph(weighted_g, vertex_props, edge_props)
        
        @testset "Combined Weight and Property Access" begin
            # Should have both weight and property interfaces
            @test prop_weighted_g isa PropertyGraphInterface
            @test prop_weighted_g.core isa WeightedGraphInterface
            
            # Test vertex properties
            for (v, expected_prop) in enumerate(vertex_props)
                @test prop_weighted_g[v] == expected_prop
            end
            
            # Test edge properties
            for (edge_idx, expected_prop) in enumerate(edge_props)
                @test edge_property(prop_weighted_g, edge_idx) == expected_prop
            end
            
            # Test weight access through PropertyGraph
            for (i, (u, v)) in enumerate(test_graph.edges)
                directed_idx = find_directed_edge_index(prop_weighted_g, u, v)
                weight = edge_weight(prop_weighted_g, directed_idx)
                @test weight ≈ test_graph.weights[i]
            end
            
            # Test neighbor_weights through PropertyGraph
            for v in vertices(prop_weighted_g)
                weight_pairs = collect(neighbor_weights(prop_weighted_g, v))
                @test length(weight_pairs) == degree(prop_weighted_g, v)
                
                for (neighbor, weight) in weight_pairs
                    @test neighbor in neighbor_indices(prop_weighted_g, v)
                    @test weight isa Number
                    @test weight > 0  # Our test weights are positive
                end
            end
        end
        
        @testset "Property and Weight Consistency" begin
            # Properties and weights should be independent
            # Changing properties shouldn't affect weights and vice versa
            
            # Test modification doesn't affect weights
            original_weight = edge_weight(prop_weighted_g, find_directed_edge_index(prop_weighted_g, 1, 2))
            set_edge_property!(prop_weighted_g, 1, "MODIFIED")
            new_weight = edge_weight(prop_weighted_g, find_directed_edge_index(prop_weighted_g, 1, 2))
            @test original_weight ≈ new_weight
        end
    end
    
    @testset "Property Type Safety" begin
        test_graph = get_test_graph("single_edge")
        core_g = build_core_graph(test_graph.edges; directed=false)
        
        @testset "Homogeneous Types" begin
            # Should work with consistent types
            vertex_props_int = [10, 20]
            edge_props_float = [3.14]
            
            prop_g = PropertyGraph(core_g, vertex_props_int, edge_props_float)
            
            @test prop_g isa PropertyGraph{CoreGraph{false}, Int, Float64}
            @test vertex_property(prop_g, 1) isa Int
            @test edge_property(prop_g, 1) isa Float64
        end
        
        @testset "Mixed Types" begin
            # Julia should handle mixed types via Union types
            mixed_vertex_props = [1, "two"]  # Int and String
            edge_props = [:edge1]
            
            prop_g = PropertyGraph(core_g, mixed_vertex_props, edge_props)
            @test prop_g isa PropertyGraph
            @test vertex_property(prop_g, 1) == 1
            @test vertex_property(prop_g, 2) == "two"
        end
        
        @testset "Complex Types" begin
            # Test with custom struct types
            struct CustomVertexData
                id::Int
                name::String
            end
            
            struct CustomEdgeData
                weight::Float64
                label::String
            end
            
            custom_vertex_props = [CustomVertexData(1, "A"), CustomVertexData(2, "B")]
            custom_edge_props = [CustomEdgeData(1.5, "connection")]
            
            prop_g = PropertyGraph(core_g, custom_vertex_props, custom_edge_props)
            
            @test prop_g isa PropertyGraph{CoreGraph{false}, CustomVertexData, CustomEdgeData}
            @test vertex_property(prop_g, 1).id == 1
            @test vertex_property(prop_g, 1).name == "A"
            @test edge_property(prop_g, 1).weight ≈ 1.5
            @test edge_property(prop_g, 1).label == "connection"
        end
    end
    
    @testset "Property Array Length Validation" begin
        test_graph = get_test_graph("k4")
        core_g = build_core_graph(test_graph.edges; directed=false)
        
        @testset "Correct Lengths" begin
            vertex_props = get_vertex_properties(test_graph.expected_nv)
            edge_props = get_edge_properties(test_graph.expected_ne)
            
            prop_g = PropertyGraph(core_g, vertex_props, edge_props)
            @test prop_g isa PropertyGraph
            @test length(vertex_properties(prop_g)) == test_graph.expected_nv
            @test length(edge_properties(prop_g)) == test_graph.expected_ne
        end
        
        @testset "Incorrect Vertex Property Length" begin
            vertex_props = ["A", "B"]  # Only 2 props for 4 vertices
            edge_props = get_edge_properties(test_graph.expected_ne)
            
            @test_throws AssertionError PropertyGraph(core_g, vertex_props, edge_props)
        end
        
        @testset "Incorrect Edge Property Length" begin
            vertex_props = get_vertex_properties(test_graph.expected_nv)
            edge_props = ["X", "Y"]  # Only 2 props for 6 edges in K4
            
            @test_throws AssertionError PropertyGraph(core_g, vertex_props, edge_props)
        end
    end
    
    @testset "Property Graph Construction Helpers" begin
        test_graph = get_test_graph("cycle4")
        
        @testset "build_property_graph helper" begin
            vertex_props = ["A", "B", "C", "D"]
            edge_props = ["e1", "e2", "e3", "e4"]
            
            prop_g = build_property_graph(test_graph.edges, vertex_props, edge_props; directed=false)
            
            @test num_vertices(prop_g) == test_graph.expected_nv
            @test num_edges(prop_g) == test_graph.expected_ne
            @test !is_directed_graph(prop_g)
            
            # Test properties are correctly assigned
            for (v, expected_prop) in enumerate(vertex_props)
                @test prop_g[v] == expected_prop
            end
            for (e, expected_prop) in enumerate(edge_props)
                @test edge_property(prop_g, e) == expected_prop
            end
        end
        
        @testset "Property-first construction" begin
            # Test auto-detecting vertex count from property length
            vertex_props = ["X", "Y", "Z"]  # 3 vertices
            edge_props = ["A", "B"]         # 2 edges
            edges = [(1, 2), (2, 3)]        # Matches 2 edges, uses vertices 1,2,3
            
            prop_g = build_property_graph(edges, vertex_props, edge_props; directed=false)
            @test num_vertices(prop_g) == 3
            @test num_edges(prop_g) == 2
        end
    end
    
    @testset "Performance and Memory Tests" begin
        # Basic performance sanity checks
        @testset "Large Property Graph" begin
            # Create a moderately large graph
            n = 100
            edges = [(i, i+1) for i in 1:(n-1)]  # Path graph
            vertex_props = ["vertex_$i" for i in 1:n]
            edge_props = ["edge_$i" for i in 1:(n-1)]
            
            prop_g = build_property_graph(edges, vertex_props, edge_props; directed=false)
            
            @test num_vertices(prop_g) == n
            @test num_edges(prop_g) == n-1
            
            # Test property access is still efficient
            @test prop_g[1] == "vertex_1"
            @test prop_g[n] == "vertex_$n"
            @test edge_property(prop_g, 1) == "edge_1"
            @test edge_property(prop_g, n-1) == "edge_$(n-1)"
        end
    end
    
    @testset "Edge Cases and Error Conditions" begin
        @testset "Empty Property Arrays" begin
            # Test with minimal graph
            empty_test = get_test_graph("empty")
            core_g = build_core_graph(empty_test.edges; directed=false)

            vertex_props = String[]
            edge_props = String[]

            prop_g = PropertyGraph(core_g, vertex_props, edge_props)
            @test num_vertices(prop_g) == 0
            @test num_edges(prop_g) == 0
            @test_throws BoundsError prop_g[1]
        end
        
        @testset "Property Access Bounds" begin
            test_graph = get_test_graph("single_edge")
            prop_g = build_property_graph(test_graph.edges, ["A", "B"], ["E"]; directed=false)
            
            # Valid access
            @test prop_g[1] == "A"
            @test prop_g[2] == "B"
            @test edge_property(prop_g, 1) == "E"
            
            # Invalid access should throw bounds errors
            @test_throws BoundsError prop_g[0]
            @test_throws BoundsError prop_g[3]
            @test_throws BoundsError edge_property(prop_g, 0)
            @test_throws BoundsError edge_property(prop_g, 2)
        end
    end
end