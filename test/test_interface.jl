using Test
using GraphCore

"""
Interface Compliance Tests
==========================

Comprehensive tests ensuring all graph types correctly implement the GraphInterface.

This test suite validates that all concrete graph implementations (CoreGraph, 
WeightedGraph, PropertyGraph, AdjGraph, etc.) provide consistent behavior across
the common interface.
"""

"""
Test that a graph implements GraphInterface correctly.
"""
function test_graph_interface_compliance(g, test_graph::TestGraph, graph_type_name::String)
    @testset "$graph_type_name Interface Compliance" begin
        @testset "Basic Interface" begin
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            @test is_directed_graph(g) == test_graph.directed
            
            # Test vertex existence
            for v in 1:test_graph.expected_nv
                @test has_vertex(g, v)
            end
            @test !has_vertex(g, 0)
            @test !has_vertex(g, test_graph.expected_nv + 1)
            
            # Test iteration ranges
            @test vertices(g) == 1:test_graph.expected_nv
            @test edge_indices(g) == 1:test_graph.expected_ne
            @test directed_edge_indices(g) == 1:num_directed_edges(g)
        end
        
        @testset "Neighbor Access" begin
            for (v, expected_neighs) in test_graph.expected_neighbors
                if has_vertex(g, v)
                    actual_neighs = collect(neighbor_indices(g, v))
                    @test sort(actual_neighs) == sort(expected_neighs)
                    @test degree(g, v) == length(expected_neighs)
                    
                    # Test individual neighbor access
                    for (i, neigh) in enumerate(neighbor_indices(g, v))
                        @test neighbor(g, v, i) == neigh
                    end
                end
            end
        end
        
        @testset "Edge Queries" begin
            for (u, v) in test_graph.edges
                if has_vertex(g, u) && has_vertex(g, v)
                    @test has_edge(g, u, v)
                    if !test_graph.directed
                        @test has_edge(g, v, u)
                    end
                end
            end
        end
        
        @testset "Edge Indexing" begin
            for (u, v) in test_graph.edges
                if has_vertex(g, u) && has_vertex(g, v) && has_edge(g, u, v)
                    edge_idx = find_edge_index(g, u, v)
                    @test 1 <= edge_idx <= num_edges(g)
                    
                    directed_idx = find_directed_edge_index(g, u, v)
                    @test 1 <= directed_idx <= num_directed_edges(g)
                    
                    if !test_graph.directed
                        edge_idx_rev = find_edge_index(g, v, u)
                        @test edge_idx == edge_idx_rev
                    end
                end
            end
        end
        
        @testset "Edge Iteration" begin
            edges_list = collect(edges(g))
            @test length(edges_list) == num_edges(g)
            
            directed_edges_list = collect(all_directed_edges(g))
            @test length(directed_edges_list) == num_directed_edges(g)
            
            if !test_graph.directed
                @test num_directed_edges(g) == 2 * num_edges(g)
            else
                @test num_directed_edges(g) == num_edges(g)
            end
        end
    end
end

"""
Test weighted graph interface compliance.
"""
function test_weighted_interface_compliance(g, test_graph::TestGraph, graph_type_name::String)
    @testset "$graph_type_name Weighted Interface" begin
        @test g isa WeightedGraphInterface
        
        # Test weight access by vertex
        for v in vertices(g)
            weights_v = collect(edge_weights(g, v))
            @test length(weights_v) == degree(g, v)
            
            neighbor_weight_pairs = collect(neighbor_weights(g, v))
            @test length(neighbor_weight_pairs) == degree(g, v)
            
            for (i, (neighbor, weight)) in enumerate(neighbor_weight_pairs)
                @test neighbor in neighbor_indices(g, v)
                @test weight isa Number
            end
        end
        
        # Test weight access by edge index
        for directed_idx in directed_edge_indices(g)
            weight = edge_weight(g, directed_idx)
            @test weight isa Number
        end
        
        # Test specific weights if provided
        if !isempty(test_graph.weights)
            edge_weight_sum = 0.0
            for v in vertices(g)
                for weight in edge_weights(g, v)
                    edge_weight_sum += weight
                end
            end
            
            if !test_graph.directed
                # Undirected: each weight counted twice
                expected_sum = 2 * sum(test_graph.weights)
            else
                expected_sum = sum(test_graph.weights)
            end
            @test edge_weight_sum ≈ expected_sum
        end
    end
end

"""
Test property graph interface compliance.
"""
function test_property_interface_compliance(g, vertex_props, edge_props, graph_type_name::String)
    @testset "$graph_type_name Property Interface" begin
        @test g isa PropertyGraphInterface
        
        # Test vertex properties
        @test length(vertex_properties(g)) == num_vertices(g)
        for (v, expected_prop) in enumerate(vertex_props)
            if v <= num_vertices(g)
                @test vertex_property(g, v) == expected_prop
                @test g[v] == expected_prop
            end
        end
        
        # Test edge properties
        @test length(edge_properties(g)) == num_edges(g)
        for (edge_idx, expected_prop) in enumerate(edge_props)
            if edge_idx <= num_edges(g)
                @test edge_property(g, edge_idx) == expected_prop
            end
        end
        
        # Test property iteration
        actual_vertex_props = collect(vertex_properties(g))
        @test length(actual_vertex_props) == num_vertices(g)
        
        actual_edge_props = collect(edge_properties(g))
        @test length(actual_edge_props) == num_edges(g)
    end
end

"""
Test mutable graph interface compliance.
"""
function test_mutable_interface_compliance(g, graph_type_name::String)
    @testset "$graph_type_name Mutable Interface" begin
        original_nv = num_vertices(g)
        original_ne = num_edges(g)
        
        # Test add_vertex! if supported
        if applicable(add_vertex!, g)
            new_vertex = add_vertex!(g)
            @test num_vertices(g) == original_nv + 1
            @test new_vertex == original_nv + 1
            @test degree(g, new_vertex) == 0
            @test has_vertex(g, new_vertex)
        end
        
        # Test add_edge! if supported
        if applicable(add_edge!, g, 1, 2) && num_vertices(g) >= 2
            # Try to add an edge between existing vertices
            u, v = 1, min(2, num_vertices(g))
            if !has_edge(g, u, v)
                edge_idx = add_edge!(g, u, v)
                @test edge_idx > 0
                @test has_edge(g, u, v)
                if !is_directed_graph(g)
                    @test has_edge(g, v, u)
                end
            end
        end
    end
end

@testset "GraphInterface Compliance Tests" begin
    test_graphs = test_graph_list()
    
    for test_graph in test_graphs
        # Skip empty graphs for some tests
        if test_graph.name == "empty"
            continue
        end
        
        @testset "$(test_graph.name) Interface Tests" begin
            @testset "CoreGraph" begin
                g = build_core_graph(test_graph.edges; directed=test_graph.directed)
                test_graph_interface_compliance(g, test_graph, "CoreGraph")
            end
            
            @testset "AdjGraph" begin
                g = build_adj_graph(test_graph.edges; directed=test_graph.directed)
                test_graph_interface_compliance(g, test_graph, "AdjGraph")
                test_mutable_interface_compliance(g, "AdjGraph")
            end
            
            # Test weighted graphs only when weights are available
            if !isempty(test_graph.weights)
                @testset "WeightedGraph" begin
                    g = build_weighted_graph(test_graph.edges, test_graph.weights; 
                                            directed=test_graph.directed)
                    test_graph_interface_compliance(g, test_graph, "WeightedGraph")
                    test_weighted_interface_compliance(g, test_graph, "WeightedGraph")
                end
                
                @testset "WeightedAdjGraph" begin
                    g = build_weighted_adj_graph(test_graph.edges, test_graph.weights; 
                                                 directed=test_graph.directed)
                    test_graph_interface_compliance(g, test_graph, "WeightedAdjGraph")
                    test_weighted_interface_compliance(g, test_graph, "WeightedAdjGraph")
                    test_mutable_interface_compliance(g, "WeightedAdjGraph")
                end
            end
            
            # Test property graphs
            @testset "PropertyGraph" begin
                vertex_props = get_vertex_properties(test_graph.expected_nv)
                edge_props = get_edge_properties(test_graph.expected_ne)
                
                base_g = build_core_graph(test_graph.edges; directed=test_graph.directed)
                g = PropertyGraph(base_g, vertex_props, edge_props)
                
                test_graph_interface_compliance(g, test_graph, "PropertyGraph")
                test_property_interface_compliance(g, vertex_props, edge_props, "PropertyGraph")
            end
        end
    end
    
    @testset "Cross-Implementation Consistency" begin
        # Test that different implementations of the same graph produce identical results
        test_graph = get_test_graph("k4")  # Use complete graph for comprehensive testing
        
        implementations = [
            ("CoreGraph", () -> build_core_graph(test_graph.edges; directed=test_graph.directed)),
            ("AdjGraph", () -> build_adj_graph(test_graph.edges; directed=test_graph.directed))
        ]
        
        graphs = [impl[2]() for impl in implementations]
        
        @testset "Structural Consistency" begin
            for i in 1:length(graphs), j in (i+1):length(graphs)
                g1, g2 = graphs[i], graphs[j]
                name1, name2 = implementations[i][1], implementations[j][1]
                
                @testset "$name1 vs $name2" begin
                    @test num_vertices(g1) == num_vertices(g2)
                    @test num_edges(g1) == num_edges(g2)
                    @test is_directed_graph(g1) == is_directed_graph(g2)
                    
                    # Same edges
                    for (u, v) in test_graph.edges
                        @test has_edge(g1, u, v) == has_edge(g2, u, v)
                    end
                    
                    # Same neighbors
                    for v in vertices(g1)
                        neighbors1 = sort(collect(neighbor_indices(g1, v)))
                        neighbors2 = sort(collect(neighbor_indices(g2, v)))
                        @test neighbors1 == neighbors2
                    end
                end
            end
        end
    end
    
    @testset "Error Handling Tests" begin
        @testset "Construction Errors" begin
            edges = [(1, 2), (2, 3)]
            
            # Weight length mismatch
            bad_weights = [1.0]  # Wrong length
            @test_throws ArgumentError build_weighted_graph(edges, bad_weights; directed=false)
            
            # Self-loops (if not supported)
            self_loop_edges = [(1, 1), (2, 3)]
            @test_throws ArgumentError build_core_graph(self_loop_edges; directed=false)
            
            # Invalid edge types
            @test_throws ArgumentError build_core_graph(["not", "edges"]; directed=false)
        end
        
        @testset "Access Errors" begin
            g = build_core_graph([(1, 2), (2, 3)]; directed=false)
            
            # Invalid vertex access
            @test_throws BoundsError degree(g, 0)
            @test_throws BoundsError degree(g, 10)
            @test_throws BoundsError neighbor(g, 1, 10)
            
            # Edge access on invalid vertices
            @test_throws BoundsError !has_edge(g, 0, 1)
            @test_throws BoundsError has_edge(g, 1, 10)
        end
    end
    
    @testset "Empty and Minimal Graphs" begin
        @testset "Empty Graph" begin
            empty_test = get_test_graph("empty")
            g = build_core_graph(empty_test.edges; directed=false)
            test_graph_interface_compliance(g, empty_test, "Empty CoreGraph")
        end
        
        @testset "Single Edge Graph" begin
            single_edge_test = get_test_graph("single_edge")
            g = build_core_graph(single_edge_test.edges; directed=false)
            test_graph_interface_compliance(g, single_edge_test, "Single Edge CoreGraph")
        end
    end
end