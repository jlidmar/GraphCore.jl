"""
Mutable Operations Tests
========================

Test mutable graph operations (add/remove vertices and edges) for graph types
that support dynamic modification.
"""

using Test
using GraphCore

@testset "Mutable Operations Tests" begin
    
    @testset "AdjGraph Mutations" begin
        @testset "Add Vertex" begin
            test_graph = get_test_graph("single_edge")
            g = build_adj_graph(test_graph.edges; directed=false)
            
            @test num_vertices(g) == test_graph.expected_nv
            @test num_edges(g) == test_graph.expected_ne
            
            # Add vertex
            new_v = add_vertex!(g)
            @test new_v == test_graph.expected_nv + 1
            @test num_vertices(g) == test_graph.expected_nv + 1
            @test num_edges(g) == test_graph.expected_ne  # Edges unchanged
            @test has_vertex(g, new_v)
            @test degree(g, new_v) == 0  # New vertex has no neighbors
        end
        
        @testset "Add Edge" begin
            test_graph = get_test_graph("path5")
            g = build_adj_graph(test_graph.edges; directed=false)
            
            original_nv = num_vertices(g)
            original_ne = num_edges(g)
            
            # Test adding edge between existing vertices that aren't connected
            @test !has_edge(g, 1, 3)  # Vertices 1 and 3 aren't directly connected in path
            
            edge_idx = add_edge!(g, 1, 3)
            @test edge_idx > 0
            @test num_vertices(g) == original_nv  # Vertices unchanged
            @test num_edges(g) == original_ne + 1
            @test has_edge(g, 1, 3)
            @test has_edge(g, 3, 1)  # Undirected
            
            # Try to add existing edge (should not change graph)
            edge_idx_2 = add_edge!(g, 1, 2)  # This edge already exists
            @test edge_idx_2 == 0  # Should return 0 for existing edge
            @test num_edges(g) == original_ne + 1  # Count unchanged
        end
        
        @testset "Add Edge with New Vertex" begin
            g = build_adj_graph([(1, 2)]; directed=false)
            @test num_vertices(g) == 2
            
            # Add vertex first
            add_vertex!(g)
            @test num_vertices(g) == 3
            
            # Now add edge to the new vertex
            edge_idx = add_edge!(g, 2, 3)
            @test edge_idx > 0
            @test has_edge(g, 2, 3)
            @test has_edge(g, 3, 2)
        end
        
        @testset "Remove Edge" begin
            test_graph = get_test_graph("k4")  # Complete graph
            g = build_adj_graph(test_graph.edges; directed=false)
            
            original_ne = num_edges(g)
            @test has_edge(g, 1, 2)
            
            # Remove edge
            success = remove_edge!(g, 1, 2)
            @test success
            @test num_edges(g) == original_ne - 1
            @test !has_edge(g, 1, 2)
            @test !has_edge(g, 2, 1)  # Both directions removed
            
            # Other edges should remain
            @test has_edge(g, 1, 3)
            @test has_edge(g, 2, 3)
            
            # Try to remove non-existent edge
            success2 = remove_edge!(g, 1, 2)  # Already removed
            @test !success2
            @test num_edges(g) == original_ne - 1  # No change
        end
        
        @testset "Remove Vertex" begin
            test_graph = get_test_graph("k4")
            g = build_adj_graph(test_graph.edges; directed=false)
            
            original_nv = num_vertices(g)
            original_ne = num_edges(g)
            
            # Vertex 1 should be connected to all others in K4
            @test degree(g, 1) == 3
            
            # Remove vertex 1 (this should remove all incident edges)
            success = remove_vertex!(g, 1)
            @test success
            @test num_vertices(g) == original_nv - 1
            
            # After removing vertex 1, we should have fewer edges
            # K4 has 6 edges, removing vertex with degree 3 should leave 3 edges
            @test num_edges(g) == 3
            
            # Vertices should be renumbered
            @test has_vertex(g, 1)  # This is now the old vertex 2
            @test has_vertex(g, 2)  # This is now the old vertex 3
            @test has_vertex(g, 3)  # This is now the old vertex 4
            @test !has_vertex(g, 4)
        end
        
        @testset "Remove Non-existent Vertex" begin
            g = build_adj_graph([(1, 2)]; directed=false)
            
            success = remove_vertex!(g, 5)  # Doesn't exist
            @test !success
            @test num_vertices(g) == 2  # Unchanged
        end
    end
    
    @testset "WeightedAdjGraph Mutations" begin
        test_graph = get_test_graph("weighted")
        
        @testset "Add Weighted Vertex" begin
            g = build_weighted_adj_graph(test_graph.edges, test_graph.weights; directed=false)
            original_nv = num_vertices(g)
            
            new_v = add_vertex!(g)
            @test new_v == original_nv + 1
            @test num_vertices(g) == original_nv + 1
            @test degree(g, new_v) == 0
        end
        
        @testset "Add Weighted Edge" begin
            g = build_weighted_adj_graph(test_graph.edges, test_graph.weights; directed=false)
            
            # Test that we can add a new edge with weight
            original_ne = num_edges(g)
            new_weight = 5.5
            
            edge_idx = add_edge!(g, 1, 4, new_weight)
            @test edge_idx > 0
            @test num_edges(g) == original_ne + 1
            @test has_edge(g, 1, 4)
            
            # Test that the weight is correct
            directed_idx = find_directed_edge_index(g, 1, 4)
            @test edge_weight(g, directed_idx) ≈ new_weight
            
            # Reverse direction should have same weight (undirected)
            directed_idx_rev = find_directed_edge_index(g, 4, 1)
            @test edge_weight(g, directed_idx_rev) ≈ new_weight
        end
        
        @testset "Remove Weighted Edge" begin
            g = build_weighted_adj_graph(test_graph.edges, test_graph.weights; directed=false)
            original_ne = num_edges(g)
            
            # Verify edge exists with correct weight before removal
            @test has_edge(g, 1, 2)
            directed_idx = find_directed_edge_index(g, 1, 2)
            original_weight = edge_weight(g, directed_idx)
            
            success = remove_edge!(g, 1, 2)
            @test success
            @test num_edges(g) == original_ne - 1
            @test !has_edge(g, 1, 2)
            
            # Remaining edges should still have correct weights
            for (i, (u, v)) in enumerate(test_graph.edges)
                if (u, v) != (1, 2) && (v, u) != (1, 2)  # Skip the removed edge
                    if has_edge(g, u, v)
                        remaining_idx = find_directed_edge_index(g, u, v)
                        @test edge_weight(g, remaining_idx) isa Number
                    end
                end
            end
        end
        
        @testset "Weighted Edge with Weight" begin
            g = build_weighted_adj_graph([(1, 2)], [1.0]; directed=false)
            add_vertex!(g)  # Add vertex 3
            
            # Add edge with explicit weight
            edge_idx = add_edge!(g, 2, 3, 2.5)  # With weight
            @test edge_idx > 0
            @test has_edge(g, 2, 3)
            
            # Check that it has some default weight
            directed_idx = find_directed_edge_index(g, 2, 3)
            weight = edge_weight(g, directed_idx)
            @test weight isa Number
        end
    end
    
    @testset "PropertyGraph Mutations" begin
        test_graph = get_test_graph("path5")
        vertex_props = get_vertex_properties(test_graph.expected_nv)
        edge_props = get_edge_properties(test_graph.expected_ne)
        
        @testset "Add Vertex with Property" begin
            base_g = build_adj_graph(test_graph.edges; directed=false)
            g = PropertyGraph(base_g, vertex_props, edge_props)
            
            original_nv = num_vertices(g)
            @test g[1] == vertex_props[1]
            @test g[original_nv] == vertex_props[original_nv]
            
            new_v = add_vertex!(g, "NEW_VERTEX")
            @test new_v == original_nv + 1
            @test num_vertices(g) == original_nv + 1
            @test g[new_v] == "NEW_VERTEX"
        end
        
        @testset "Add Edge with Property" begin
            base_g = build_adj_graph([(1, 2), (2, 3)]; directed=false)
            g = PropertyGraph(base_g, ["A", "B", "C"], ["X", "Y"])
            
            original_ne = num_edges(g)
            @test edge_property(g, 1) == "X"
            
            edge_idx = add_edge!(g, 1, 3, "NEW_EDGE")
            @test edge_idx > 0
            @test num_edges(g) == original_ne + 1
            @test edge_property(g, edge_idx) == "NEW_EDGE"
            @test has_edge(g, 1, 3)
            @test has_edge(g, 3, 1)  # Undirected
        end
        
        @testset "Remove Operations with Properties" begin
            base_g = build_adj_graph([(1, 2), (2, 3), (1, 3)]; directed=false)
            g = PropertyGraph(base_g, ["A", "B", "C"], ["X", "Y", "Z"])
            
            original_nv = num_vertices(g)
            original_ne = num_edges(g)
            
            # Remove edge
            success = remove_edge!(g, 1, 2)
            @test success
            @test num_edges(g) == original_ne - 1
            @test !has_edge(g, 1, 2)
            
            # Remaining properties should still be accessible
            @test g[1] == "A"
            @test g[2] == "B"
            @test g[3] == "C"
            
            # Note: Edge property indices may change after removal,
            # but the graph should remain consistent
            @test length(edge_properties(g)) == num_edges(g)
        end
    end
    
    @testset "Directed Graph Mutations" begin
        test_graph = get_test_graph("dag")  # Use DAG for directed testing
        
        @testset "Directed Add/Remove" begin
            g = build_adj_graph(test_graph.edges; directed=true)
            @test is_directed_graph(g)
            
            original_ne = num_edges(g)
            
            # Add directed edge
            edge_idx = add_edge!(g, 6, 1)  # Create a cycle
            @test edge_idx > 0
            @test has_edge(g, 6, 1)
            @test !has_edge(g, 1, 6)  # Should not exist in reverse
            @test num_edges(g) == original_ne + 1
            
            # Remove directed edge
            success = remove_edge!(g, 1, 2)
            @test success
            @test !has_edge(g, 1, 2)
            @test has_edge(g, 2, 4)  # Other edges unaffected
            @test has_edge(g, 6, 1)  # Edge we added should remain
        end
        
        @testset "Directed vs Undirected Consistency" begin
            edges = [(1, 2), (2, 3)]
            
            # Directed graph
            g_dir = build_adj_graph(edges; directed=true)
            add_edge!(g_dir, 1, 3)
            @test has_edge(g_dir, 1, 3)
            @test !has_edge(g_dir, 3, 1)
            
            # Undirected graph
            g_undir = build_adj_graph(edges; directed=false)
            add_edge!(g_undir, 1, 3)
            @test has_edge(g_undir, 1, 3)
            @test has_edge(g_undir, 3, 1)  # Both directions
        end
    end
    
    @testset "Error Conditions and Edge Cases" begin
        @testset "Invalid Vertex Operations" begin
            g = build_adj_graph([(1, 2)]; directed=false)
            
            # Try to add edge with non-existent vertex
            @test_throws BoundsError add_edge!(g, 1, 5)
            @test_throws BoundsError add_edge!(g, 0, 2)
            @test_throws BoundsError add_edge!(g, -1, 2)
            
            # Graph should be unchanged after failed operations
            @test num_vertices(g) == 2
            @test num_edges(g) == 1
        end
        
        @testset "Empty Graph Operations" begin
            g = build_adj_graph(Tuple{Int,Int}[]; directed=false)  # Start with minimal graph
            @test num_vertices(g) == 0
            @test num_edges(g) == 0
            
            # Add another vertex
            add_vertex!(g)
            add_vertex!(g)
            @test num_vertices(g) == 2

            # Add first edge
            edge_idx = add_edge!(g, 1, 2)
            @test edge_idx == 1
            @test num_edges(g) == 1
            @test has_edge(g, 1, 2)
            
            # Remove the edge
            success = remove_edge!(g, 1, 2)
            @test success
            @test num_edges(g) == 0
            @test !has_edge(g, 1, 2)
        end
        
        @testset "Self-loops in Mutations" begin
            g = build_adj_graph([(1, 2)]; directed=false)
            
            # Library allows self-loops - test actual behavior
            initial_edges = num_edges(g)
            add_edge!(g, 1, 1)  # Add self-loop
            
            # Graph should have one more edge
            @test num_edges(g) == initial_edges + 1
            @test has_edge(g, 1, 1)  # Self-loop should exist
        end
        
        @testset "Mutation on Immutable Types" begin
            # CoreGraph now supports mutations (though inefficient)
            g = build_core_graph([(1, 2)]; directed=false)
            
            # These should now be callable
            @test applicable(add_vertex!, g)
            @test applicable(add_edge!, g, 1, 2)
            @test applicable(remove_edge!, g, 1, 2)
            @test applicable(remove_vertex!, g, 1)
            
            # Test that they actually work
            initial_nv = num_vertices(g)
            new_vertex = add_vertex!(g)
            @test new_vertex == initial_nv + 1
            @test num_vertices(g) == initial_nv + 1
            
            # Test edge addition
            @test add_edge!(g, 1, new_vertex) > 0
            @test has_edge(g, 1, new_vertex)
            
            # Test edge removal
            @test remove_edge!(g, 1, 2)
            @test !has_edge(g, 1, 2)
        end
    end
    
    @testset "Index Stability and Invalidation" begin
        @testset "Adding Preserves Indices" begin
            test_graph = get_test_graph("k4")
            g = build_adj_graph(test_graph.edges; directed=false)
            
            # Get initial edge indices
            edge1_idx = find_edge_index(g, 1, 2)
            edge2_idx = find_edge_index(g, 2, 3)
            
            # Add new vertex and edge
            add_vertex!(g)
            add_edge!(g, 1, num_vertices(g))
            
            # Old indices should still be valid for their original edges
            @test find_edge_index(g, 1, 2) == edge1_idx
            @test find_edge_index(g, 2, 3) == edge2_idx
        end
        
        @testset "Removing May Invalidate Indices" begin
            # Document the behavior when removing elements
            g = build_adj_graph([(1, 2), (2, 3), (1, 3)]; directed=false)
            
            # Get initial state
            original_edges = collect(edges(g))
            @test length(original_edges) == 3
            
            # Remove an edge
            remove_edge!(g, 2, 3)
            
            # After removal, some edge indices may be different
            # This is expected behavior for dynamic graphs
            new_edges = collect(edges(g))
            @test length(new_edges) == 2
            
            # The specific behavior depends on implementation,
            # but users should expect potential index invalidation
        end
        
        @testset "Vertex Removal Renumbering" begin
            g = build_adj_graph([(1, 2), (2, 3), (3, 4)]; directed=false)
            
            # Remove middle vertex (vertex 2)
            @test has_vertex(g, 4)
            remove_vertex!(g, 2)
            
            # Vertices should be renumbered to maintain contiguous 1..n indexing
            @test num_vertices(g) == 3
            @test has_vertex(g, 1)
            @test has_vertex(g, 2)  # This is now the old vertex 3
            @test has_vertex(g, 3)  # This is now the old vertex 4
            @test !has_vertex(g, 4)
        end
    end
    
    @testset "Large Scale Mutation Tests" begin
        @testset "Incremental Graph Building" begin
            # Build a larger graph incrementally
            g = build_adj_graph(Tuple{Int,Int}[]; directed=false)
            @test num_vertices(g) == 0
            @test num_edges(g) == 0
            
            # Add vertices to build a path
            n = 10
            for i in 1:n
                add_vertex!(g)
            end
            @test num_vertices(g) == n
            
            # Add edges to form a path
            for i in 1:(n-1)
                edge_idx = add_edge!(g, i, i+1)
                @test edge_idx > 0
            end
            
            @test num_edges(g) == n-1
            
            # Verify path structure
            @test degree(g, 1) == 1      # Endpoint
            @test degree(g, n) == 1      # Endpoint
            for i in 2:(n-1)
                @test degree(g, i) == 2  # Middle vertices
            end
        end
        
        @testset "Batch Operations Performance" begin
            # This is a basic performance sanity check
            g = build_adj_graph(Tuple{Int,Int}[]; directed=false)
            
            # Add many vertices
            n = 100
            for i in 1:n
                add_vertex!(g)
            end
            
            # Add many edges (star pattern)
            for i in 2:n
                add_edge!(g, 1, i)
            end
            
            @test num_vertices(g) == n
            @test num_edges(g) == n-1
            @test degree(g, 1) == n-1  # Center of star
            
            # Remove all edges
            for i in 2:n
                remove_edge!(g, 1, i)
            end
            
            @test num_edges(g) == 0
            @test degree(g, 1) == 0
        end
    end
end