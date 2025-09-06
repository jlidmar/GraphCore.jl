"""
Basic Operations Tests
=====================

Tests for fundamental graph operations that all graph types should support.
This includes vertex/edge existence, degree calculations, neighbor access, etc.
"""

using Test
using GraphCore

"""
Test basic graph properties against expected values.
"""
function test_basic_properties(g, test_graph::TestGraph)
    @testset "Basic Properties - $(test_graph.name)" begin
        @test num_vertices(g) == test_graph.expected_nv
        @test num_edges(g) == test_graph.expected_ne
        @test is_directed_graph(g) == test_graph.directed
        
        # Test vertex existence
        for v in 1:test_graph.expected_nv
            @test has_vertex(g, v)
        end
        
        # Test boundary conditions
        @test !has_vertex(g, 0)
        @test !has_vertex(g, test_graph.expected_nv + 1)
        
        # Test directed edge count consistency
        if test_graph.directed
            @test num_directed_edges(g) == test_graph.expected_ne
        else
            @test num_directed_edges(g) == 2 * test_graph.expected_ne
        end
    end
end

"""
Test neighbor access and degree calculations.
"""
function test_neighbors(g, test_graph::TestGraph)
    @testset "Neighbors - $(test_graph.name)" begin
        for (v, expected_neighs) in test_graph.expected_neighbors
            if has_vertex(g, v)  # Skip if vertex doesn't exist in this instance
                actual_neighs = collect(neighbor_indices(g, v))
                
                # Sort for comparison (order might differ)
                @test sort(actual_neighs) == sort(expected_neighs)
                
                # Test degree
                @test degree(g, v) == length(expected_neighs)
                
                # Test individual neighbor access
                for (i, neigh) in enumerate(neighbor_indices(g, v))
                    @test neighbor(g, v, i) == neigh
                end
            end
        end
    end
end

"""
Test edge existence checks.
"""
function test_edge_existence(g, test_graph::TestGraph)
    @testset "Edge Existence - $(test_graph.name)" begin
        # Test all edges from the test graph
        for (u, v) in test_graph.edges
            if has_vertex(g, u) && has_vertex(g, v)  # Skip if vertices don't exist
                @test has_edge(g, u, v)
                if !test_graph.directed
                    @test has_edge(g, v, u)  # Undirected edges work both ways
                end
            end
        end
        
        # Test some non-existent edges (if graph isn't complete)
        nv = num_vertices(g)
        if nv >= 2 && num_edges(g) < nv * (nv - 1) ÷ 2
            # Find at least one missing edge
            found_missing = false
            for u in 1:nv, v in (u+1):nv
                if !has_edge(g, u, v)
                    @test !has_edge(g, u, v)  # Confirm it's missing
                    found_missing = true
                    break
                end
            end
            @test found_missing || num_edges(g) == nv * (nv - 1) ÷ 2  # Unless it's complete
        end
    end
end

"""
Test edge indexing and iteration.
"""
function test_edge_indexing(g, test_graph::TestGraph)
    @testset "Edge Indexing - $(test_graph.name)" begin
        # Test edge index retrieval
        for (u, v) in test_graph.edges
            if has_vertex(g, u) && has_vertex(g, v) && has_edge(g, u, v)
                edge_idx = find_edge_index(g, u, v)
                @test edge_idx > 0
                @test edge_idx <= num_edges(g)
                
                directed_idx = find_directed_edge_index(g, u, v)
                @test directed_idx > 0
                @test directed_idx <= num_directed_edges(g)
                
                if !test_graph.directed
                    # For undirected graphs, both directions map to same edge
                    edge_idx_rev = find_edge_index(g, v, u)
                    @test edge_idx == edge_idx_rev
                    
                    # But different directed indices
                    directed_idx_rev = find_directed_edge_index(g, v, u)
                    @test directed_idx != directed_idx_rev
                end
            end
        end
        
        # Test edge iteration
        actual_edges = collect(edges(g))
        if test_graph.directed
            # For directed graphs, should match exactly
            expected_canonical = test_graph.edges
        else
            # For undirected graphs, edges() returns canonical form (u ≤ v)
            expected_canonical = [(min(u,v), max(u,v)) for (u,v) in test_graph.edges]
            expected_canonical = unique(expected_canonical)
        end
        
        @test length(actual_edges) == length(expected_canonical)
        
        # Test directed edge iteration
        directed_edges = collect(all_directed_edges(g))
        @test length(directed_edges) == num_directed_edges(g)
    end
end

"""
Test vertex and edge ranges for iteration.
"""
function test_iteration_ranges(g, test_graph::TestGraph)
    @testset "Iteration Ranges - $(test_graph.name)" begin
        @test vertices(g) == 1:num_vertices(g)
        @test edge_indices(g) == 1:num_edges(g)
        @test directed_edge_indices(g) == 1:num_directed_edges(g)
        
        # Test that iteration works
        vertex_count = 0
        for v in vertices(g)
            vertex_count += 1
            @test 1 <= v <= num_vertices(g)
        end
        @test vertex_count == num_vertices(g)
        
        edge_count = 0
        for e_idx in edge_indices(g)
            edge_count += 1
            @test 1 <= e_idx <= num_edges(g)
        end
        @test edge_count == num_edges(g)
    end
end

"""
Run all basic operation tests on a graph.
"""
function test_basic_operations(g, test_graph::TestGraph)
    test_basic_properties(g, test_graph)
    test_neighbors(g, test_graph)
    test_edge_existence(g, test_graph)
    test_edge_indexing(g, test_graph)
    test_iteration_ranges(g, test_graph)
end

@testset "Basic Operations Tests" begin
    test_graphs = test_graph_list()
    
    for test_graph in test_graphs
        @testset "$(test_graph.name) tests" begin
            # Test with different graph implementations
            @testset "CoreGraph" begin
                g = build_core_graph(test_graph.edges; directed=test_graph.directed)
                test_basic_operations(g, test_graph)
            end
            
            @testset "AdjGraph" begin
                g = build_adj_graph(test_graph.edges; directed=test_graph.directed)
                test_basic_operations(g, test_graph)
            end
            
            # Only test weighted graphs with actual weights
            if !isempty(test_graph.weights)
                @testset "WeightedGraph" begin
                    g = build_weighted_graph(test_graph.edges, test_graph.weights; 
                                            directed=test_graph.directed)
                    test_basic_operations(g, test_graph)
                end
                
                @testset "WeightedAdjGraph" begin
                    g = build_weighted_adj_graph(test_graph.edges, test_graph.weights; 
                                                 directed=test_graph.directed)
                    test_basic_operations(g, test_graph)
                end
            end
        end
    end
    
    @testset "Edge Cases" begin
        @testset "Invalid Operations" begin
            g = build_core_graph([(1, 2), (2, 3)]; directed=false)
            
            # Test invalid vertex access
            @test_throws BoundsError neighbor(g, 1, 10)  # Invalid neighbor index
            @test_throws BoundsError degree(g, 0)       # Invalid vertex
            @test_throws BoundsError degree(g, 10)      # Invalid vertex
            
            # Test invalid edge queries - library returns false for invalid vertices
            @test_throws BoundsError has_edge(g, 1, 10)             # Invalid vertex
            @test_throws BoundsError has_edge(g, 10, 1) # Invalid vertex
        end
        
        @testset "Empty Graph Operations" begin
            empty_test = get_test_graph("empty")
            g = build_core_graph(empty_test.edges; directed=false)
            
            @test num_vertices(g) == 0
            @test num_edges(g) == 0
            @test_throws BoundsError degree(g, 1)
            @test_throws BoundsError neighbor_indices(g, 1)
            @test isempty(collect(edges(g)))
        end
    end
    
    @testset "Bounds Checking Tests" begin
        test_graph = get_test_graph("k4")
        g = build_core_graph(test_graph.edges; directed=false)
        
        @testset "@inbounds Disables Bounds Checking" begin
            # Test that @inbounds can disable bounds checking for vertex access
            # Note: This test verifies that @inbounds works, but the behavior 
            # when accessing out-of-bounds is undefined and may still throw or crash
            
            # First verify that normal access throws BoundsError
            @test_throws BoundsError neighbor_indices(g, 10)
            @test_throws BoundsError has_edge(g, 10, 1)
            
            # Test directed edge bounds checking 
            @test_throws BoundsError directed_edge_indices(g, 10)
            @test_throws BoundsError directed_edge_index(g, 10, 1)
            @test_throws BoundsError find_directed_edge_index(g, 10, 1)
            @test_throws BoundsError find_directed_edge_index(g, 1, 10)
            
            # Test that @inbounds at least doesn't cause compilation errors
            # and potentially disables bounds checking (behavior is implementation-dependent)
            try
                # These calls should compile successfully with @inbounds
                @inbounds neighbor_indices(g, 1)  # Valid access should always work
                @inbounds directed_edge_indices(g, 1)  # Valid directed edge access
                @inbounds find_directed_edge_index(g, 1, 2)  # Valid directed edge lookup
                @test true  # If we get here, @inbounds works for valid access
            catch e
                @test false
                @error "@inbounds should not cause errors for valid access: $e"
            end
            
            # Test property access bounds checking can be disabled
            if !isempty(test_graph.edges)
                vertex_props = ["v$i" for i in 1:num_vertices(g)]
                edge_props = ["e$i" for i in 1:num_edges(g)]
                prop_g = PropertyGraph(g, vertex_props, edge_props)
                
                # Verify normal bounds checking works
                @test_throws BoundsError vertex_property(prop_g, 10)
                @test_throws BoundsError edge_property(prop_g, 10)
                
                # Test that @inbounds works for valid access
                try
                    result = @inbounds vertex_property(prop_g, 1)
                    @test result == "v1"
                    result = @inbounds edge_property(prop_g, 1)
                    @test result == "e1"
                catch e
                    @test false
                    @error "@inbounds should not cause errors for valid property access: $e"
                end
            end
        end
    end
end
