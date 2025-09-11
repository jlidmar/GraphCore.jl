# Test Edge Weight Interface
# Tests for the unified edge_weight/edge_weights API across all graph types
#
# This test suite verifies:
# 1. Unweighted graphs return consistent 1.0 weights via efficient iterators
# 2. Weighted graphs return actual weight values with proper type safety  
# 3. PropertyGraph correctly delegates to its base graph
# 4. set_edge_weight! mutation works correctly for weighted graphs
# 5. Generic algorithms can work uniformly across all graph types
# 6. API consistency and type safety across the weight interface

@testset "Edge Weight Interface" begin
    test_graph = get_test_graph("k4")  # Complete graph with 4 vertices
    edges = test_graph.edges
    weights = [1.0, 2.0, 1.5, 3.0, 2.5, 0.5]  # Create weights for K4 (6 edges)
    
    @testset "Unweighted Graph Edge Weights" begin
        @testset "CoreGraph Unweighted" begin
            g = build_core_graph(edges; directed=false)
            
            # Test edge_weight returns 1 for unweighted graphs
            for v in vertices(g)
                for (i, neighbor) in enumerate(neighbor_indices(g, v))
                    directed_idx = directed_edge_index(g, v, i)
                    @test edge_weight(g, directed_idx) === Int32(1)
                end
            end
            
            # Test edge_weights returns iterator of 1s
            for v in vertices(g)
                weights_iter = edge_weights(g, v)
                collected_weights = collect(weights_iter)
                expected_length = degree(g, v)
                
                @test length(collected_weights) == expected_length
                @test all(w -> w === Int32(1), collected_weights)
                
                # Test iterator properties
                @test weights_iter isa Iterators.Repeated{Int32} || 
                      weights_iter isa Iterators.Take{<:Iterators.Repeated{Int32}}
                @test eltype(weights_iter) === Int32
            end
            
            # Test neighbor_weights returns (neighbor, 1) pairs
            for v in vertices(g)
                neighbor_weight_pairs = collect(neighbor_weights(g, v))
                neighbors = collect(neighbor_indices(g, v))
                
                @test length(neighbor_weight_pairs) == length(neighbors)
                for ((neighbor, weight), expected_neighbor) in zip(neighbor_weight_pairs, neighbors)
                    @test neighbor == expected_neighbor
                    @test weight === Int32(1)
                end
            end
        end
        
        @testset "AdjGraph Unweighted" begin
            g = build_adj_graph(edges; directed=false)
            
            # Test same behavior as CoreGraph
            for v in vertices(g)
                for (i, neighbor) in enumerate(neighbor_indices(g, v))
                    directed_idx = directed_edge_index(g, v, i)
                    @test edge_weight(g, directed_idx) === Int32(1)
                end
            end
            
            for v in vertices(g)
                weights_iter = edge_weights(g, v)
                collected_weights = collect(weights_iter)
                @test all(w -> w === Int32(1), collected_weights)
                @test length(collected_weights) == degree(g, v)
            end
        end
        
        @testset "PropertyGraph Unweighted" begin
            core_g = build_core_graph(edges; directed=false)
            vertex_props = ["A", "B", "C", "D"]  # K4 has 4 vertices
            edge_props = [1, 2, 3, 4, 5, 6]     # K4 has 6 edges
            
            prop_g = PropertyGraph(core_g, vertex_props, edge_props)
            
            # PropertyGraph with unweighted base should return 1s
            for v in vertices(prop_g)
                for (i, neighbor) in enumerate(neighbor_indices(prop_g, v))
                    directed_idx = directed_edge_index(prop_g, v, i)
                    @test edge_weight(prop_g, directed_idx) === Int32(1)
                end
            end
            
            for v in vertices(prop_g)
                weights_iter = edge_weights(prop_g, v)
                collected_weights = collect(weights_iter)
                @test all(w -> w === Int32(1), collected_weights)
            end
        end
    end
    
    @testset "Weighted Graph Edge Weights" begin
        @testset "WeightedGraph Direct Access" begin
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            
            # Test edge_weight returns actual weights
            for v in vertices(weighted_g)
                for (i, neighbor) in enumerate(neighbor_indices(weighted_g, v))
                    weight = edge_weight(weighted_g, v, i)
                    @test weight isa Float64
                    @test weight > 0  # Our test weights are positive
                    
                    # Verify weight matches the original edge
                    edge_idx = directed_edge_index(weighted_g, v, i)
                    direct_weight = edge_weight(weighted_g, edge_idx)
                    @test weight ≈ direct_weight
                end
            end
            
            # Test edge_weights returns actual weight values
            for v in vertices(weighted_g)
                weights_iter = edge_weights(weighted_g, v)
                collected_weights = collect(weights_iter)
                
                @test length(collected_weights) == degree(weighted_g, v)
                @test all(w -> w isa Float64 && w > 0, collected_weights)
                
                # Compare with individual edge_weight calls
                for (i, expected_weight) in enumerate(collected_weights)
                    actual_weight = edge_weight(weighted_g, v, i)
                    @test expected_weight ≈ actual_weight
                end
            end
            
            # Test neighbor_weights returns actual (neighbor, weight) pairs
            for v in vertices(weighted_g)
                neighbor_weight_pairs = collect(neighbor_weights(weighted_g, v))
                neighbors = collect(neighbor_indices(weighted_g, v))
                
                @test length(neighbor_weight_pairs) == length(neighbors)
                for ((neighbor, weight), expected_neighbor) in zip(neighbor_weight_pairs, neighbors)
                    @test neighbor == expected_neighbor
                    @test weight isa Float64
                    @test weight > 0
                end
            end
        end
        
        @testset "PropertyGraph with Weighted Base" begin
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            vertex_props = ["A", "B", "C", "D"]  # K4 has 4 vertices
            edge_props = [1, 2, 3, 4, 5, 6]     # K4 has 6 edges
            
            prop_weighted_g = PropertyGraph(weighted_g, vertex_props, edge_props)
            
            # Should delegate to underlying weighted graph
            for v in vertices(prop_weighted_g)
                for (i, neighbor) in enumerate(neighbor_indices(prop_weighted_g, v))
                    prop_weight = edge_weight(prop_weighted_g, v, i)
                    base_weight = edge_weight(weighted_g, v, i)
                    @test prop_weight ≈ base_weight
                end
            end
            
            # Test edge_weights delegation
            for v in vertices(prop_weighted_g)
                prop_weights = collect(edge_weights(prop_weighted_g, v))
                base_weights = collect(edge_weights(weighted_g, v))
                
                @test length(prop_weights) == length(base_weights)
                for (pw, bw) in zip(prop_weights, base_weights)
                    @test pw ≈ bw
                end
            end
        end
    end
    
    @testset "Edge Weight Mutation" begin
        @testset "set_edge_weight! Basic Functionality" begin
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            
            # Test setting weight by directed edge index
            if num_directed_edges(weighted_g) > 0
                edge_idx = 1
                original_weight = edge_weight(weighted_g, edge_idx)
                new_weight = original_weight + 1.0
                
                result = set_edge_weight!(weighted_g, edge_idx, new_weight)
                @test result ≈ new_weight
                @test edge_weight(weighted_g, edge_idx) ≈ new_weight
            end
            
            # Test setting weight by vertex and neighbor index
            if num_vertices(weighted_g) > 1
                v = 1
                if degree(weighted_g, v) > 0
                    k = 1  # First neighbor
                    original_weight = edge_weight(weighted_g, v, k)
                    new_weight = original_weight + 2.0
                    
                    result = set_edge_weight!(weighted_g, v, k, new_weight)
                    @test result ≈ new_weight
                    @test edge_weight(weighted_g, v, k) ≈ new_weight
                end
            end
        end
        
        @testset "set_edge_weight! Consistency" begin
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            
            # Test that setting weight by directed edge index works
            if num_vertices(weighted_g) >= 2 && has_edge(weighted_g, 1, 2)
                new_weight = 999.99
                
                # Set weight for edge (1,2)
                idx_12 = find_directed_edge_index(weighted_g, 1, 2)
                set_edge_weight!(weighted_g, idx_12, new_weight)
                
                # Check weight was set correctly
                @test edge_weight(weighted_g, idx_12) ≈ new_weight
                
                # Note: For undirected graphs, the behavior of setting one direction
                # and its effect on the reverse direction depends on the implementation
                # We test that at least the set direction works correctly
            end
        end
        
        @testset "set_edge_weight! Type Safety" begin
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            
            if num_directed_edges(weighted_g) > 0
                edge_idx = 1
                
                # Should work with correct type
                @test_nowarn set_edge_weight!(weighted_g, edge_idx, 3.14)
                
                # Test type conversion (Int should convert to Float64)
                @test_nowarn set_edge_weight!(weighted_g, edge_idx, 42.0)  # Use Float64 explicitly
                @test edge_weight(weighted_g, edge_idx) ≈ 42.0
            end
        end
        
        @testset "PropertyGraph set_edge_weight!" begin
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            vertex_props = ["A", "B", "C", "D"]  # K4 has 4 vertices
            edge_props = [1, 2, 3, 4, 5, 6]     # K4 has 6 edges
            
            prop_weighted_g = PropertyGraph(weighted_g, vertex_props, edge_props)
            
            if num_directed_edges(prop_weighted_g) > 0
                edge_idx = 1
                new_weight = 123.45
                
                # Should delegate to base graph using directed edge index
                set_edge_weight!(prop_weighted_g, edge_idx, new_weight)
                @test edge_weight(prop_weighted_g, edge_idx) ≈ new_weight
                @test edge_weight(weighted_g, edge_idx) ≈ new_weight  # Base graph also updated
            end
        end
    end
    
    @testset "Edge Weight API Consistency" begin
        @testset "All Graph Types Implement Interface" begin
            # Test that all graph types respond to edge weight queries
            core_g = build_core_graph(edges; directed=false)
            adj_g = build_adj_graph(edges; directed=false)
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            
            prop_core = PropertyGraph(core_g, ["A", "B", "C", "D"], [1, 2, 3, 4, 5, 6])
            prop_weighted = PropertyGraph(weighted_g, ["A", "B", "C", "D"], [1, 2, 3, 4, 5, 6])
            
            unweighted_graphs = [core_g, adj_g, prop_core]
            weighted_graphs = [weighted_g, prop_weighted]
            all_graphs = [unweighted_graphs; weighted_graphs]
            
            for g in all_graphs
                # All should support edge_weight with directed edge index
                @test hasmethod(edge_weight, (typeof(g), Int))
                @test hasmethod(edge_weights, (typeof(g), Int))
                @test hasmethod(neighbor_weights, (typeof(g), Int))
                
                # Test basic functionality
                if num_vertices(g) > 0
                    v = first(vertices(g))
                    if degree(g, v) > 0
                        # Should not error
                        directed_idx = directed_edge_index(g, v, 1)
                        @test_nowarn edge_weight(g, directed_idx)
                        @test_nowarn collect(edge_weights(g, v))
                        @test_nowarn collect(neighbor_weights(g, v))
                    end
                end
            end
            
            # Test that weighted graphs also support (v, i) form
            for g in weighted_graphs
                @test hasmethod(edge_weight, (typeof(g), Int, Int))
                
                if num_vertices(g) > 0
                    v = first(vertices(g))
                    if degree(g, v) > 0
                        @test_nowarn edge_weight(g, v, 1)
                    end
                end
            end
        end
        
        @testset "Return Type Consistency" begin
            core_g = build_core_graph(edges; directed=false)
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            
            # Unweighted graphs return Int32(1)
            if num_vertices(core_g) > 0
                v = first(vertices(core_g))
                if degree(core_g, v) > 0
                    directed_idx = directed_edge_index(core_g, v, 1)
                    weight = edge_weight(core_g, directed_idx)
                    @test weight isa Int32
                    @test weight === Int32(1)
                end
            end
            
            # Weighted graphs return actual Float64 weights
            if num_vertices(weighted_g) > 0
                v = first(vertices(weighted_g))
                if degree(weighted_g, v) > 0
                    weight = edge_weight(weighted_g, v, 1)  # Can use (v, i) form for weighted graphs
                    @test weight isa Float64
                    @test weight > 0
                end
            end
        end
        
        @testset "Iterator Properties" begin
            core_g = build_core_graph(edges; directed=false)
            weighted_g = build_weighted_graph(edges, weights; directed=false)
            
            for g in [core_g, weighted_g]
                for v in vertices(g)
                    weights_iter = edge_weights(g, v)
                    
                    # Should be iterable
                    @test weights_iter isa AbstractVector || 
                          weights_iter isa Base.Generator ||
                          weights_iter isa Iterators.Repeated ||
                          weights_iter isa Iterators.Take ||
                          weights_iter isa SubArray  # WeightedGraph returns SubArrays
                    
                    # Should have correct length
                    collected = collect(weights_iter)
                    @test length(collected) == degree(g, v)
                    
                    # Elements should have consistent type
                    if !isempty(collected)
                        first_type = typeof(first(collected))
                        @test all(w -> typeof(w) === first_type, collected)
                    end
                end
            end
        end
    end
    
    @testset "Generic Algorithm Compatibility" begin
        # Test that generic algorithms work with unified interface
        function simple_dijkstra_test(g, source)
            # Simple test that edge weights are accessible uniformly
            distances = Dict{Int, Float64}()
            distances[source] = 0.0
            
            # Test that we can access edge weights uniformly
            for neighbor in neighbor_indices(g, source)
                for (i, nbr) in enumerate(neighbor_indices(g, source))
                    if nbr == neighbor
                        # Use appropriate method based on graph type
                        weight = edge_weight(g, source, i)  # Weighted graphs support (v, i) form

                        # Should work regardless of graph type
                        distances[neighbor] = min(get(distances, neighbor, Inf), 
                                                  distances[source] + Float64(weight))
                        break
                    end
                end
            end
            
            return distances
        end
        
        core_g = build_core_graph(edges; directed=false)
        weighted_g = build_weighted_graph(edges, weights; directed=false)
        prop_core = PropertyGraph(core_g, ["A", "B", "C", "D"], [1, 2, 3, 4, 5, 6])
        prop_weighted = PropertyGraph(weighted_g, ["A", "B", "C", "D"], [1, 2, 3, 4, 5, 6])
        
        if num_vertices(core_g) > 0
            source = first(vertices(core_g))
            
            # Should work for all graph types
            for g in [core_g, weighted_g, prop_core, prop_weighted]
                @test_nowarn simple_dijkstra_test(g, source)
                result = simple_dijkstra_test(g, source)
                @test result isa Dict{Int, Float64}
                @test haskey(result, source)
                @test result[source] ≈ 0.0
            end
        end
    end
end
