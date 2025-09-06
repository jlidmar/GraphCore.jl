"""
Type Conversion Tests
====================

Test conversions between different graph types and external libraries.
"""

using Test
using GraphCore
using GraphCore.Conversions
using GraphCore.Builders: build_graph
using Graphs: Graph, SimpleGraph

function test_conversions()
    @testset "Conversion Tests" begin
        
        @testset "Internal Conversions" begin
            # Test to_core_graph
            edges = [(1,2), (2,3), (1,3)]
            weights = [1.0, 2.0, 1.5]
            
            # WeightedGraph to CoreGraph
            wg = build_weighted_graph(edges, weights; directed=false)
            cg = to_core_graph(wg)
            
            @test num_vertices(cg) == num_vertices(wg)
            @test num_edges(cg) == num_edges(wg)
            @test is_directed_graph(cg) == is_directed_graph(wg)
            
            # Verify same edges exist
            for (u, v) in edges
                @test has_edge(cg, u, v) == has_edge(wg, u, v)
                @test has_edge(cg, v, u) == has_edge(wg, v, u)  # Undirected
            end

            # Test to_weighted_graph
            cg_back = to_core_graph(wg)
            wg_back = to_weighted_graph(wg)  # Should be identity for WeightedGraph
            
            @test num_vertices(wg_back) == num_vertices(wg)
            @test num_edges(wg_back) == num_edges(wg)
            @test is_directed_graph(wg_back) == is_directed_graph(wg)
            
            # Verify weights preserved
            for (u, v) in edges
                if has_edge(wg, u, v)
                    idx_orig = find_directed_edge_index(wg, u, v)
                    idx_back = find_directed_edge_index(wg_back, u, v)
                    @test edge_weight(wg, idx_orig) == edge_weight(wg_back, idx_back)
                end
            end
        end
        
        @testset "Graphs.jl Conversions" begin
            # Test from_graphs_jl
            simple_graph = Graphs.cycle_graph(4)
            our_graph = from_graphs_jl(simple_graph)
            
            @test num_vertices(our_graph) == 4
            @test num_edges(our_graph) == 4
            @test !is_directed_graph(our_graph)
            
            # Test specific edges (cycle: 1-2-3-4-1)
            @test has_edge(our_graph, 1, 2)
            @test has_edge(our_graph, 2, 3)
            @test has_edge(our_graph, 3, 4)
            @test has_edge(our_graph, 4, 1)
            
            # Test from_weighted_graphs_jl
            simple_path = Graphs.path_graph(3)
            weights = [1.0, 2.0]  # 2 edges in path
            
            our_weighted = from_weighted_graphs_jl(simple_path, weights)
            
            @test num_vertices(our_weighted) == 3
            @test num_edges(our_weighted) == 2
            @test our_weighted isa WeightedGraphInterface
            @test !is_directed_graph(our_weighted)
        end
    end
end

@testset "Type Conversion Tests" begin
    # Setup test data  
    petersen_data = get_test_graph("petersen")
    edges = petersen_data.edges
    weights = Float64[i for i in 1:length(edges)]
    vertex_props = get_vertex_properties(petersen_data.expected_nv)
    edge_props = get_edge_properties(length(edges))
    
    @testset "GraphInterface Conversions" begin
        # Create source graphs
        core_g = build_core_graph(edges; directed=false)
        weighted_g = build_weighted_graph(edges, weights; directed=false)

        @testset "to_core_graph" begin
            # Convert weighted to core
            converted = to_core_graph(weighted_g)
            @test converted isa CoreGraph
            @test num_vertices(converted) == 10
            @test num_edges(converted) == 15
            @test !is_directed_graph(converted)
            
            # Verify topology is preserved
            for (u, v) in edges
                @test has_edge(converted, u, v)
                @test has_edge(converted, v, u)
            end
            
            # Test with property graph
            prop_g = PropertyGraph(core_g, vertex_props, edge_props)
            converted_prop = to_core_graph(prop_g)
            @test converted_prop isa CoreGraph
            @test num_vertices(converted_prop) == 10
            @test num_edges(converted_prop) == 15
        end
        
        @testset "to_weighted_graph" begin
            # Convert weighted to weighted (should preserve weights)
            converted = to_weighted_graph(weighted_g)
            @test converted isa WeightedGraph
            @test num_vertices(converted) == 10
            @test num_edges(converted) == 15
            
            # Verify weights are preserved
            for (u, v) in edges
                orig_idx = find_directed_edge_index(weighted_g, u, v)
                conv_idx = find_directed_edge_index(converted, u, v)
                @test edge_weight(weighted_g, orig_idx) ≈ edge_weight(converted, conv_idx)
            end
        end
    end
    
    @testset "Builder Conversions" begin
        @testset "CoreGraph to different targets" begin
            core_g = build_core_graph(edges; directed=false)
            
            # To AdjGraph
            adj_g = to_adj_graph(core_g)  # This function needs to be implemented
            @test adj_g isa AdjGraph
            @test num_vertices(adj_g) == 10
            @test num_edges(adj_g) == 15
            
            for (u, v) in edges
                @test has_edge(adj_g, u, v)
            end
        end
        
        @testset "Builder finalization targets" begin
            builder = WeightedGraphBuilder(Float64; directed=false)
            
            for (i, (u, v)) in enumerate(edges)
                add_edge!(builder, u, v; weight=Float64(i))
            end
            
            # build to different types
            core_result = build_graph(builder, CoreGraph)
            @test core_result isa CoreGraph
            @test num_vertices(core_result) == 10
            @test num_edges(core_result) == 15

            weighted_result = build_graph(builder, WeightedGraph{Float64})
            @test weighted_result isa WeightedGraph{Float64}
            @test num_vertices(weighted_result) == 10
            @test num_edges(weighted_result) == 15

            adj_result = build_graph(builder, AdjGraph)
            @test adj_result isa AdjGraph
            @test num_vertices(adj_result) == 10
            @test num_edges(adj_result) == 15
        end
    end
    
    @testset "Round-trip Conversions" begin
        @testset "CoreGraph round-trip" begin
            original = build_core_graph(edges; directed=false)
            
            # CoreGraph → Builder → CoreGraph
            builder = GraphBuilder(directed=false)
            for (u, v) in edges
                add_edge!(builder, u, v)
            end
            reconstructed = build_graph(builder, CoreGraph)

            # Should be topologically identical
            @test num_vertices(original) == num_vertices(reconstructed)
            @test num_edges(original) == num_edges(reconstructed)
            @test is_directed_graph(original) == is_directed_graph(reconstructed)
            
            for (u, v) in edges
                @test has_edge(original, u, v) == has_edge(reconstructed, u, v)
            end
        end
        
        @testset "WeightedGraph round-trip" begin
            original = build_weighted_graph(edges, weights; directed=false)
            
            # WeightedGraph → Builder → WeightedGraph
            builder = WeightedGraphBuilder(Float64; directed=false)
            for (i, (u, v)) in enumerate(edges)
                add_edge!(builder, u, v; weight=weights[i])
            end
            reconstructed = build_graph(builder, WeightedGraph{Float64})

            # Topology should be identical
            @test num_vertices(original) == num_vertices(reconstructed)
            @test num_edges(original) == num_edges(reconstructed)
            
            # Weights should be preserved
            for (u, v) in edges
                orig_idx = find_directed_edge_index(original, u, v)
                recon_idx = find_directed_edge_index(reconstructed, u, v)
                @test edge_weight(original, orig_idx) ≈ edge_weight(reconstructed, recon_idx)
            end
        end
    end
    
    @testset "Conversion Consistency" begin
        # All conversions of the same graph should produce topologically identical results
        core_g = build_core_graph(edges; directed=false)
        weighted_g = build_weighted_graph(edges, weights; directed=false)
        adj_g = build_adj_graph(edges; directed=false)
        
        graphs = [core_g, weighted_g, adj_g]
        
        for g1 in graphs, g2 in graphs
            @test num_vertices(g1) == num_vertices(g2)
            @test num_edges(g1) == num_edges(g2)
            @test is_directed_graph(g1) == is_directed_graph(g2)
            
            # All should have the same edges
            for (u, v) in edges
                @test has_edge(g1, u, v) == has_edge(g2, u, v)
            end
            
            # Neighbor sets should be identical
            for v in vertices(g1)
                neighbors1 = sort(collect(neighbor_indices(g1, v)))
                neighbors2 = sort(collect(neighbor_indices(g2, v)))
                @test neighbors1 == neighbors2
            end
        end
    end
end
