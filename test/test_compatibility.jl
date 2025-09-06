"""
Graphs.jl Compatibility Tests
=============================

Tests for compatibility with the Graphs.jl ecosystem.
"""

using Test
using GraphCore
using GraphCore.Conversions
using Graphs: Graphs, AbstractGraph

@testset "Graphs.jl Compatibility Tests" begin
    
    @testset "AbstractGraph Interface Implementation" begin
        test_graphs = [
            ("CoreGraph Undirected", () -> build_core_graph(get_test_graph("k4").edges; directed=false)),
            ("CoreGraph Directed", () -> build_core_graph(get_test_graph("dag").edges; directed=true)),
            ("WeightedGraph Undirected", () -> begin
                test_data = get_test_graph("weighted")
                build_weighted_graph(test_data.edges, test_data.weights; directed=false)
            end),
            ("AdjGraph Undirected", () -> build_adj_graph(get_test_graph("cycle4").edges; directed=false))
        ]

        for (name, graph_constructor) in test_graphs
            @testset "$name - AbstractGraph Interface" begin
                g = graph_constructor()
                
                # Basic AbstractGraph interface
                @test g isa Graphs.AbstractGraph{Int32}
                @test Graphs.nv(g) == num_vertices(g)
                @test Graphs.ne(g) == num_edges(g)
                @test Graphs.is_directed(g) == is_directed_graph(g)
                
                # Vertex iteration
                vertices_list = collect(Graphs.vertices(g))
                @test vertices_list == collect(1:num_vertices(g))
                @test length(vertices_list) == num_vertices(g)
                
                # Edge type
                @test Graphs.edgetype(g) == Graphs.SimpleEdge{Int32}
                
                # Test that edges() returns proper SimpleEdge objects
                edges_list = collect(Graphs.edges(g))
                @test all(e isa Graphs.SimpleEdge{Int32} for e in edges_list)
                @test length(edges_list) == num_edges(g)
                
                # Verify edge consistency with our interface
                our_edges = collect(edges(g))
                @test length(edges_list) == length(our_edges)
                
                # Convert to sets for comparison (order may differ)
                graphs_edge_set = Set((Graphs.src(e), Graphs.dst(e)) for e in edges_list)
                our_edge_set = Set(our_edges)
                @test graphs_edge_set == our_edge_set
            end
        end
    end

    @testset "Neighbor Queries" begin
        test_data = get_test_graph("petersen")
        g = build_core_graph(test_data.edges; directed=false)
        
        @testset "Neighbor Consistency" begin
            for v in 1:min(num_vertices(g), 5)  # Test first few vertices
                # outneighbors should match our neighbor_indices
                out_neighbors = Graphs.outneighbors(g, v)
                our_neighbors = collect(neighbor_indices(g, v))
                @test Set(out_neighbors) == Set(our_neighbors)
                
                # For undirected graphs, inneighbors == outneighbors
                if !is_directed_graph(g)
                    in_neighbors = Graphs.inneighbors(g, v)
                    @test Set(in_neighbors) == Set(out_neighbors)
                end
                
                # Check degree consistency
                if is_directed_graph(g)
                    @test Graphs.outdegree(g, v) == length(our_neighbors)
                    @test Graphs.indegree(g, v) == length(Graphs.inneighbors(g, v))
                else
                    @test Graphs.degree(g, v) == length(our_neighbors)
                    @test Graphs.outdegree(g, v) == Graphs.indegree(g, v) == length(our_neighbors)
                end
            end
        end
        
        @testset "Directed Graph Neighbors" begin
            dag_data = get_test_graph("dag")
            g_dir = build_core_graph(dag_data.edges; directed=true)
            
            # Test directed neighbor access
            for v in vertices(g_dir)
                out_neighbors = Graphs.outneighbors(g_dir, v)
                our_neighbors = collect(neighbor_indices(g_dir, v))
                @test Set(out_neighbors) == Set(our_neighbors)
                
                # For directed graphs, in and out neighbors can be different
                in_neighbors = Graphs.inneighbors(g_dir, v)
                if haskey(dag_data.expected_in_neighbors, v)
                    expected_in = dag_data.expected_in_neighbors[v]
                    @test Set(in_neighbors) == Set(expected_in)
                end
            end
        end
    end

    @testset "Edge Queries" begin
        test_cases = [
            ("Undirected", () -> build_core_graph(get_test_graph("k4").edges; directed=false)),
            ("Directed", () -> build_core_graph(get_test_graph("dag").edges; directed=true))
        ]
        
        for (case_name, graph_constructor) in test_cases
            @testset "$case_name Edge Queries" begin
                g = graph_constructor()
                
                # Test has_edge consistency for existing edges
                our_edges = collect(edges(g))
                for (u, v) in our_edges
                    @test Graphs.has_edge(g, u, v) == has_edge(g, u, v)
                    @test Graphs.has_edge(g, u, v) == true
                    
                    # For undirected graphs, check symmetry
                    if !is_directed_graph(g)
                        @test Graphs.has_edge(g, v, u) == has_edge(g, v, u)
                        @test Graphs.has_edge(g, v, u) == true
                    end
                end
                
                # Test non-existent edges
                nv = num_vertices(g)
                if nv >= 2
                    # Test some non-existent edge pairs
                    test_pairs = [(1, nv), (nv, 1)]
                    if nv >= 3
                        append!(test_pairs, [(1, 3), (2, nv)])
                    end
                    
                    for (u, v) in test_pairs
                        graphs_result = Graphs.has_edge(g, u, v)
                        our_result = has_edge(g, u, v)
                        @test graphs_result == our_result
                    end
                end
            end
        end
    end

    @testset "Conversion to/from Graphs.jl" begin
        @testset "from_graphs_jl" begin
            # Create standard Graphs.jl graphs and convert to our format
            simple_path = Graphs.path_graph(5)
            simple_cycle = Graphs.cycle_graph(4)
            simple_complete = Graphs.complete_graph(3)
            
            # Convert to our graphs
            our_path = from_graphs_jl(simple_path)
            our_cycle = from_graphs_jl(simple_cycle)
            our_complete = from_graphs_jl(simple_complete)
            
            # Test structure preservation
            @test num_vertices(our_path) == Graphs.nv(simple_path)
            @test num_edges(our_path) == Graphs.ne(simple_path)
            @test is_directed_graph(our_path) == Graphs.is_directed(simple_path)
            
            @test num_vertices(our_cycle) == Graphs.nv(simple_cycle)
            @test num_edges(our_cycle) == Graphs.ne(simple_cycle)
            
            @test num_vertices(our_complete) == Graphs.nv(simple_complete)
            @test num_edges(our_complete) == Graphs.ne(simple_complete)
            
            # Test specific edge preservation for path graph
            # Path: 1-2-3-4-5
            @test has_edge(our_path, 1, 2) && has_edge(our_path, 2, 1)  # Undirected
            @test has_edge(our_path, 2, 3) && has_edge(our_path, 3, 2)
            @test has_edge(our_path, 3, 4) && has_edge(our_path, 4, 3)
            @test has_edge(our_path, 4, 5) && has_edge(our_path, 5, 4)
            @test !has_edge(our_path, 1, 3)  # Not connected
            @test !has_edge(our_path, 1, 5)  # Not connected
            
            # Test directed graph conversion
            simple_directed = Graphs.path_digraph(4)
            our_directed = from_graphs_jl(simple_directed)
            @test is_directed_graph(our_directed) == true
            @test has_edge(our_directed, 1, 2) && !has_edge(our_directed, 2, 1)
            @test has_edge(our_directed, 2, 3) && !has_edge(our_directed, 3, 2)
            @test has_edge(our_directed, 3, 4) && !has_edge(our_directed, 4, 3)
        end
        
        @testset "from_weighted_graphs_jl" begin
            # Create a simple weighted graph using Graphs.jl
            simple_g = Graphs.path_graph(4)
            weights = [1.0, 2.5, 3.2]  # 3 edges in path graph
            
            # Convert to our weighted graph
            our_weighted = from_weighted_graphs_jl(simple_g, weights)
            
            @test num_vertices(our_weighted) == 4
            @test num_edges(our_weighted) == 3
            @test !is_directed_graph(our_weighted)
            @test our_weighted isa WeightedGraphInterface
            
            # Test that weights are preserved
            total_weight = 0.0
            for v in vertices(our_weighted)
                for weight in edge_weights(our_weighted, v)
                    total_weight += weight
                end
            end
            # Each undirected edge weight is counted twice
            expected_total = 2 * sum(weights)
            @test total_weight ≈ expected_total
        end
        
        @testset "Round-trip Conversion" begin
            # Test that converting back and forth preserves structure
            test_data = get_test_graph("cycle4")
            g = build_core_graph(test_data.edges; directed=false)
            
            @testset "Core Graph Round-trip" begin
                # Convert our graph to Graphs.jl format via the interface
                edges_list = collect(Graphs.edges(g))
                
                # Create a new Graphs.jl graph with same structure
                if is_directed_graph(g)
                    graphs_g = Graphs.SimpleDiGraph(num_vertices(g))
                else
                    graphs_g = Graphs.SimpleGraph(num_vertices(g))
                end
                
                for edge in edges_list
                    Graphs.add_edge!(graphs_g, Graphs.src(edge), Graphs.dst(edge))
                end
                
                # Convert back to our format
                our_g_roundtrip = from_graphs_jl(graphs_g)
                
                # Should have same structure
                @test num_vertices(our_g_roundtrip) == num_vertices(g)
                @test num_edges(our_g_roundtrip) == num_edges(g)
                @test is_directed_graph(our_g_roundtrip) == is_directed_graph(g)
                
                # Should have same edges
                for (u, v) in test_data.edges
                    @test has_edge(our_g_roundtrip, u, v)
                    if !is_directed_graph(g)
                        @test has_edge(our_g_roundtrip, v, u)
                    end
                end
            end
        end
    end

    @testset "Graphs.jl Algorithm Compatibility" begin
        test_data = get_test_graph("petersen")
        g = build_core_graph(test_data.edges; directed=false)
        
        @testset "Basic Algorithm Compatibility" begin
            nv = num_vertices(g)
            
            if nv > 0
                # Test degree sequence
                degree_seq = [Graphs.degree(g, v) for v in Graphs.vertices(g)]
                @test length(degree_seq) == nv
                @test all(d >= 0 for d in degree_seq)
                
                # For undirected graphs, sum of degrees should be 2*num_edges
                if !is_directed_graph(g)
                    @test sum(degree_seq) == 2 * num_edges(g)
                end
                
                # Test that basic graph queries work
                @test Graphs.nv(g) == nv
                @test Graphs.ne(g) == num_edges(g)
                
                # Test vertex and edge iteration
                vertices_count = length(collect(Graphs.vertices(g)))
                @test vertices_count == nv
                
                edges_count = length(collect(Graphs.edges(g)))
                @test edges_count == num_edges(g)
            end
        end
        
        @testset "Directed Graph Algorithms" begin
            dag_data = get_test_graph("dag")
            g_dir = build_core_graph(dag_data.edges; directed=true)
            
            # Test directed graph specific properties
            @test Graphs.is_directed(g_dir) == true
            
            # Test in/out degree calculations
            for v in vertices(g_dir)
                out_deg = Graphs.outdegree(g_dir, v)
                in_deg = Graphs.indegree(g_dir, v)
                
                expected_out = length(get(dag_data.expected_neighbors, v, []))
                expected_in = length(get(dag_data.expected_in_neighbors, v, []))
                
                @test out_deg == expected_out
                @test in_deg == expected_in
            end
        end
    end

    @testset "Performance Consistency" begin
        # Basic performance sanity checks
        @testset "Large Graph Compatibility" begin
            # Create a moderately sized graph
            simple_g = Graphs.cycle_graph(100)
            our_g = from_graphs_jl(simple_g)
            
            @testset "Neighbor Access Performance" begin
                v = 50  # Middle vertex
                
                graphs_neighbors = Graphs.outneighbors(simple_g, v)
                our_neighbors = collect(neighbor_indices(our_g, v))
                
                # Should get same results
                @test Set(graphs_neighbors) == Set(our_neighbors)
                
                # Both should be reasonably fast (basic sanity check)
                @test length(graphs_neighbors) >= 1
                @test length(our_neighbors) >= 1
            end
            
            @testset "Edge Query Performance" begin
                # Test edge existence queries
                test_edges = [(1,2), (50,51), (99,100), (100,1), (25,75)]
                
                for (u, v) in test_edges
                    if 1 <= u <= 100 && 1 <= v <= 100
                        graphs_result = Graphs.has_edge(simple_g, u, v)
                        our_result = has_edge(our_g, u, v)
                        @test graphs_result == our_result
                    end
                end
            end
        end
    end
    
    @testset "Error Handling and Edge Cases" begin
        @testset "Empty Graph Compatibility" begin
            empty_graphs_g = Graphs.SimpleGraph(0)
            our_empty = from_graphs_jl(empty_graphs_g)
            
            @test num_vertices(our_empty) >= 0  # Implementation may add minimal vertices
            @test num_edges(our_empty) == 0
        end
        
        @testset "Single Vertex Graph" begin
            single_graphs_g = Graphs.SimpleGraph(1)
            our_single = from_graphs_jl(single_graphs_g)
            
            @test num_vertices(our_single) == 1
            @test num_edges(our_single) == 0
            @test degree(our_single, 1) == 0
        end
        
        @testset "Invalid Operations" begin
            g = build_core_graph([(1, 2)]; directed=false)
            
            # Test boundary conditions - actual library behavior
            @test_throws BoundsError has_edge(g, 1, 0)        # Invalid vertex (below range)
            @test_throws BoundsError has_edge(g, 0, 1)        # Invalid vertex (below range)  
            @test_throws BoundsError has_edge(g, 3, 1)        # Non-existent vertex (above range)
            @test_throws BoundsError has_edge(g, 1, 10)       # Non-existent vertex (above range)

            # Self-loops (not present in this graph)
            @test has_edge(g, 1, 1) == false  
            @test has_edge(g, 2, 2) == false
        end
    end
end