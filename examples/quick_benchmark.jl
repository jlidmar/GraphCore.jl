# examples/quick_benchmark.jl
# Quick performance demonstration for GraphCore.jl

using GraphCore
using BenchmarkTools
using Random
using Printf

function quick_demo()
    println("GraphCore.jl Quick Performance Demo")
    println("="^40)
    
    Random.seed!(42)
    
    # Create test graphs
    edges_1k = [(rand(1:1000), rand(1:1000)) for _ in 1:2000]
    edges_1k = unique([(min(u,v), max(u,v)) for (u,v) in edges_1k if u != v])
    
    edges_5k = [(rand(1:5000), rand(1:5000)) for _ in 1:10000]
    edges_5k = unique([(min(u,v), max(u,v)) for (u,v) in edges_5k if u != v])
    
    println("\nTest graphs: $(length(edges_1k)) edges (1K), $(length(edges_5k)) edges (5K)")
    
    # Graph construction
    println("\nConstruction Times:")
    
    b1 = @benchmark build_graph(CoreGraph, $edges_1k; directed=false)
    @printf "  CoreGraph (1K): %.2f ms\n" median(b1).time/1e6
    
    b2 = @benchmark build_graph(AdjGraph, $edges_1k; directed=false)
    @printf "  AdjGraph (1K):  %.2f ms\n" median(b2).time/1e6
    
    b3 = @benchmark build_graph(CoreGraph, $edges_5k; directed=false)
    @printf "  CoreGraph (5K): %.2f ms\n" median(b3).time/1e6
    
    b4 = @benchmark build_graph(AdjGraph, $edges_5k; directed=false)
    @printf "  AdjGraph (5K):  %.2f ms\n" median(b4).time/1e6
    
    core_1k = build_graph(CoreGraph, edges_1k; directed=false)
    adj_1k = build_graph(AdjGraph, edges_1k; directed=false)
    
    # Neighbor access
    println("\nNeighbor Access (per vertex):")
    
    test_vertices = rand(1:1000, 100)
    
    b1 = @benchmark begin
        for v in $test_vertices
            neighbors = neighbor_indices($core_1k, v)
            length(neighbors)
        end
    end
    @printf "  CoreGraph: %.1f ns\n" median(b1).time / length(test_vertices)
    
    b2 = @benchmark begin
        for v in $test_vertices
            neighbors = neighbor_indices($adj_1k, v)
            length(neighbors)
        end
    end
    @printf "  AdjGraph:  %.1f ns\n" median(b2).time / length(test_vertices)
    
    # Property graphs
    println("\nProperty Graph Construction:")
    
    vertex_props = ["vertex_$i" for i in 1:1000]
    edge_props = ["edge_$i" for i in 1:length(edges_1k)]
    
    b1 = @benchmark build_graph(PropertyGraph{CoreGraph,String,String}, $edges_1k;
                              directed=false,
                              vertex_properties=$vertex_props,
                              edge_properties=$edge_props)
    @printf "  PropertyGraph: %.2f ms\n" median(b1).time / 1e6
    
    println("\n" * "="^40)
    println("CoreGraph: Static analysis, memory-efficient")
    println("AdjGraph:  Dynamic modifications")
    println("PropertyGraph: Attributes with minimal overhead")
end

# Property graph demo
function property_graph_demo()
    println("\nProperty Graph Demo")
    println("="^20)
    
    # Create a simple social network
    edges = [(1,2), (2,3), (3,4), (1,4), (2,4), (1,3)]
    people = ["Alice", "Bob", "Charlie", "Diana"]
    relationships = ["friend", "colleague", "family", "neighbor", "teammate", "classmate"]
    
    social_g = build_graph(PropertyGraph{CoreGraph,String,String}, edges;
                          directed=false,
                          vertex_properties=people,
                          edge_properties=relationships)
    
    println("\nConnections:")
    for i in 1:length(people)
        person = vertex_property(social_g, i)
        connections = [vertex_property(social_g, n) for n in neighbor_indices(social_g, i)]
        println("  $person ↔ $(join(connections, ", "))")
    end
end

# Run demos
if abspath(PROGRAM_FILE) == @__FILE__
    quick_demo()
    property_graph_demo()
end
