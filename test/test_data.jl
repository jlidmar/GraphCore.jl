"""
Centralized Test Data
====================

This module provides standardized test graphs and data to ensure consistency
across all test files and eliminate redundancy.
"""

module TestData

export TestGraph, get_test_graph, test_graph_list
export StandardTestGraphs, get_vertex_properties, get_edge_properties

using GraphCore

"""
    TestGraph

Structure to hold test graph data including edges, expected properties, and metadata.
"""
struct TestGraph{T}
    name::String
    edges::Vector{Tuple{Int,Int}}
    expected_nv::Int
    expected_ne::Int
    directed::Bool
    weights::Vector{T}
    expected_neighbors::Dict{Int,Vector{Int}}
    expected_in_neighbors::Union{Dict{Int,Vector{Int}},Nothing}
    description::String
end

# Constructor for unweighted graphs
function TestGraph(name::String, edges::Vector{Tuple{Int,Int}}, expected_nv::Int, 
                  expected_ne::Int, directed::Bool, expected_neighbors::Dict{Int,Vector{Int}},
                  expected_in_neighbors=nothing, description::String="")
    TestGraph{Float64}(name, edges, expected_nv, expected_ne, directed, Float64[], 
                       expected_neighbors, expected_in_neighbors, description)
end

# Constructor for weighted graphs
function TestGraph(name::String, edges::Vector{Tuple{Int,Int}}, expected_nv::Int, 
                  expected_ne::Int, directed::Bool, weights::Vector{T},
                  expected_neighbors::Dict{Int,Vector{Int}},
                  expected_in_neighbors=nothing, description::String="") where T
    TestGraph{T}(name, edges, expected_nv, expected_ne, directed, weights, 
                 expected_neighbors, expected_in_neighbors, description)
end

"""
Standard Test Graphs
"""
module StandardTestGraphs

using ..TestData

"""
Petersen Graph (10 vertices, 15 edges, undirected)
Famous graph with interesting properties: non-planar, 3-regular
"""
function petersen_graph()
    edges = [
        # Outer pentagon
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 1),
        # Inner star  
        (6, 8), (8, 10), (10, 7), (7, 9), (9, 6),
        # Cross connections
        (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)
    ]
    
    expected_neighbors = Dict(
        1 => [2, 5, 6],    2 => [1, 3, 7],    3 => [2, 4, 8],
        4 => [3, 5, 9],    5 => [1, 4, 10],   6 => [8, 9, 1],
        7 => [10, 9, 2],   8 => [6, 10, 3],   9 => [7, 6, 4],
        10 => [8, 7, 5]
    )
    
    TestGraph("petersen", edges, 10, 15, false, expected_neighbors, nothing,
              "Famous non-planar 3-regular graph")
end

"""
Directed Acyclic Graph (DAG) for testing directed graphs
6 vertices, 7 edges, represents a simple task dependency graph
"""
function dag_graph()
    edges = [
        (1, 2), (1, 3),           # 1 → {2, 3}
        (2, 4), (3, 4), (3, 5),   # 2 → 4, 3 → {4, 5}
        (4, 6), (5, 6)            # 4 → 6, 5 → 6
    ]
    
    expected_neighbors = Dict(  # Outgoing neighbors for directed graph
        1 => [2, 3],   2 => [4],      3 => [4, 5],
        4 => [6],      5 => [6],      6 => Int[]
    )
    
    expected_in_neighbors = Dict(  # Incoming neighbors for directed graph
        1 => Int[],    2 => [1],      3 => [1],
        4 => [2, 3],   5 => [3],      6 => [4, 5]
    )
    
    TestGraph("dag", edges, 6, 7, true, expected_neighbors, expected_in_neighbors,
              "Task dependency DAG")
end

"""
Complete Graph K4 (4 vertices, 6 edges, undirected)
Every vertex connected to every other vertex
"""
function complete4_graph()
    edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    
    expected_neighbors = Dict(
        1 => [2, 3, 4],   2 => [1, 3, 4],
        3 => [1, 2, 4],   4 => [1, 2, 3]
    )
    
    TestGraph("k4", edges, 4, 6, false, expected_neighbors, nothing,
              "Complete graph on 4 vertices")
end

"""
Path Graph P5 (5 vertices, 4 edges)
Linear chain: 1-2-3-4-5
"""
function path5_graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    
    expected_neighbors = Dict(
        1 => [2],      2 => [1, 3],    3 => [2, 4],
        4 => [3, 5],   5 => [4]
    )
    
    TestGraph("path5", edges, 5, 4, false, expected_neighbors, nothing,
              "Linear path with 5 vertices")
end

"""
Cycle Graph C4 (4 vertices, 4 edges)
Square cycle: 1-2-3-4-1
"""
function cycle4_graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    
    expected_neighbors = Dict(
        1 => [2, 4],   2 => [1, 3],
        3 => [2, 4],   4 => [3, 1]
    )
    
    TestGraph("cycle4", edges, 4, 4, false, expected_neighbors, nothing,
              "4-vertex cycle")
end

"""
Star Graph (5 vertices, 4 edges)
Hub vertex connected to all others: 1 is center connected to 2,3,4,5
"""
function star5_graph()
    edges = [(1, 2), (1, 3), (1, 4), (1, 5)]
    
    expected_neighbors = Dict(
        1 => [2, 3, 4, 5],   2 => [1],   3 => [1],
        4 => [1],            5 => [1]
    )
    
    TestGraph("star5", edges, 5, 4, false, expected_neighbors, nothing,
              "Star graph with center vertex 1")
end

"""
Weighted test graph with interesting weight patterns
"""
function weighted_graph()
    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    weights = [1.0, 2.0, 1.5, 3.0, 2.5]
    
    expected_neighbors = Dict(
        1 => [2, 3],      2 => [1, 3, 4],
        3 => [1, 2, 4],   4 => [2, 3]
    )
    
    TestGraph("weighted", edges, 4, 5, false, weights, expected_neighbors, nothing,
              "Small weighted graph for testing weight functionality")
end

"""
Empty graph (1 vertex, 0 edges)
Minimal graph for testing edge cases
"""
function empty_graph()
    edges = Tuple{Int,Int}[]
    expected_neighbors = Dict(1 => Int[])
    
    TestGraph("empty", edges, 0, 0, false, expected_neighbors, nothing,
              "Minimal empty graph")
end

"""
Single edge graph (2 vertices, 1 edge)
"""
function single_edge_graph()
    edges = [(1, 2)]
    expected_neighbors = Dict(1 => [2], 2 => [1])
    
    TestGraph("single_edge", edges, 2, 1, false, expected_neighbors, nothing,
              "Graph with single edge")
end

end  # StandardTestGraphs

"""
    test_graph_list()

Returns a list of all available test graphs.
"""
function test_graph_list()
    return [
        StandardTestGraphs.petersen_graph(),
        StandardTestGraphs.dag_graph(),
        StandardTestGraphs.complete4_graph(),
        StandardTestGraphs.path5_graph(),
        StandardTestGraphs.cycle4_graph(),
        StandardTestGraphs.star5_graph(),
        StandardTestGraphs.weighted_graph(),
        StandardTestGraphs.empty_graph(),
        StandardTestGraphs.single_edge_graph()
    ]
end

"""
    get_test_graph(name::String)

Retrieve a specific test graph by name.
"""
function get_test_graph(name::String)
    graphs = test_graph_list()
    for graph in graphs
        if graph.name == name
            return graph
        end
    end
    throw(ArgumentError("Unknown test graph: $name"))
end

"""
Common test properties for validation
"""
const TEST_VERTEX_PROPERTIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
const TEST_EDGE_PROPERTIES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]

"""
    get_vertex_properties(n::Int)

Get test vertex properties for n vertices.
"""
get_vertex_properties(n::Int) = TEST_VERTEX_PROPERTIES[1:n]

"""
    get_edge_properties(n::Int)

Get test edge properties for n edges.
"""
get_edge_properties(n::Int) = TEST_EDGE_PROPERTIES[1:n]

end  # module TestData
