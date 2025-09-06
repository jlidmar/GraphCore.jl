# examples/basic_usage.jl
using GraphCore

# Social network example
function social_network_example()
    # People and their relationships
    people = ["Alice", "Bob", "Charlie", "Diana"]
    relationships = [
        ("Alice", "Bob", "friend"),
        ("Bob", "Charlie", "colleague"), 
        ("Charlie", "Diana", "family"),
        ("Alice", "Diana", "friend")
    ]
    
    # Convert to indices
    name_to_id = Dict(name => i for (i, name) in enumerate(people))
    edges = [(name_to_id[src], name_to_id[dst]) for (src, dst, _) in relationships]
    edge_types = [rel_type for (_, _, rel_type) in relationships]
    
    # Build graph
    social_graph = build_property_graph(edges, people, edge_types; directed=false)

    # Query the graph
    println("$(people[1])'s friends:")
    for neighbor_id in neighbor_indices(social_graph, 1)
        println("  - $(people[neighbor_id])")
    end
    
    return social_graph
end

# examples/performance_comparison.jl
# Benchmark different graph types for specific use cases

# examples/algorithms_demo.jl  
# Show integration with graph algorithms

# examples/graphs_jl_integration.jl
# Demonstrate Graphs.jl interoperability