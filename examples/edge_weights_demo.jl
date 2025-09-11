# Edge Weights Interface Demonstration
# Shows the unified edge weight API working across all graph types

using GraphCore

println("🔗 Edge Weights Interface Demo")
println("=" ^ 40)

# Create test edges for a small graph
edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
weights = [1.0, 2.0, 1.5, 3.0, 2.5]

println("\n📊 Test Graph:")
println("   Edges: ", edges)
println("   Weights: ", weights)

# Create different graph types
println("\n🏗️  Creating Graph Types...")
core_g = build_core_graph(edges; directed=false)
adj_g = build_adj_graph(edges; directed=false)
weighted_g = build_weighted_graph(edges, weights; directed=false)

# Create PropertyGraphs
vertex_props = ["A", "B", "C", "D"]
edge_props = ["a", "b", "c", "d", "e"]
prop_core = PropertyGraph(core_g, vertex_props, edge_props)
prop_weighted = PropertyGraph(weighted_g, vertex_props, edge_props)

println("   ✅ CoreGraph (unweighted)")
println("   ✅ AdjGraph (unweighted)")
println("   ✅ WeightedGraph")
println("   ✅ PropertyGraph + CoreGraph")
println("   ✅ PropertyGraph + WeightedGraph")

# Demonstrate unified edge weight interface
println("\n⚖️  Unified Edge Weight Interface:")
println()

function demo_edge_weights(g, name)
    println("📈 $name:")
    
    # Show edge weights for first vertex
    v = 1
    neighbors = collect(neighbor_indices(g, v))
    println("   Vertex $v neighbors: $neighbors")
    
    # Method 1: Individual edge weights
    println("   Edge weights (individual access):")
    for (i, neighbor) in enumerate(neighbors)
        weight = edge_weight(g, v, i)  # Efficient (v, i) form for weighted graphs (including PropertyGraph!)
        println("     edge($v → $neighbor): $weight")
    end
    
    # Method 2: Batch edge weights
    weights_iter = edge_weights(g, v)
    weights_collected = collect(weights_iter)
    println("   Edge weights (batch): $weights_collected")
    
    # Method 3: Combined neighbor-weight iteration
    println("   Neighbor-weight pairs:")
    for (neighbor, weight) in neighbor_weights(g, v)
        println("     $neighbor: $weight")
    end
    
    println()
end

# Demo all graph types
demo_edge_weights(core_g, "CoreGraph (unweighted)")
demo_edge_weights(adj_g, "AdjGraph (unweighted)")
demo_edge_weights(weighted_g, "WeightedGraph")
demo_edge_weights(prop_core, "PropertyGraph + CoreGraph")
demo_edge_weights(prop_weighted, "PropertyGraph + WeightedGraph")

# Demonstrate weight mutation
println("🔧 Weight Mutation Demo:")
println()

if num_directed_edges(weighted_g) > 0
    # Show original weight
    edge_idx = 1
    original_weight = edge_weight(weighted_g, edge_idx)
    println("   Original weight at edge $edge_idx: $original_weight")
    
    # Modify weight
    new_weight = 99.99
    set_edge_weight!(weighted_g, edge_idx, new_weight)
    updated_weight = edge_weight(weighted_g, edge_idx)
    println("   Updated weight at edge $edge_idx: $updated_weight")
    
    # Show that PropertyGraph sees the change too
    prop_weight = edge_weight(prop_weighted, edge_idx)
    println("   PropertyGraph sees same weight: $prop_weight")
    
    # Restore original weight
    set_edge_weight!(weighted_g, edge_idx, original_weight)
    println("   ✅ Weight restored to: $(edge_weight(weighted_g, edge_idx))")
end

println("\n🎯 Generic Algorithm Demo:")
println()

# Generic function that works with any graph type
function compute_total_weight(g)
    total = 0.0
    for v in vertices(g)
        for (neighbor, weight) in neighbor_weights(g, v)
            total += weight
        end
    end
    return total / 2  # Divide by 2 for undirected graphs (each edge counted twice)
end

# Works with all graph types!
graphs_to_test = [
    (core_g, "CoreGraph"),
    (adj_g, "AdjGraph"), 
    (weighted_g, "WeightedGraph"),
    (prop_core, "PropertyGraph+CoreGraph"),
    (prop_weighted, "PropertyGraph+WeightedGraph")
]

for (g, name) in graphs_to_test
    total = compute_total_weight(g)
    println("   Total weight ($name): $total")
end

println("\n✨ Key Benefits:")
println("   🚀 Unified API: Same code works for weighted and unweighted graphs")
println("   ⚡ Efficient: Specialized iterators avoid allocation for unweighted graphs")  
println("   🔒 Type Safe: Compile-time dispatch ensures optimal performance")
println("   🧩 Composable: PropertyGraph cleanly wraps any base graph type")
println("   📝 Consistent: All graph types implement the same interface")

println("\n🎉 Demo completed successfully!")
