# examples/plotting_example.jl
#
# This example demonstrates how to visualize GraphCore graphs using Plots.jl and GraphRecipes.jl
#
# To run this example, first install the plotting dependencies:
#   julia --project -e "using Pkg; Pkg.add([\"Plots\", \"GraphRecipes\"])"
#
# Then run:
#   julia --project examples/plotting_example.jl

using GraphCore
using Plots, GraphRecipes
using GraphCore.Conversions: to_graphs_jl

function plot_social_network()
    println("Creating a social network graph...")
    
    # Create a simple social network
    people = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    connections = [
        (1, 2), (1, 3), (1, 4),  # Alice knows Bob, Charlie, Diana
        (2, 3), (2, 5),          # Bob knows Charlie, Eve  
        (3, 4), (3, 6),          # Charlie knows Diana, Frank
        (4, 5),                  # Diana knows Eve
        (5, 6)                   # Eve knows Frank
    ]
    
    # Build the graph
    social_graph = build_core_graph(connections; directed=false)
    
    # Convert to SimpleGraph for plotting (GraphRecipes expects Graphs.jl types)
    # simple_graph = to_graphs_jl(social_graph)

    # Create the plot directly from GraphCore graph
    plt = graphplot(social_graph, 
                   names=people,
                   nodesize=0.15,
                   nodecolor=:lightblue,
                   edgewidth=(s,d,w)-> 2,
                   edgecolor=:gray,
                   curves=false,
                   title="Social Network Example")
    
    # Save the plot
    savefig(plt, "social_network.png")
    println("✅ Plot saved as 'social_network.png'")
    
    return plt
end

function main()
    println("GraphCore.jl Plotting Examples")
    println("=" ^ 35)
    
    try
        # Social network visualization
        plot_social_network()
        display(current())
        println()
        
        println("🎨 All plots created successfully!")
        println("   Check the current directory for PNG files.")
        
    catch e
        if e isa ArgumentError && occursin("Package Plots not found", string(e))
            println("❌ Plotting packages not installed!")
            println("   Install with: julia --project -e \"using Pkg; Pkg.add([\\\"Plots\\\", \\\"GraphRecipes\\\"])\"")
        else
            println("❌ Error: $e")
        end
    end
end

# Run the examples if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
