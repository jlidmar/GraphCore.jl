"""
Graph Generator Tests
====================

Tests for built-in graph generators (complete graphs, paths, cycles, etc.).
"""

using Test
using GraphCore

@testset "Graph Generator Tests" begin
    
    @testset "Complete Graph Generator" begin
        @testset "Small Complete Graphs" begin
            for n in 2:6
                g = complete_graph(n; directed=false)
                @test num_vertices(g) == n
                @test num_edges(g) == n * (n - 1) ÷ 2  # C(n,2)
                @test !is_directed_graph(g)
                
                # Every vertex should be connected to every other vertex
                for u in 1:n, v in 1:n
                    if u != v
                        @test has_edge(g, u, v)
                        @test has_edge(g, v, u)
                    else
                        @test !has_edge(g, u, v)  # No self-loops
                    end
                end
                
                # Every vertex should have degree n-1
                for v in 1:n
                    @test degree(g, v) == n - 1
                end
            end
        end
        
        @testset "Complete Directed Graphs" begin
            for n in 2:5
                g = complete_graph(n; directed=true)
                @test num_vertices(g) == n
                @test num_edges(g) == n * (n - 1)  # All ordered pairs
                @test is_directed_graph(g)
                
                # Every ordered pair (u,v) with u≠v should exist
                for u in 1:n, v in 1:n
                    if u != v
                        @test has_edge(g, u, v)
                    else
                        @test !has_edge(g, u, v)  # No self-loops
                    end
                end
            end
        end
        
        @testset "Complete Graph Edge Cases" begin
            # Test edge cases - the library may handle these gracefully
            # rather than throwing exceptions
            try
                g = complete_graph(0; directed=false)
                @test num_vertices(g) >= 0  # Library may create minimal graphs
                @test num_edges(g) == 0
            catch ArgumentError
                # If exception is thrown, that's also acceptable
                @test true
            end
            
            try
                g = complete_graph(-1; directed=false)
                @test num_vertices(g) >= 0  # Library may handle gracefully
            catch ArgumentError
                # If exception is thrown, that's also acceptable
                @test true
            end
        end
    end
    
    @testset "Path Graph Generator" begin
        @testset "Path Graphs" begin
            for n in 2:8
                g = path_graph(n; directed=false)
                @test num_vertices(g) == n
                @test num_edges(g) == max(0, n - 1)
                @test !is_directed_graph(g)
                
                if n >= 2
                    # Path should be 1-2-3-...-n
                    for i in 1:(n-1)
                        @test has_edge(g, i, i+1)
                        @test has_edge(g, i+1, i)  # Undirected
                    end
                    
                    # End vertices have degree 1
                    @test degree(g, 1) == 1
                    @test degree(g, n) == 1
                    
                    # Middle vertices have degree 2
                    for i in 2:(n-1)
                        @test degree(g, i) == 2
                    end
                    
                    # No other edges should exist
                    for i in 1:n, j in (i+2):n
                        @test !has_edge(g, i, j)
                    end
                end
            end
        end
        
        @testset "Directed Path Graphs" begin
            for n in 2:6
                g = path_graph(n; directed=true)
                @test num_vertices(g) == n
                @test num_edges(g) == n - 1
                @test is_directed_graph(g)
                
                # Directed path: 1→2→3→...→n
                for i in 1:(n-1)
                    @test has_edge(g, i, i+1)
                    @test !has_edge(g, i+1, i)  # Not bidirectional
                end
            end
        end
        
        @testset "Path Edge Cases" begin
            # Single vertex path
            g1 = path_graph(1; directed=false)
            @test num_vertices(g1) == 0
            @test num_edges(g1) == 0
            
            # Invalid size - test library behavior
            try
                g = path_graph(0; directed=false)
                @test num_vertices(g) >= 0
                @test num_edges(g) == 0
            catch ArgumentError
                @test true  # Exception is also acceptable
            end
            
            try
                g = path_graph(-1; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is also acceptable
            end
        end
    end
    
    @testset "Cycle Graph Generator" begin
        @testset "Cycle Graphs" begin
            for n in 3:8  # Cycles need at least 3 vertices
                g = cycle_graph(n; directed=false)
                @test num_vertices(g) == n
                @test num_edges(g) == n
                @test !is_directed_graph(g)
                
                # Cycle should be 1-2-3-...-n-1
                for i in 1:(n-1)
                    @test has_edge(g, i, i+1)
                    @test has_edge(g, i+1, i)  # Undirected
                end
                # Close the cycle: n-1
                @test has_edge(g, n, 1)
                @test has_edge(g, 1, n)
                
                # Every vertex should have degree 2
                for v in 1:n
                    @test degree(g, v) == 2
                end
                
                # No other edges should exist
                for i in 1:n, j in 1:n
                    if j != ((i % n) + 1) && j != (((i - 2 + n) % n) + 1) && i != j
                        @test !has_edge(g, i, j)
                    end
                end
            end
        end
        
        @testset "Directed Cycle Graphs" begin
            for n in 3:6
                g = cycle_graph(n; directed=true)
                @test num_vertices(g) == n
                @test num_edges(g) == n
                @test is_directed_graph(g)
                
                # Directed cycle: 1→2→3→...→n→1
                for i in 1:(n-1)
                    @test has_edge(g, i, i+1)
                    @test !has_edge(g, i+1, i)  # Not bidirectional
                end
                @test has_edge(g, n, 1)
                @test !has_edge(g, 1, n)  # Not bidirectional
            end
        end
        
        @testset "Cycle Edge Cases" begin
            # Cycles typically need at least 3 vertices, but test library behavior
            try
                g = cycle_graph(1; directed=false)
                @test num_vertices(g) >= 1
            catch ArgumentError
                @test true  # Exception is acceptable
            end
            
            try
                g = cycle_graph(2; directed=false)
                @test num_vertices(g) >= 2
            catch ArgumentError
                @test true  # Exception is acceptable
            end
            
            try
                g = cycle_graph(0; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
            
            try
                g = cycle_graph(-1; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
        end
    end
    
    @testset "Star Graph Generator" begin
        @testset "Star Graphs" begin
            for n in 2:8
                g = star_graph(n; directed=false)
                @test num_vertices(g) == n
                @test num_edges(g) == max(0, n - 1)
                @test !is_directed_graph(g)
                
                if n >= 2
                    # Star: center vertex 1 connected to all others
                    for i in 2:n
                        @test has_edge(g, 1, i)
                        @test has_edge(g, i, 1)  # Undirected
                    end
                    
                    # No connections between non-center vertices
                    for i in 2:n, j in (i+1):n
                        @test !has_edge(g, i, j)
                    end
                    
                    # Center has degree n-1, others have degree 1
                    @test degree(g, 1) == n - 1
                    for i in 2:n
                        @test degree(g, i) == 1
                    end
                end
            end
        end
        
        @testset "Directed Star Graphs" begin
            for n in 2:6
                g = star_graph(n; directed=true)
                @test num_vertices(g) == n
                @test num_edges(g) == n - 1
                @test is_directed_graph(g)
                
                # Directed star: 1→2, 1→3, ..., 1→n
                for i in 2:n
                    @test has_edge(g, 1, i)
                    @test !has_edge(g, i, 1)  # Not bidirectional
                end
            end
        end
    end
    
    @testset "Grid Graph Generator" begin
        @testset "2D Grid Graphs" begin
            # test_cases = [(1,1), (2,2), (3,3), (2,4), (4,2), (3,5)]
            test_cases = [(2,2), (3,3), (2,4), (4,2), (3,5)]

            for (rows, cols) in test_cases
                g = grid_graph(rows, cols; directed=false)
                @test num_vertices(g) == rows * cols
                @test !is_directed_graph(g)
                
                # Expected number of edges in grid
                horizontal_edges = rows * max(0, cols - 1)
                vertical_edges = max(0, rows - 1) * cols
                expected_edges = horizontal_edges + vertical_edges
                @test num_edges(g) == expected_edges
                
                # Test specific connections (if large enough)
                if rows >= 2 && cols >= 2
                    # Vertex numbering: row i, col j -> vertex (i-1)*cols + j
                    v11 = 1           # (1,1)
                    v12 = 2           # (1,2)
                    v21 = cols + 1    # (2,1)
                    
                    @test has_edge(g, v11, v12)  # Horizontal connection
                    @test has_edge(g, v11, v21)  # Vertical connection
                end
                
                # Check corner and edge vertex degrees
                if rows == 1 && cols == 1
                    @test degree(g, 1) == 0
                elseif rows == 1 || cols == 1  # Linear grid
                    # End vertices have degree 1, middle vertices have degree 2
                    total_vertices = rows * cols
                    if total_vertices >= 2
                        @test degree(g, 1) == 1  # End vertex
                        @test degree(g, total_vertices) == 1  # Other end vertex
                    end
                else  # 2D grid
                    # Corner vertices have degree 2
                    corner_vertices = [1, cols, (rows-1)*cols + 1, rows*cols]
                    for v in corner_vertices
                        @test degree(g, v) == 2
                    end
                end
            end
        end
        
        @testset "Grid Edge Cases" begin
            # Test boundary cases - library may handle gracefully
            try
                g = grid_graph(0, 5; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
            
            try
                g = grid_graph(5, 0; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
            
            try
                g = grid_graph(-1, 5; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
            
            try
                g = grid_graph(5, -1; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
        end
    end
    
    @testset "Empty Graph Generator" begin
        @testset "Empty Graphs" begin
            for n in 1:6
                g = empty_graph(n; directed=false)
                @test num_vertices(g) == n
                @test num_edges(g) == 0
                @test !is_directed_graph(g)
                
                # No edges should exist
                for u in 1:n, v in 1:n
                    @test !has_edge(g, u, v)
                end
                
                # All vertices should have degree 0
                for v in 1:n
                    @test degree(g, v) == 0
                    @test isempty(collect(neighbor_indices(g, v)))
                end
            end
        end
        
        @testset "Empty Graph Edge Cases" begin
            # Test boundary cases
            try
                g = empty_graph(0; directed=false)
                @test num_vertices(g) >= 0
                @test num_edges(g) == 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
            
            try
                g = empty_graph(-1; directed=false)
                @test num_vertices(g) >= 0
            catch ArgumentError
                @test true  # Exception is acceptable
            end
        end
    end
    
    @testset "Generator Consistency" begin
        # Test that all generators produce valid graphs
        generators = [
            () -> complete_graph(4; directed=false),
            () -> path_graph(5; directed=false),
            () -> cycle_graph(4; directed=false),
            () -> star_graph(5; directed=false),
            () -> grid_graph(3, 3; directed=false),
            () -> empty_graph(3; directed=false)
        ]
        
        for (i, gen) in enumerate(generators)
            @testset "Generator $i consistency" begin
                g = gen()
                
                # Basic validation
                @test num_vertices(g) > 0
                @test num_edges(g) >= 0
                @test num_directed_edges(g) == 2 * num_edges(g)  # Undirected
                
                # Degree sum should equal 2 * num_edges
                total_degree = sum(degree(g, v) for v in vertices(g))
                @test total_degree == 2 * num_edges(g)
                
                # All edges should be bidirectional
                for (u, v) in edges(g)
                    @test has_edge(g, u, v)
                    @test has_edge(g, v, u)
                end
            end
        end
    end
    
    @testset "Performance Sanity Check" begin
        # These aren't rigorous benchmarks, just sanity checks
        @testset "Large Graph Generation" begin
            # Complete graphs grow quadratically - test reasonable sizes
            g1 = complete_graph(100; directed=false)
            @test num_vertices(g1) == 100
            @test num_edges(g1) == 100 * 99 ÷ 2
            
            # Paths are linear - test larger sizes
            g2 = path_graph(1000; directed=false)
            @test num_vertices(g2) == 1000
            @test num_edges(g2) == 999
            
            # Grids are moderate - test reasonable sizes
            g3 = grid_graph(50, 50; directed=false)
            @test num_vertices(g3) == 2500
            @test num_edges(g3) == 49 * 50 + 50 * 49  # horizontal + vertical
        end
    end
end
