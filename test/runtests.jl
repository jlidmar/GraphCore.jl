using GraphCore
using Test

include("test_data.jl")
using .TestData

@testset "GraphCore Tests" verbose=true begin
    # Test organization:
    # 1. Core functionality (construction, basic operations)
    # 2. Interface compliance (all graph types implement GraphInterface correctly)
    # 3. Advanced features (properties, weights, mutations)
    # 4. Conversions and compatibility
    # 5. Performance and type stability

    @testset "Core Functionality" begin
        include("test_construction.jl")
        include("test_basic_operations.jl")
    end

    @testset "Interface Compliance" begin
        include("test_interface.jl")
        include("test_type_stability.jl")
    end

    @testset "Advanced Features" begin
        include("test_properties.jl")
        include("test_edge_weights.jl")
        include("test_mutation.jl")
    end

    @testset "Conversions & Compatibility" begin
        include("test_conversion.jl")
        include("test_compatibility.jl")
    end

    @testset "API & Generators" begin
        include("test_build_api.jl")
        include("test_generators.jl")
    end
end
