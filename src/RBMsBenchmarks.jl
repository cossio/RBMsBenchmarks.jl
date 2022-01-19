module RBMsBenchmarks

using MKL
using Statistics, Random, LinearAlgebra
using CairoMakie, BenchmarkTools, StatsBase, ProgressMeter
import Flux, Zygote, MLDatasets, ValueHistories
import RestrictedBoltzmannMachines as RBMs

# make sure we are using MKL, which is faster
@show BLAS.get_config()

include("util.jl")

module Mnist
    import ..moving_average
    include("mnist.jl")
end

end
