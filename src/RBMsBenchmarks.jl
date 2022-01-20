module RBMsBenchmarks

using MKL
using Statistics, Random, LinearAlgebra
using CairoMakie, BenchmarkTools, StatsBase, ProgressMeter
import Flux, Zygote, MLDatasets, ValueHistories
import RestrictedBoltzmannMachines as RBMs

# make sure we are using MKL, which is faster
@show BLAS.get_config()

include("util.jl")
include("mnist.jl")

end
