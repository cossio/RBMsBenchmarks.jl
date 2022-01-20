module RBMsBenchmarks

using MKL
using Statistics, Random, LinearAlgebra
using CairoMakie, BenchmarkTools, StatsBase, ProgressMeter
import Flux, Zygote, MLDatasets, ValueHistories, BSON
import RestrictedBoltzmannMachines as RBMs

include("util.jl")
include("mnist.jl")

function __init__()
    if Threads.nthreads() == 1
        @warn """
        Julia has only one thread.
        It is recommended to start julia with `julia -t 8` or similar to run
        benchmarks in multiple threads.
        """
    end

    # we run different benchmarks in parallel, so it's best to use a single
    # BLAS thread to avoid oversubscribing
    if BLAS.get_num_threads() > 1
        @warn """
        BLAS threads is > 1.
        It is recommended to `export MKL_NUM_THREADS=1` before starting Julia.
        """
    end

    # so that datasets aRE dowloaded gracefully
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # directory where we place plots
    global OUTDIR = joinpath(pwd(), "out")
    @info "saving to $OUTDIR"
end

end
