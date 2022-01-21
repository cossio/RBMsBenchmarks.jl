#=
Preliminaries
=#

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MKL, Statistics, Random, LinearAlgebra, Logging, Serialization
using CairoMakie, BenchmarkTools, StatsBase, ProgressMeter, ImageFiltering, OffsetArrays
import Flux, Zygote, MLDatasets, ValueHistories
import RestrictedBoltzmannMachines as RBMs

# so that MNIST is dowloaded gracefully
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

Threads.nthreads() > 1 || @warn "running on 1 Julia thread"
BLAS.get_num_threads() == 1 || @warn ">1 BLAS threads"

# directory where we place plots
const OUTDIR = joinpath(pwd(), "out/mnist")
rm(OUTDIR; force=true, recursive=true) # clean
mkpath(OUTDIR)
@info "saving to $OUTDIR"

#= #############
Util functions
=# #############

function moving_average(x::AbstractVector, σ::Int = max(1, length(x) ÷ 20))
	# https://stackoverflow.com/a/59589877/855050
	kernel = OffsetArray(fill(1/(2σ + 1), 2σ + 1), -σ:σ)
	return imfilter(x, kernel)
end

#= #############
Load MNIST
=# #############

# Float32 is typically faster than Float64, and uses less memory
const Float = Float32

# Load MNIST data
train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
# binarize
train_x = train_x .> 0.5
tests_x = tests_x .> 0.5
# all data together
datas_x = cat(train_x, tests_x; dims=3)
datas_y = cat(train_y, tests_y; dims=1)

#= #############
Helper functions used by the benchmarks
=# #############

# produces generated samples from an RBM
function generate_samples(; rbm, chains=4000, len=50)
    F_from_rand = (avg = Float[], std = Float[])
	v_from_rand = bitrand(28, 28, chains)
	for _ in 1:len
	    v_from_rand .= RBMs.sample_v_from_v(rbm, v_from_rand)
	    F = RBMs.free_energy(rbm, v_from_rand)
	    push!(F_from_rand.avg, mean(F))
	    push!(F_from_rand.std, std(F))
	end
	F_from_data = (avg = Float[], std = Float[])
	v_from_data = copy(train_x[:,:,rand(1:size(train_x,3), chains)])
	for _ in 1:len
	    v_from_data .= RBMs.sample_v_from_v(rbm, v_from_data)
	    F = RBMs.free_energy(rbm, v_from_data)
	    push!(F_from_data.avg, mean(F))
	    push!(F_from_data.std, std(F))
	end
    return (; F_from_rand, v_from_rand, F_from_data, v_from_data)
end

# Helper function to plot RBM diagnostics and digits
function produce_plot(; rbm, history, samples)
	∂m = RBMs.∂free_energy(rbm, samples.v_from_rand)
	∂d = RBMs.∂free_energy(rbm, tests_x)
	train_F = RBMs.free_energy(rbm, train_x)
	tests_F = RBMs.free_energy(rbm, tests_x)
	datas_F = RBMs.free_energy(rbm, datas_x)

    set_theme!(Theme(fontsize = 12))
	fig = Figure(resolution=(1200, 800))

	# pseudolikelihood during training
	axPL = Axis(fig[1:2,1][1,1], xlabel="epoch", ylabel="ln(PL)")
	lines!(axPL, get(history, :lpl)..., label="PL", color=(:blue, 0.3))
	lines!(axPL, get(history, :lpl)[1], moving_average(get(history, :lpl)[2]), label="PL", color=:blue)

	ax∂ = Axis(fig[1:2,1][2,1], xlabel="epoch", ylabel="||∂||")
	batches_in_epochs = range(start=get(history, :lpl)[1][begin], stop=get(history, :lpl)[1][end], length=length(get(history, :∂)[1]))
	lines!(ax∂, batches_in_epochs, [∂.w/sqrt(length(rbm.w)) for ∂ in get(history, :∂)[2]], label="∂w", color=(:red, 0.25))
	lines!(ax∂, batches_in_epochs, moving_average([∂.w/sqrt(length(rbm.w)) for ∂ in get(history, :∂)[2]]), color=:red)

	axstat = Axis(fig[1,3], xlabel="data statistics", ylabel="model statistics")
	scatter!(axstat, -vec(∂d.w), -vec(∂m.w), label="∂w", markersize=5)
	for p in propertynames(∂m.visible)
	    scatter!(axstat, -vec(getproperty(∂d.visible, p)), -vec(getproperty(∂m.visible, p)), label="v∂$p", markersize=5)
	    lines!(ax∂, batches_in_epochs, [getproperty(∂.visible, p)/sqrt(length(rbm.visible)) for ∂ in get(history, :∂)[2]], color=(:gold, 0.25))
	    lines!(ax∂, batches_in_epochs, moving_average([getproperty(∂.visible, p)/sqrt(length(rbm.visible)) for ∂ in get(history, :∂)[2]]), label="v∂$p", color=:gold)
	end
	for p in propertynames(∂m.hidden)
	    scatter!(axstat, -vec(getproperty(∂d.hidden, p)), -vec(getproperty(∂m.hidden, p)), label="h∂$p", markersize=5)
	    lines!(ax∂, batches_in_epochs, [getproperty(∂.hidden, p)/sqrt(length(rbm.hidden)) for ∂ in get(history, :∂)[2]], color=(:green, 0.25))
	    lines!(ax∂, batches_in_epochs, moving_average([getproperty(∂.hidden, p)/sqrt(length(rbm.hidden)) for ∂ in get(history, :∂)[2]]), label="h∂$p", color=:green)
	end
	abline!(axstat, 0, 1, color=:red, linewidth=2)
	xlims!(axstat, 0, 1)
	ylims!(axstat, 0, 1)
	axislegend(ax∂, position=:rt, orientation=:horizontal)
	axislegend(axstat, position=:rb)

	axF = Axis(fig[1:2,1][3,1], xlabel="step", ylabel="F")
	errorbars!(axF, 1:length(samples.F_from_rand.avg), samples.F_from_rand.avg, samples.F_from_rand.std, color=(:lightgray, 0.1))
	errorbars!(axF, 1:length(samples.F_from_data.avg), samples.F_from_data.avg, samples.F_from_data.std, color=(:lightblue, 0.1))
	lines!(axF, 1:length(samples.F_from_rand.avg), samples.F_from_rand.avg, color=:black, label="MC (rand)")
	lines!(axF, 1:length(samples.F_from_data.avg), samples.F_from_data.avg, color=:blue, label="MC (data)")
	hlines!(axF, mean(RBMs.free_energy(rbm, tests_x)), label="tests", color=:red, linestyle=:solid)
	hlines!(axF, mean(RBMs.free_energy(rbm, train_x)), label="train", color=:orange, linestyle=:dash)
	axislegend(axF, position=:rt, orientation=:horizontal, nbanks=2)

	axHist = Axis(fig[1,2][1,1], xlabel="free energy", ylabel="frequency")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, train_x), nbins=20), mode=:pdf), label="train")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, tests_x), nbins=20), mode=:pdf), label="tests")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, samples.v_from_rand), nbins=20), mode=:pdf), label="MC (rand)")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, samples.v_from_data), nbins=20), mode=:pdf), label="MC (data)")
	axislegend(axHist, position=:rt, orientation=:horizontal, nbanks=2)

	axDigits = Axis(fig[2,2][1,1], ylabel="-free energy", xticks = 0:9)
	ylims!(axDigits, -mean(datas_F) - 3std(datas_F), -mean(datas_F) + 3std(datas_F))
	barplot!(axDigits, 0:9, -[mean(train_F[train_y .== d]) for d in 0:9], color=:gray, label="train", width=0.5)
	barplot!(axDigits, (0:9) .+ 0.5, -[mean(tests_F[tests_y .== d]) for d in 0:9], color=:lightblue, label="tests", width=0.5)
	errorbars!(axDigits, 0:9, -[mean(train_F[train_y .== d]) for d in 0:9], [std(train_F[train_y .== d]) for d in 0:9], color=:black, linewidth=2, label="std", whiskerwidth=5)
	errorbars!(axDigits, (0:9) .+ 0.5, -[mean(tests_F[tests_y .== d]) for d in 0:9], [std(tests_F[tests_y .== d]) for d in 0:9], color=:black, linewidth=2, whiskerwidth=5)
	axislegend(axDigits, position=:rt, orientation=:horizontal, margin=(0,0,0,0))

	axΔt = Axis(fig[2,2][2,1], xlabel="secs/epoch", ylabel="freq.")
	hist!(axΔt, get(history, :Δt)[2], color=:gray)
	vlines!(axΔt, mean(get(history, :Δt)[2]), color=:red)
	text!(axΔt, "total: $(round(sum(get(history, :Δt)[2])/60, digits=2)) min", position=axΔt.finallimits[].origin + axΔt.finallimits[].widths/2, color=:red, textsize=16)

	# show some sampled digits
	for i in 1:3, j in 1:3
	    axImg = Axis(fig[2,3][i,j], yreversed=true)
	    hidedecorations!(axImg)
	    heatmap!(axImg, samples.v_from_rand[:, :, rand(1:size(samples.v_from_rand,3))])
	end

    set_theme!() # revert Makie defaults
	return fig
end


###############################
#= Benchmark definitions=#
###############################


function binary_cd_sgd(; M, batchsize, epochs)
    @sync for η in [0.0001, 0.001], k in [1, 10]
        prefix = "$OUTDIR/CD-$(k)_SGD-$(η)"
        Threads.@spawn open("$prefix.log", "w") do io
            with_logger(SimpleLogger(io, Logging.Debug)) do
                rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
                RBMs.initialize!(rbm, train_x)
                history = RBMs.cd!(
                    rbm, train_x; epochs, batchsize, steps=k, optimizer=Flux.Descent(η)
                )
                t_sampling = @elapsed samples = generate_samples(; rbm)
                serialize("$prefix.data", (; rbm, history, samples))
                @info "saved; sampling took $t_sampling seconds"
            end
        end
    end
end


function binary_pcd_sgd(; M, batchsize, epochs)
    @sync for η in [0.0001, 0.001], k in [1, 10]
        prefix = "$OUTDIR/CD-$(k)_SGD-$(η)"
        Threads.@spawn open("$prefix.log", "w") do io
            with_logger(SimpleLogger(io, Logging.Debug)) do
                rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
                RBMs.initialize!(rbm, train_x)
                history = RBMs.pcd!(
                    rbm, train_x; epochs, batchsize, steps=k, optimizer=Flux.Descent(η)
                )
                t_sampling = @elapsed samples = generate_samples(; rbm)
                serialize("$prefix.data", (; rbm, history, samples))
                @info "saved; sampling took $t_sampling seconds"
            end
        end
    end
end


function binary_cd_adam(; M, batchsize, epochs)
    @sync for η in [0.0001, 0.001], k in [1, 10]
        prefix = "$OUTDIR/CD-$(k)_ADAM-$(η)"
        Threads.@spawn open("$prefix.log", "w") do io
            with_logger(SimpleLogger(io, Logging.Debug)) do
                rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
                RBMs.initialize!(rbm, train_x)
                history = RBMs.cd!(
                    rbm, train_x; epochs, batchsize, steps=k, optimizer=Flux.ADAM(η)
                )
                t_sampling = @elapsed samples = generate_samples(; rbm)
                serialize("$prefix.data", (; rbm, history, samples))
                @info "saved; sampling took $t_sampling seconds"
            end
        end
    end
end


function binary_pcd_adam(; M, batchsize, epochs)
    @sync for η in [0.0001, 0.001], k in [1, 10]
        prefix = "$OUTDIR/PCD-$(k)_ADAM-$(η)"
        Threads.@spawn open("$prefix.log", "w") do io
            with_logger(SimpleLogger(io, Logging.Debug)) do
                rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
                RBMs.initialize!(rbm, train_x)
                history = RBMs.pcd!(
                    rbm, train_x; epochs, batchsize, steps=k, optimizer=Flux.ADAM(η)
                )
                t_sampling = @elapsed samples = generate_samples(; rbm)
                serialize("$prefix.data", (; rbm, history, samples))
                @info "saved; sampling took $t_sampling seconds"
            end
        end
    end
end


function binary_pcd_center_sgd(; M, batchsize, epochs)
    @sync for η in [0.0001, 0.001], k in [1, 10], cv in [true, false], ch in [true, false]
        prefix = "$OUTDIR/PCD-$(k)_cv-$(Int(cv))_ch-$(Int(ch))SGD-$(η)"
        Threads.@spawn open("$prefix.log", "w") do io
            with_logger(SimpleLogger(io, Logging.Debug)) do
                rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
                RBMs.initialize!(rbm, train_x)
                history = RBMs.pcd_centered!(
                    rbm, train_x; epochs, batchsize, steps=k, optimizer=Flux.Descent(η), center_v=cv, center_h=ch
                )
                t_sampling = @elapsed samples = generate_samples(; rbm)
                serialize("$prefix.data", (; rbm, history, samples))
                @info "saved; sampling took $t_sampling seconds"
            end
        end
    end
end


function binary_pcd_center_adam(; M, batchsize, epochs)
    @sync for η in [0.0001, 0.001], k in [1, 10], cv in [true, false], ch in [true, false]
        prefix = "$OUTDIR/PCD-$(k)_cv-$(Int(cv))_ch-$(Int(ch))_ADAM-$(η)"
        Threads.@spawn open("$prefix.log", "w") do io
            with_logger(SimpleLogger(io, Logging.Debug)) do
                rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
                RBMs.initialize!(rbm, train_x)
                history = RBMs.pcd_centered!(
                    rbm, train_x; epochs, batchsize, steps=k, optimizer=Flux.ADAM(η), center_v=cv, center_h=ch
                )
                t_sampling = @elapsed samples = generate_samples(; rbm)
                serialize("$prefix.data", (; rbm, history, samples))
                @info "saved; sampling took $t_sampling seconds"
            end
        end
    end
end


function binary_rdm_sgd(; M, batchsize, epochs)
    # Repro one of the experiments in Decelle's paper
    # http://arxiv.org/abs/2105.13889
    @sync for η in [0.0001, 0.001], k in [10, 20]
        prefix = "$OUTDIR/Rdm-$(k)_SGD-$(η)"
        Threads.@spawn open("$prefix.log", "w") do io
            with_logger(SimpleLogger(io, Logging.Debug)) do
                rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
                RBMs.initialize!(rbm, train_x)
                history = RBMs.rdm!(rbm, train_x; epochs, batchsize, steps=k, optimizer=Flux.Descent(1e-4))
                t_sampling = @elapsed samples = generate_samples(; rbm, len=k)
                serialize("$prefix.data", (; rbm, history, samples))
                @info "saved; sampling took $t_sampling seconds"
            end
        end
    end
end


##############
# Run benchmarks in parallel
##############

M = 128 # number of hidden units
batchsize = 256 # batch size
epochs = 3

# list all the benchmark functions defined above
benchmarks = (
    binary_cd_sgd,
    binary_pcd_sgd,
    binary_cd_adam,
    binary_pcd_adam,
    binary_pcd_center_sgd,
    binary_pcd_center_adam,
    binary_rdm_sgd
)

@sync for benchmark in benchmarks
    Threads.@spawn benchmark(; M, batchsize, epochs)
end

#= Make and save plots. Since Makie is not threadsafe
(see https://github.com/JuliaPlots/Makie.jl/issues/812)
we do this in series. =#
for file in readdir(OUTDIR; join=true)
    if endswith(file, ".data")
        @info "plotting $(basename(file))"
        (; rbm, history, samples) = deserialize(file)
        fig = produce_plot(; rbm, history, samples)
        save(file[begin:(end - 11)] * ".pdf", fig)
    end
end
