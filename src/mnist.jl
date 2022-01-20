
#=
Helper functions to evaluate different RBM training schemes on MNIST
=#

# Float32 is typically faster than Float64, and uses less memory
Float = Float32

# Load MNIST data
train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
# binarize
train_x = train_x .> 0.5
tests_x = tests_x .> 0.5
# join
datas_x = cat(train_x, tests_x; dims=3)
datas_y = cat(train_y, tests_y; dims=1)

# Helper function to plot RBM diagnostics and digits
function mnist_plots(; rbm, history, nsamples=4096, tsamples=5000, figtitle="RBM training")
	samples_F_from_rand = (avg = Float[], std = Float[])
	samples_v_from_rand = bitrand(28,28,nsamples)
	@showprogress "MC from rand " for t in 1:tsamples
	    samples_v_from_rand .= RBMs.sample_v_from_v(rbm, samples_v_from_rand)
	    F = RBMs.free_energy(rbm, samples_v_from_rand)
	    push!(samples_F_from_rand.avg, mean(F))
	    push!(samples_F_from_rand.std, std(F))
	end
	samples_F_from_data = (avg = Float[], std = Float[])
	samples_v_from_data = copy(train_x[:,:,rand(1:size(train_x,3),nsamples)])
	@showprogress "MC from data " for t in 1:tsamples
	    samples_v_from_data .= RBMs.sample_v_from_v(rbm, samples_v_from_data)
	    F = RBMs.free_energy(rbm, samples_v_from_data)
	    push!(samples_F_from_data.avg, mean(F))
	    push!(samples_F_from_data.std, std(F))
	end

	∂m = RBMs.∂free_energy(rbm, samples_v_from_rand)
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
	errorbars!(axF, 1:length(samples_F_from_rand.avg), samples_F_from_rand.avg, samples_F_from_rand.std, color=(:lightgray, 0.1))
	errorbars!(axF, 1:length(samples_F_from_data.avg), samples_F_from_data.avg, samples_F_from_data.std, color=(:lightblue, 0.1))
	lines!(axF, 1:length(samples_F_from_rand.avg), samples_F_from_rand.avg, color=:black, label="MC (rand)")
	lines!(axF, 1:length(samples_F_from_data.avg), samples_F_from_data.avg, color=:blue, label="MC (data)")
	hlines!(axF, mean(RBMs.free_energy(rbm, tests_x)), label="tests", color=:red, linestyle=:solid)
	hlines!(axF, mean(RBMs.free_energy(rbm, train_x)), label="train", color=:orange, linestyle=:dash)
	axislegend(axF, position=:rt, orientation=:horizontal, nbanks=2)

	axHist = Axis(fig[1,2][1,1], xlabel="free energy", ylabel="frequency")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, train_x), nbins=20), mode=:pdf), label="train")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, tests_x), nbins=20), mode=:pdf), label="tests")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, samples_v_from_rand), nbins=20), mode=:pdf), label="MC (rand)")
	stairs!(axHist, normalize(fit(Histogram, RBMs.free_energy(rbm, samples_v_from_data), nbins=20), mode=:pdf), label="MC (data)")
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
	    heatmap!(axImg, samples_v_from_rand[:, :, rand(1:size(samples_v_from_rand,3))])
	end

    set_theme!() # revert Makie defaults
	return fig
end



###############################
#= Benchmarks =#
###############################


function run_binary_mnist_benchmarks(output_dir::String = joinpath(pwd(), "../out"))
    M = 128
    batch = 256


    rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
    RBMs.initialize!(rbm, train_x)
    history = RBMs.cd!(rbm, train_x; epochs=100, batch=256, verbose=false, steps=1, optimizer=Flux.Descent(1e-4))
    fig = diagnostic_plots(; rbm, history)
    save(joinpath(output_dir, "CD-1_SGD.pdf"), fig)


    rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
    RBMs.initialize!(rbm, train_x);
    history = RBMs.cd!(rbm, train_x; epochs=100, batch=256, verbose=false, steps=1, optimizer=Flux.ADAM())
    fig = diagnostic_plots(; rbm, history)
    save(joinpath(output_dir, "CD-1_ADAM.pdf"), fig)


    rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
    RBMs.initialize!(rbm, train_x)
    history = RBMs.cd!(rbm, train_x; epochs=100, batch=256, verbose=false, steps=20, optimizer=Flux.ADAM())
    fig = diagnostic_plots(; rbm, history)
    save(joinpath(output_dir, "CD-20_ADAM.pdf"), fig)


    rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
    RBMs.initialize!(rbm, train_x)
    history = RBMs.pcd!(rbm, train_x; epochs=100, batchsize=batch, verbose=false, steps=1, optimizer=Flux.ADAM(1e-4))
    fig = diagnostic_plots(; rbm, history)
    save(joinpath(output_dir, "PCD-1_ADAM.pdf"), fig)


    rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
    RBMs.initialize!(rbm, train_x)
    history = RBMs.pcd_centered!(rbm, train_x; epochs=100, batchsize=batch, verbose=false, steps=1, optimizer=Flux.ADAM(1e-4), center_v=true, center_h=true, α=0.9)
    fig = diagnostic_plots(; rbm, history)
    save(joinpath(output_dir, "PCD-1_center_vh_ADAM.pdf"), fig)


    rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,M), zeros(Float,28,28,M))
    RBMs.initialize!(rbm, train_x)
    history = RBMs.pcd_centered!(rbm, train_x; epochs=100, batchsize=batch, verbose=false, steps=1, optimizer=Flux.ADAM(1e-4), center_v=true, center_h=false, α=0.9)
    fig = diagnostic_plots(; rbm, history)
    save(joinpath(output_dir, "PCD-1_center_v_ADAM.pdf"), fig)


    # Repro one of the experiments in Decelle's paper
    rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,500), zeros(Float,28,28,500))
    RBMs.initialize!(rbm, train_x)
    history = RBMs.cd!(rbm, train_x; epochs=1000, batch=500, verbose=false, steps=10, optimizer=Flux.Descent(1e-4))
    fig = diagnostic_plots(; rbm, history, tsamples=10)
    save(joinpath(output_dir, "Rdm-10_SGD.pdf"), fig)


end
