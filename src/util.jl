
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

# so that datasets are dowloaded gracefully
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# directory where we place plots
const OUTDIR = joinpath(pwd(), "out")
@info "saving to $OUTDIR"


#= ##################
Util functions
=# ##################

function moving_average(x::AbstractVector, σ::Int = max(1, length(x) ÷ 20))
	# https://stackoverflow.com/a/59589877/855050
	kernel = OffsetArray(fill(1/(2σ + 1), 2σ + 1), -σ:σ)
	return imfilter(x, kernel)
end
