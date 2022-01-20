export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

julia --project=. -t 16 src/mnist.jl