# RBMsBenchmarks Julia scripts

Benchmarks for https://github.com/cossio/RestrictedBoltzmannMachines.jl.
The benchmarks are run in parallel.

You should set `export MKL_NUM_THREADS=1` if using MKL (recommended) or `export OPENBLAS_NUM_THREADS=1` before running these scripts.
Otherwise the internal threads of BLAS will conflict with the threads used to run the benchmarks an the overall performance will be slower.

For example, to run the MNIST benchmarks with 8 threads, do:

```
export MKL_NUM_THREADS=1
julia -t 8 mnist.jl
```

