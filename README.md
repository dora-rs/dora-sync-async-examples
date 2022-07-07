# Examples and test harness of several async sync threadpools

This is a testing repository of dora nodes that implements several cpu sync and async threadpool. Feel free to use it, to figure out what works best for you. This repository might not be up to date.
## Getting started

```bash
# Download the data
./download_data.sh
# Clone dora in parent folder if it is not already there
cd ../
git clone https://github.com/dora-rs/dora.git
# Get an executable coordinator from `dora`
cd ./dora
cargo build  --manifest-path coordinator/Cargo.toml --release
cd ../dora-sync-async-examples
cp ../dora/target/release/dora-coordinator ./bin/dora-coordinator

# The running command
cargo build --all --release && ./bin/dora-coordinator run dataflow.yml
```
## Modifying the workload to test

If you go in the common crate, you will be able to change synchronous functions you want to benchmark, that will be readily available on all threadpool nodes.

## Issues

Fixing `onnxruntime-rs` shared library issues requires that you link the downloaded `onnxruntime` shared library in your target build folder:

```bash
export ORT_USE_CUDA=1
export LD_LIBRARY_PATH="PATH TO DORA_CPU_BOUND..."/cpu-bound-async/target/release/build/onnxruntime-sys-186188f4edb1a21e/out/onnxruntime/onnxruntime-linux-x64-gpu-1.8.0/lib:${LD_LIBRARY_PATH}
```

Feel free to go on [haixuantao/onnxruntime-rs](https://github.com/haixuanTao/onnxruntime-rs) on branch `owned_environment` if you have any issues with the GPU.