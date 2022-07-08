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
mkdir bin
cp ../dora/target/release/dora-coordinator ./bin/dora-coordinator

# Setup a Jaeger suite to collect metrics
docker run -d -p6831:6831/udp -p6832:6832/udp -p16686:16686 jaegertracing/all-in-one:latest

# The running command
cargo build --all --release && ./bin/dora-coordinator run dataflow.yml
```

In order to fetch the metric from Jaeger, I wrote a `histogram.py` that will collect the latest data in a tabular form from Jaeger API. The script looks like this:

```python
import pandas as pd

link = "http://localhost:16686/api/traces?service=sender&lookback=6h&prettyPrint=true&limit=1"

df = pd.read_json(link)
df = pd.DataFrame(df.data[0]["spans"])
```

## Modifying the workload to test

If you go into the common crate, you will be able to change sync functions you want to benchmark. Those functions will be readily available on all threadpool nodes.

## Issues

Fixing `onnxruntime-rs` shared library issues requires that you link the downloaded `onnxruntime` shared library in your target build folder:

```bash
export ORT_USE_CUDA=1
export LD_LIBRARY_PATH="PATH TO DORA_CPU_BOUND..."/cpu-bound-async/target/release/build/onnxruntime-sys-186188f4edb1a21e/out/onnxruntime/onnxruntime-linux-x64-gpu-1.8.0/lib:${LD_LIBRARY_PATH}
```

Feel free to go on [haixuantao/onnxruntime-rs](https://github.com/haixuanTao/onnxruntime-rs) on branch `owned_environment` if you have any issues with the GPU.
