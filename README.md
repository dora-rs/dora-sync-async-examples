# Tests alternatives for CPU-bound tasks.

Fixing `onnxruntime-rs` shared library issues requires that you link the downloaded `onnxruntime` shared library in your target build folder:

```bash
export ORT_USE_CUDA=1
export LD_LIBRARY_PATH="PATH TO DORA_CPU_BOUND..."/cpu-bound-async/target/release/build/onnxruntime-sys-186188f4edb1a21e/out/onnxruntime/onnxruntime-linux-x64-gpu-1.8.0/lib:${LD_LIBRARY_PATH}
```

Feel free to go on [haixuantao/onnxruntime-rs](https://github.com/haixuanTao/onnxruntime-rs) on branch `owned_environment` if you have any issues with the GPU.