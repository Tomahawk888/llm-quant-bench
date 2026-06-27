# This repository has moved

`quant-bench` has been merged into **cuda-codebook**, so everything lives in one place:

### https://github.com/Tomahawk888/cuda-codebook

In the merged repository:

- the quantization speed vs accuracy vs memory benchmark (AWQ, AQLM, fp16) is in
  [`bench/`](https://github.com/Tomahawk888/cuda-codebook/tree/main/bench)
- the model-level scripts (quantize Llama, the CUDA-graph serving integration, the
  AQLM-style additive-VQ training) are in
  [`model/`](https://github.com/Tomahawk888/cuda-codebook/tree/main/model)
- the CUDA kernels are at the repository root
- the paper is in
  [`paper/`](https://github.com/Tomahawk888/cuda-codebook/tree/main/paper)

This repo is kept as an archive. All development continues in `cuda-codebook`.
