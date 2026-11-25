# Appendix E: Twelve Inference-Engineering Projects

This appendix lists twelve portfolio projects from a public LinkedIn post by Ravi A Samy. The descriptions here are this book's own. Projects 1, 4, 6 and 7 have kernel-level chapters in this book (Part 8). The systems projects live in the companion book, [Programming the H100](https://maximumbeings.github.io/programming-the-h100/). Chapter slugs below follow that site's file names; confirm the links after its next deploy.

| # | Project | Where it is covered |
|---|---|---|
| 1 | Custom Triton kernel optimizer: fused attention or RMSNorm, profiled with Nsight Compute, benchmarked against PyTorch | Chapter 23 (RMSNorm), Chapter 24 (decode attention) |
| 2 | Disaggregated prefill/decode cluster over RDMA/InfiniBand | [Programming the H100, Chapter 35](https://maximumbeings.github.io/programming-the-h100/35-disaggregated-prefill-decode/) (Part XII) |
| 3 | Hierarchical KV-cache offloader: VRAM, CPU RAM, NVMe | [Programming the H100, Chapter 34](https://maximumbeings.github.io/programming-the-h100/34-kv-cache-offloading/) (Part XII); builds on this book's Chapter 24 |
| 4 | Grammar-constrained decoding engine | Chapter 26 |
| 5 | MoE expert-parallel router | [Programming the H100, Chapter 33](https://maximumbeings.github.io/programming-the-h100/33-moe-expert-parallel/) (Part XII) |
| 6 | FP4/INT4 quantization and calibration pipeline | Chapter 25 |
| 7 | Speculative decoding harness | Chapter 27 (verify kernel and lossless test); the 70B multi-GPU run is [Programming the H100, Chapter 37](https://maximumbeings.github.io/programming-the-h100/37-speculative-decoding-70b/) (Part XII) |
| 8 | Queue-based GPU autoscaler (FinOps) | [Programming the H100, Chapter 40](https://maximumbeings.github.io/programming-the-h100/40-gpu-autoscaler/) (Part XIII) |
| 9 | Semantic AI gateway and cache | [Programming the H100, Chapter 38](https://maximumbeings.github.io/programming-the-h100/38-semantic-cache/) (Part XIII) and the combined gateway in [Chapter 42](https://maximumbeings.github.io/programming-the-h100/42-gateway-capstone/) |
| 10 | Multi-model complexity router | [Programming the H100, Chapter 39](https://maximumbeings.github.io/programming-the-h100/39-complexity-router/) (Part XIII) |
| 11 | WebGPU / MLX edge fallback | [Programming the H100, Chapter 41](https://maximumbeings.github.io/programming-the-h100/41-edge-fallback-resilience/) (Part XIII) |
| 12 | Public inference benchmark teardown | [Programming the H100, Chapter 36](https://maximumbeings.github.io/programming-the-h100/36-benchmark-teardown/) (Part XII) |

Every project should end with a measured number, not a claim: percent of peak bandwidth, speedup against a stated baseline, perplexity delta, cost per million tokens. State the workload, software versions and hardware, and report where the approach loses as well as where it wins.
