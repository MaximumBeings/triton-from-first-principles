# Appendix E: Twelve Inference-Engineering Projects

This appendix lists twelve portfolio projects from a public LinkedIn post by Ravi A Samy. The descriptions here are this book's own. Projects 1, 4, 6 and 7 have kernel-level chapters in this book (Part 8). The systems projects live in the companion book, [Programming the H100](https://maximumbeings.github.io/programming-the-h100/).

| # | Project | Where it is covered |
|---|---|---|
| 1 | Custom Triton kernel optimizer: fused attention or RMSNorm, profiled with Nsight Compute, benchmarked against PyTorch | Chapter 23 (RMSNorm), Chapter 24 (decode attention) |
| 2 | Disaggregated prefill/decode cluster over RDMA/InfiniBand | Programming the H100, Part XII |
| 3 | Hierarchical KV-cache offloader: VRAM, CPU RAM, NVMe | Programming the H100, Part XII (needs Chapter 24) |
| 4 | Grammar-constrained decoding engine | Chapter 26 (planned) |
| 5 | MoE expert-parallel router | Programming the H100, Part XII |
| 6 | FP4/INT4 quantization and calibration pipeline | Chapter 25 (planned) |
| 7 | Speculative decoding harness | Chapter 27 (planned); 70B run in Programming the H100 |
| 8 | Queue-based GPU autoscaler (FinOps) | Programming the H100, Part XIII |
| 9 | Semantic AI gateway and cache | Programming the H100, Part XIII |
| 10 | Multi-model complexity router | Programming the H100, Part XIII |
| 11 | WebGPU / MLX edge fallback | Programming the H100, Part XIII |
| 12 | Public inference benchmark teardown | Programming the H100, Part XII |

Chapters marked "planned" are not written yet.

Every project should end with a measured number, not a claim: percent of peak bandwidth, speedup against a stated baseline, perplexity delta, cost per million tokens. State the workload, software versions and hardware, and report where the approach loses as well as where it wins.
