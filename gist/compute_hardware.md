# [Requirements to compute hardware between training and inference](gist/compute_hardware.md)

## Comparison between training, cloud-inference, and edge-inference

Training improves the model, cloud inference serves users, and edge inference controls the physical world.

| Dimension            | Training              | Cloud Inference        | Edge Inference           |
| -------------------- | --------------------- | ---------------------- | ------------------------ |
| Objective            | Model quality         | Scale + cost           | Real-time control        |
| Mode                 | Offline               | Online                 | Closed-loop              |
| Batch                | Large                 | Medium                 | 1                        |
| Latency Requirements | None                  | Low                    | Hard real-time           |
| Scaling              | Scale-out             | Scale-out              | No scale-out             |
| Bottleneck           | Compute / HBM / comms | KV / scheduling / $    | Latency / jitter / power |
| Memory Preference    | HBM                   | HBM + KV optimizations | SRAM-first               |
| Cost Model           | $ per training run    | $ per token            | $ per latency + Watt     |


# Groq
*Groq was aqcuired by Nvidia in 12/2025 for 20 Billion USD.*

> GPUs win throughput economics; Groq wins latency economics.

Groq's LPU (language processing unit)

* SRAM-Heavy, Deterministic Inference ASIC. 
* Keep working data (KV, activations) on-chip in SRAM and pre-schedule all compute + dataflow to eliminate stalls, jitter, and DRAM round trips.

* Strategic Summary
	* Groq = SRAM-first
		* Optimizes for KV locality
		* Targets latency + determinism
		* Perfect for robotics / AV / edge / Batchsize=1 workloads
	* GPU = HBM-first
		* Optimizes for bandwidth & throughput
		* Targets training + cloud batch inference
		* Perfect for QPS-heavy LLM serving

