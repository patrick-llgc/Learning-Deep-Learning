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


## Groq
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

## DRAM (HBM, DDR) vs SRAM
- SRAM = on-chip, closest to compute


| Property      | **SRAM**                | **DRAM**                     |
| ------------- | ----------------------- | ---------------------------- |
| Structure       | 6-transistor (6T) latch     | charge in a capacitor (1T1C)       |
| Latency       | **Very low (~1ns)**     | **Higher (~50–100ns)**       |
| Bandwidth     | Very high (on-chip)     | Medium (DDR) to high (HBM)   |
| Need refresh? | **No**                  | **Yes**                      |
| Density       | Low                     | High                         |
| Power         | Higher (static leakage) | Lower                        |
| Cost          | Higher                  | Lower                        |
| Location      | **On-chip (cache)**     | **Off-chip (DDR/LPDDR/HBM)** |




## Types of DRAM
- DRAM (HBM/DDR) = off-chip, across a memory bus
	- HBM (High Bandwidth Memory) is DDR engineered for bandwidth via 3D integration, with TSV (thorugh silicon vias)
	- DDR (Double Data Rate DRAM), LPDDR is low power DDR. Double means that they can transfer data in both rising and falling edge. Used in gaming GPUs.

```	
Memory
 └── Volatile
      ├── SRAM (on-chip cache)
      └── DRAM
            ├── DDR / LPDDR
            └── HBM   
```


| Type of DRAM  | Power | Bandwidth | Capacity | Cost | Use Case       |
| ----- | ----- | --------- | -------- | ---- | -------------- |
| DDR   | Med   | Med       | High     | Low  | PCs / Servers  |
| LPDDR | Low   | Med       | Med      | Med  | Mobile / Edge  |
| HBM   | Med   | High      | Med      | High | AI / GPU / HPC |


### DDR vs HBM 

![](https://substackcdn.com/image/fetch/$s_!BU5Q!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6e5e7bdc-529d-453b-9e42-ef521e6db8d7_936x526.png)