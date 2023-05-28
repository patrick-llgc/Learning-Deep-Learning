# Semantic Occupancy Prediction

# Comparison

| Paper | Network Architecture | Loss | Eval | Range | Voxel Format | Voxel Size | Main dataset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MonoScene (CVPR 2022) | * 2D-to-3D with FLoSP (similar to OFT) \* 3D feature refinement with CRP (context relation prior) | * CE \* relation (affinity matrix loss) \* scal \* Feature proportion loss | SSC \* geometry: IoU \* semantic: mIoU | 51.2m x 51.2m x 6.4m | 256x256x32 | 0.2 m | SemanticKITTI |
| VoxFormer (CVPR 2023) | * 2D-to-3D with monodepth to generate sparse 3D query \* 2D-to-3D with BEVFormer to enhance 3D queries \* 3D feature enhancement with self-attention | * CE \* scene-class affinity loss | SSC \* geometry: IoU \* semantic: mIoU | 51.2m x 51.2m x 6.4m | 256x256x32 | 0.2 m | SemanticKITTI |
| TPVFormer (CVPR 2023) | * 2D-to-3D with BEVFormer to generate 3 TPV features \* Cross view attention to enhance TPV \* broadcast TPV features to volume features | * CE \* Lovasz softmax | No Eval for SOP | [-50m, 50m] x [-50m, 50m] x [-3m, 5m] | 200x200x16 | 0.5 m | NuScenes |
| SurroundOcc (Arxiv 2023/03) | * 2D-to-3D with BEVFormer \* 3D feature enhancement via 3D deconv | * CE \* scene-class affinity loss | * geometry: IoU \* semantic: mIoU | [-50m, 50m] x [-50m, 50m] x [-5m, 3m] | 200x200x16 | 0.5 m | NuScenes-derived dataset |
| OpenOccupancy (Arxiv 2023/03) | * 2D-to-3D with LSS \* 3D feature enhancement in a coarse-to-fine way | * CE \* Lovasz softmax \* geo scal \* sem scal \* direct depth supervision | * geometry: IoU \* semantic: mIoU | [-51.2m, 51.2m] x [-51.2m, 51.2m] x [-3m, 5m] | 512x512x40 | 0.2 m | NuScenes-Occupancy |
| Occ3D (Arxiv 2023/04) | * 2D-to-3D with BEVFormer \* Masked TopK voxel query to save computation | * CE \* binary CE for mask pred | mIoU | [-40m, 40m] x [-40m, 40m] x [-5m, 7.8m] | 200x200x32 | 0.4 m | Occ3D-Waymo  |
| ––〃–– | ––〃–– | ––〃–– | ––〃–– | [-40m, 40m] x [-40m, 40m] x [-1m, 5.4m] | 200x200x16 | 0.4 m | Occ3D-nuScenes |
| OccFormer (Arxiv 2023/04) | * 2D-to-3D with LSS \* 3D feature enhancecment \* Per class query (not per pixel query) based decoder. | * bipartite matching \* classification \* direct depth supervision | * geometry: IoU \* semantic: mIoU | 51.2m x 51.2m x 6.4m | 256x256x32 | 0.2 m | SemanticKITTI |



# Reference

- [MonoScene: Monocular 3D Semantic Scene Completion]([https://arxiv.org/abs/2112.00726](https://arxiv.org/abs/2112.00726)) <kbd>CVPR 2022</kbd> [[Notes](paper_notes/monoscene.md)] [Occupancy Network, single cam]
- [VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion]([https://arxiv.org/abs/2302.12251](https://arxiv.org/abs/2302.12251)), CVPR 2023
- [TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction]([https://arxiv.org/abs/2302.07817](https://arxiv.org/abs/2302.07817)), CVPR 2023
- [OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception](https://arxiv.org/abs/2303.03991) [[Notes](notion://www.notion.so/paper_notes/openoccupancy.md)] [Occupancy Network, Jiwen Lu]
- [SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving]([https://arxiv.org/abs/2303.09551](https://arxiv.org/abs/2303.09551)) [Occupancy Network]
- [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/abs/2304.14365) [Occupancy Network, Zhao Hang]
- [OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2304.05316) [Occupancy Network, PhiGent]