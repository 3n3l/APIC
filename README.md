## The Affine Particle-In-Cell Method (APIC)
Implementation of the [The Affine Particle-In-Cell Method](https://doi.org/10.1145/2766996) in [Taichi](https://www.taichi-lang.org/).

<!-- 
<p align="center">
  <img src="https://github.com/user-attachments/assets/747153a1-90e8-4462-be5a-e2413c021f92" alt="animated" height=720px />
</p>
-->

![video-ezgif com-optimize (1)](https://github.com/user-attachments/assets/7a237a82-fd0f-4055-885c-54f443c6fefc)


### Installation
Dependencies are managed with Conda:
```bash
conda env create -f environment.yaml
conda activate APIC
```
You also need to install [cuSPARSE libraries](https://pypi.org/project/nvidia-cusparse-cu12/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for the CUDA backend, and [Vulkan Drivers](https://developer.nvidia.com/vulkan-driver) for the GGUI frontend. Both of these are optional, but result in better performance and visibility.
