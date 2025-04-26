## The Affine Particle-In-Cell Method (APIC)
Implementation of the [The Affine Particle-In-Cell Method](https://doi.org/10.1145/2766996) in [Taichi](https://www.taichi-lang.org/).


### Installation
Dependencies are managed with Conda:
```bash
conda env create -f environment.yaml
conda activate APIC
```
You also need to install [cuSPARSE libraries](https://pypi.org/project/nvidia-cusparse-cu12/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for the CUDA backend, and [Vulkan Drivers](https://developer.nvidia.com/vulkan-driver) for the GGUI frontend. Both of these are optional, but result in better performance and visibility.
