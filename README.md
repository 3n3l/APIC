## The Affine Particle-In-Cell Method (APIC)
Implementation of the [The Affine Particle-In-Cell Method](https://doi.org/10.1145/2766996) in [Taichi](https://www.taichi-lang.org/).
<p align="center">
  <img src="https://github.com/user-attachments/assets/ce2ce1a3-9116-458a-91e1-3e3cab57ee09" alt="animated" height=400px> 
  <img src="https://github.com/user-attachments/assets/fc6c4c5e-af70-4e83-8c49-0ec339bcce8b" alt="animated" height=400px>
</p>

### Installation
Dependencies are managed with Conda:
```bash
conda env create -f environment.yaml
conda activate APIC
```
You also need to install [cuSPARSE libraries](https://pypi.org/project/nvidia-cusparse-cu12/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for the CUDA backend, and [Vulkan Drivers](https://developer.nvidia.com/vulkan-driver) for the GGUI frontend. Both of these are optional, but result in better performance and visibility.
