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

### Simulation
```bash
python main.py --arch=CPU     # runs the simulation on the CPU
python main.py --arch=CUDA    # runs the simulation on the GPU
```

### Options
``` bash
  -h, --help            show this help message and exit
  -c [CONFIGURATION], --configuration [CONFIGURATION]
                        Available Configurations:
                        [0] -> Centered Dam Break
                        [1] -> Dam Break
                        [2] -> Spherefall
                        [3] -> Stationary Pool
                        [4] -> Waterjet
                        [5] -> Waterjet Hits Pool
  -q [QUALITY], --quality [QUALITY]
                        Choose a quality multiplicator for the simulation (higher is better).
  -a [{CPU,CUDA}], --arch [{CPU,CUDA}]
                        Choose the Taichi architecture to run on.
  -g [{Staggered,Collocated}], --grid [{Staggered,Collocated}]
                        Choose the grid type (collocated or staggered)

Press R to reset, SPACE to pause/unpause the simulation!
```
