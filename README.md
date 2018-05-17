# GraggGPU

## Presentation of the different files

#### Testing the different algorithms/ technologies individually
**main_cpu_double.cu**: *CPU is used, with a double precision* to compute the n roots of a secular equation, n being given by the user. <\br>
**main_cpu_float.cu**: *CPU is used, with a single precision* to compute the n roots of a secular equation, n being given by the user. 
**main_gpu_double.cu**: *GPU is used, with a single precision* to compute the n roots of a secular equation, n being given by the user. 

#### Comparing the performance
**comparison.cu**: *Both CPU and GPU with single precision* are used to compute the n roots of a secular equation, n being given by the user. The differential in performance can be seen immediately in the console (running time and magnitude of the loss).
**comp_table.cu**: *Both CPU and GPU with single precision* are used to compute the n roots of a secular equation. A range of n is given by the user and performance (running time and magnitude of the loss) is stored on a csv file ('result.txt'). To compare them on a fair basis, the user can choose to run the test several time. For each n being tested, the GPU is warmed-up before the first iteration.
