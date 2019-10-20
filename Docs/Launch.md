# Launch

You can find some examples in `./Examples`. See [Input & Output](Docs/Input-Output.md) for details about load and save of files. All supported command line parameters can be found in [Settings.inc.h](Source/Settings/Settings.inc.h).

# Parallel Mode

To launch multiple processes (MPI) just build with parallel support:
```sh
mpiexec -n <N> ./fdtd3d --cmd-from-file cmd.txt
```

To launch computations on GPU pass next parameters to `fdtd3d`:
```sh
--use-cuda
--cuda-gpus <gpu_id>
--num-cuda-threads-x <Nx>
--num-cuda-threads-y <Ny>
--num-cuda-threads-z <Nz>
```

If you want to use both MPI and Cuda, specify GPU id to be used on each computational node, or -1 for CPU computations on the selected node. Buffers should be at least of size `2` in this mode:
```sh
--use-cuda
--cuda-gpus <gpu_id1>,<gpu_id2>,...,<gpu_idN>
--cuda-buffer-size 2
--buffer-size 2
--num-cuda-threads-x <Nx>
--num-cuda-threads-y <Ny>
--num-cuda-threads-z <Nz>
```
