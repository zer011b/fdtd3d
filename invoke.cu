#include "invoke.h"

#include "cstdio"

__global__ void fdtd_step_Ez (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy, FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev, FieldValue gridTimeStep, FieldValue gridStep)
{
  int index1 = blockIdx.x * blockDim.y + blockIdx.y;
  int index2 = (blockIdx.x - 1) * blockDim.y + blockIdx.y;
  int index3 = blockIdx.x * blockDim.y + blockIdx.y - 1;

  Ez[index1] = Ez_prev[index1] + (gridTimeStep / (0.0000000000088541878176203892 * gridStep)) *
    (Hx_prev[index3] - Hx_prev[index1] + Hy_prev[index1] - Hy_prev[index2]);

  Ez_prev[index1] = Ez[index1];
}

__global__ void fdtd_step_Hx (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy, FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev, FieldValue gridTimeStep, FieldValue gridStep)
{
  int index1 = blockIdx.x * blockDim.y + blockIdx.y;
  int index2 = blockIdx.x * blockDim.y + blockIdx.y + 1;

  Hx[index1] = Hx_prev[index1] + (gridTimeStep / (0.0000012566370614359173 * gridStep)) *
    (Ez_prev[index1] - Ez_prev[index2]);

  Hx_prev[index1] = Hx[index1];
}

__global__ void fdtd_step_Hy (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy, FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev, FieldValue gridTimeStep, FieldValue gridStep)
{
  int index1 = blockIdx.x * blockDim.y + blockIdx.y;
  int index2 = (blockIdx.x + 1) * blockDim.y + blockIdx.y;

  Hy[index1] = Hy_prev[index1] + (gridTimeStep / (0.0000012566370614359173 * gridStep)) *
    (Ez_prev[index2] - Ez_prev[index1]);

  Hy_prev[index1] = Hy[index1];
}

void execute (FieldValue *tmp_Ez, FieldValue *tmp_Hx, FieldValue *tmp_Hy, FieldValue *tmp_Ez_prev, FieldValue *tmp_Hx_prev, FieldValue *tmp_Hy_prev,
              int sx, int sy,
              FieldValue gridTimeStep,
              FieldValue gridStep,
              int totalStep)
{
  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;

  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;

  int size = sx * sy;

  cudaMalloc ((void **) &Ez_cuda, size);
  cudaMalloc ((void **) &Hx_cuda, size);
  cudaMalloc ((void **) &Hy_cuda, size);

  cudaMalloc ((void **) &Ez_cuda_prev, size);
  cudaMalloc ((void **) &Hx_cuda_prev, size);
  cudaMalloc ((void **) &Hy_cuda_prev, size);

  cudaMemcpy (Ez_cuda, tmp_Ez, size, cudaMemcpyHostToDevice);
  cudaMemcpy (Hx_cuda, tmp_Hx, size, cudaMemcpyHostToDevice);
  cudaMemcpy (Hy_cuda, tmp_Hy, size, cudaMemcpyHostToDevice);

  cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, size, cudaMemcpyHostToDevice);
  cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, size, cudaMemcpyHostToDevice);
  cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, size, cudaMemcpyHostToDevice);

  for (int t = 0; t < totalStep; ++t)
  {
    printf ("%d", t);
    dim3 N (sx, sy);
    fdtd_step_Ez <<< N, 1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep);
    fdtd_step_Hx <<< N, 1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep);
    fdtd_step_Hy <<< N, 1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep);
  }

  cudaMemcpy (tmp_Ez, Ez_cuda, size, cudaMemcpyDeviceToHost);
  cudaMemcpy (tmp_Hx, Hx_cuda, size, cudaMemcpyDeviceToHost);
  cudaMemcpy (tmp_Hy, Hy_cuda, size, cudaMemcpyDeviceToHost);

  cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, size, cudaMemcpyDeviceToHost);
  cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, size, cudaMemcpyDeviceToHost);
  cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, size, cudaMemcpyDeviceToHost);



  cudaFree (Ez_cuda);
  cudaFree (Hx_cuda);
  cudaFree (Hy_cuda);

  cudaFree (Ez_cuda_prev);
  cudaFree (Hx_cuda_prev);
  cudaFree (Hy_cuda_prev);
//   int N = 10;
//   int *a, *b, *c;
// int *d_a, *d_b, *d_c;
// int size = N * sizeof( int );
//
// int THREADS_PER_BLOCK = 1;
//
// /* allocate space for device copies of a, b, c */
//
// int err = cudaMalloc( (void **) &d_a, size );
// if (err != cudaSuccess)
// {
//   printf ("Err");
//   printf ("%s", cudaGetErrorString ((cudaError_t)err));
//   exit (1);
// }
// err = cudaMalloc( (void **) &d_b, size );
// err = cudaMalloc( (void **) &d_c, size );
//
// /* allocate space for host copies of a, b, c and setup input values */
//
// a = (int *)malloc( size );
// b = (int *)malloc( size );
// c = (int *)malloc( size );
//
// for( int i = 0; i < N; i++ )
// {
//   a[i] = b[i] = i;
//   c[i] = 0;
// }
//
// /* copy inputs to device */
// /* fix the parameters needed to copy data to the device */
// cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
// cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );
//
// /* launch the kernel on the GPU */
// /* insert the launch parameters to launch the kernel properly using blocks and threads */
// vector_add<<< N, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );
//
// /* copy result back to host */
// /* fix the parameters needed to copy data back to the host */
// cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );
//
//
// printf( "c[%d] = %d\n",0,c[0] );
// printf( "c[%d] = %d\n",N-1, c[N-1] );
//
//   printf( "a[%d] = %d\n",0,a[0] );
// printf( "a[%d] = %d\n",N-1, a[N-1] );
//
// /* clean up */
//
// free(a);
// free(b);
// free(c);
// cudaFree( d_a );
// cudaFree( d_b );
// cudaFree( d_c );
}
