#include "invoke.h"

#include "cstdio"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void fdtd_step_Ez (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                              FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                              FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy, int t)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  int index1 = i * sy + j;
  int index2 = (i - 1) * sy + j;
  int index3 = i * sy + j - 1;

  /*printf ("Cuda block #(x=%d,y=%d) of size #(%d,%d), thread #(x=%d, y=%d) = %d %d. Index = %d\n",
    blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, i, j, index1);*/

  if (i < 0 || j < 0 || i >= sx || j >= sy)
  {
    return;
  }

  if (i == 0 || j == 0)
  {
    Ez[index1] = Ez_prev[index1];
    return;
  }

  Ez[index1] = Ez_prev[index1] + (gridTimeStep / (0.0000000000088541878176203892 * gridStep)) *
    (Hx_prev[index3] - Hx_prev[index1] + Hy_prev[index1] - Hy_prev[index2]);

  if (i == sx / 2 && j == sy / 2)
  {
    Ez[index1] = cos (t * 3.1415 / 12);
  }

  /*printf ("Cuda block #(x=%d,y=%d) of size #(%d,%d), thread #(x=%d, y=%d) = %d %d %d %d. Val = %f\n",
    blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, i, j, sx, sy, Hx_prev[index1]);*/

  Ez_prev[index1] = Ez[index1];
}

__global__ void fdtd_step_Hx (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                              FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                              FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  int index1 = i * sy + j;
  int index2 = i * sy + j + 1;

  if (i < 0 || j < 0 || i >= sx || j >= sy)
  {
    return;
  }

  if (i == 0 || j == sy - 1)
  {
    Hx[index1] = Hx_prev[index1];
    return;
  }

  Hx[index1] = Hx_prev[index1] + (gridTimeStep / (0.0000012566370614359173 * gridStep)) *
    (Ez_prev[index1] - Ez_prev[index2]);

  Hx_prev[index1] = Hx[index1];
}

__global__ void fdtd_step_Hy (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                              FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                              FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  int index1 = i * sy + j;
  int index2 = (i + 1) * sy + j;

  if (i < 0 || j < 0 || i >= sx || j >= sy)
  {
    return;
  }

  if (i == sx - 1 || j == 0)
  {
    Hy[index1] = Hy_prev[index1];
    return;
  }

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

  int size = sx * sy * sizeof (FieldValue);

  cudaMalloc ((void **) &Ez_cuda, size);
  cudaCheckErrors ("1");
  cudaMalloc ((void **) &Hx_cuda, size);
  cudaCheckErrors ("2");
  cudaMalloc ((void **) &Hy_cuda, size);
  cudaCheckErrors ("3");

  cudaMalloc ((void **) &Ez_cuda_prev, size);
  cudaCheckErrors ("4");
  cudaMalloc ((void **) &Hx_cuda_prev, size);
  cudaCheckErrors ("5");
  cudaMalloc ((void **) &Hy_cuda_prev, size);
  cudaCheckErrors ("6");

  cudaMemcpy (Ez_cuda, tmp_Ez, size, cudaMemcpyHostToDevice);
  cudaCheckErrors ("7");
  cudaMemcpy (Hx_cuda, tmp_Hx, size, cudaMemcpyHostToDevice);
  cudaCheckErrors ("8");
  cudaMemcpy (Hy_cuda, tmp_Hy, size, cudaMemcpyHostToDevice);
  cudaCheckErrors ("9");

  cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, size, cudaMemcpyHostToDevice);
  cudaCheckErrors ("10");
  cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, size, cudaMemcpyHostToDevice);
  cudaCheckErrors ("11");
  cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, size, cudaMemcpyHostToDevice);
  cudaCheckErrors ("12");

  int NN = 32;

  dim3 N (sx / NN, sy / NN);
  dim3 N1 (NN, NN);

  for (int t = 0; t < totalStep; ++t)
  {
    fdtd_step_Ez <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy, t);
    fdtd_step_Hx <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy);
    fdtd_step_Hy <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy);
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
