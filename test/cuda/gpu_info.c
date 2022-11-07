#include <stdio.h>

#include <cuda_runtime_api.h>

int main (int argc, char **argv) {
   int n;
   cudaGetDeviceCount (&n);
   printf ("%d\n", cudaGetLastError() == cudaSuccess);
   return 0;
}
