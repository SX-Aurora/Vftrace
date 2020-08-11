#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

int main(int argc, char** argv) {

   MPI_Init(NULL, NULL);

   // Get number or processes
   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
   // Get rank of process
   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   // require at least two processes
   if (comm_size < 2) {
      printf("At least two ranks are required\n");
      printf("Run again with '-np 2'\n");
      MPI_Finalize();
      return 1;
   }

   // Messaging 
   if (my_rank == 0) {
      // sending rank
      char send_char;
      MPI_Send(&send_char, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
      short send_short;
      MPI_Send(&send_short, 1, MPI_SHORT, 1, 0, MPI_COMM_WORLD);
      int send_int;
      MPI_Send(&send_int, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
      long send_long;
      MPI_Send(&send_long, 1, MPI_LONG, 1, 0, MPI_COMM_WORLD);
      long long send_long_long;
      MPI_Send(&send_long_long, 1, MPI_LONG_LONG_INT, 1, 0, MPI_COMM_WORLD);

      unsigned char send_uchar;
      MPI_Send(&send_uchar, 1, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD);
      unsigned short send_ushort;
      MPI_Send(&send_ushort, 1, MPI_UNSIGNED_SHORT, 1, 0, MPI_COMM_WORLD);
      unsigned int send_uint;
      MPI_Send(&send_uint, 1, MPI_UNSIGNED, 1, 0, MPI_COMM_WORLD);
      unsigned long send_ulong;
      MPI_Send(&send_ulong, 1, MPI_UNSIGNED_LONG, 1, 0, MPI_COMM_WORLD);
      unsigned long long send_ulong_long;
      MPI_Send(&send_ulong_long, 1, MPI_UNSIGNED_LONG_LONG, 1, 0, MPI_COMM_WORLD);

      float send_float;
      MPI_Send(&send_float, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
      double send_double;
      MPI_Send(&send_double, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      long double send_long_double;
      MPI_Send(&send_long_double, 1, MPI_LONG_DOUBLE, 1, 0, MPI_COMM_WORLD);
      wchar_t send_wchar;
      MPI_Send(&send_wchar, 1, MPI_WCHAR, 1, 0, MPI_COMM_WORLD);

      _Bool send_bool;
      MPI_Send(&send_bool, 1, MPI_C_BOOL, 1, 0, MPI_COMM_WORLD);
      int8_t send_int8;
      MPI_Send(&send_int8, 1, MPI_INT8_T, 1, 0, MPI_COMM_WORLD);
      int16_t send_int16;
      MPI_Send(&send_int16, 1, MPI_INT16_T, 1, 0, MPI_COMM_WORLD);
      int32_t send_int32;
      MPI_Send(&send_int32, 1, MPI_INT32_T, 1, 0, MPI_COMM_WORLD);
      int64_t send_int64;
      MPI_Send(&send_int64, 1, MPI_INT64_T, 1, 0, MPI_COMM_WORLD);
      uint8_t send_uint8;
      MPI_Send(&send_uint8, 1, MPI_UINT8_T, 1, 0, MPI_COMM_WORLD);
      uint16_t send_uint16;
      MPI_Send(&send_uint16, 1, MPI_UINT16_T, 1, 0, MPI_COMM_WORLD);
      uint32_t send_uint32;
      MPI_Send(&send_uint32, 1, MPI_UINT32_T, 1, 0, MPI_COMM_WORLD);
      uint64_t send_uint64;
      MPI_Send(&send_uint64, 1, MPI_UINT64_T, 1, 0, MPI_COMM_WORLD);

      float _Complex send_complex;
      MPI_Send(&send_complex, 1, MPI_C_COMPLEX, 1, 0, MPI_COMM_WORLD);
      double _Complex send_double_complex;
      MPI_Send(&send_double_complex, 1, MPI_C_DOUBLE_COMPLEX, 1, 0, MPI_COMM_WORLD);
      long double _Complex send_long_double_complex;
      MPI_Send(&send_long_double_complex, 1, MPI_C_LONG_DOUBLE_COMPLEX, 1, 0, MPI_COMM_WORLD);

      char send_byte;
      MPI_Send(&send_byte, 1, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
   } else if (my_rank == 1) {
      // receiving rank
      MPI_Status recvstatus;

      char recv_char;
      MPI_Recv(&recv_char, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &recvstatus);
      short recv_short;
      MPI_Recv(&recv_short, 1, MPI_SHORT, 0, 0, MPI_COMM_WORLD, &recvstatus);
      int recv_int;
      MPI_Recv(&recv_int, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &recvstatus);
      long recv_long;
      MPI_Recv(&recv_long, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, &recvstatus);
      long long recv_long_long;
      MPI_Recv(&recv_long_long, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD, &recvstatus);

      unsigned char recv_uchar;
      MPI_Recv(&recv_uchar, 1, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &recvstatus);
      unsigned short recv_ushort;
      MPI_Recv(&recv_ushort, 1, MPI_UNSIGNED_SHORT, 0, 0, MPI_COMM_WORLD, &recvstatus);
      unsigned int recv_uint;
      MPI_Recv(&recv_uint, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &recvstatus);
      unsigned long recv_ulong;
      MPI_Recv(&recv_ulong, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &recvstatus);
      unsigned long long recv_ulong_long;
      MPI_Recv(&recv_ulong_long, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD, &recvstatus);

      float recv_float;
      MPI_Recv(&recv_float, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &recvstatus);
      double recv_double;
      MPI_Recv(&recv_double, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recvstatus);
      long double recv_long_double;
      MPI_Recv(&recv_long_double, 1, MPI_LONG_DOUBLE, 0, 0, MPI_COMM_WORLD, &recvstatus);
      wchar_t recv_wchar;
      MPI_Recv(&recv_wchar, 1, MPI_WCHAR, 0, 0, MPI_COMM_WORLD, &recvstatus);

      _Bool recv_bool;
      MPI_Recv(&recv_bool, 1, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD, &recvstatus);
      int8_t recv_int8;
      MPI_Recv(&recv_int8, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD, &recvstatus);
      int16_t recv_int16;
      MPI_Recv(&recv_int16, 1, MPI_INT16_T, 0, 0, MPI_COMM_WORLD, &recvstatus);
      int32_t recv_int32;
      MPI_Recv(&recv_int32, 1, MPI_INT32_T, 0, 0, MPI_COMM_WORLD, &recvstatus);
      int64_t recv_int64;
      MPI_Recv(&recv_int64, 1, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, &recvstatus);
      uint8_t recv_uint8;
      MPI_Recv(&recv_uint8, 1, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, &recvstatus);
      uint16_t recv_uint16;
      MPI_Recv(&recv_uint16, 1, MPI_UINT16_T, 0, 0, MPI_COMM_WORLD, &recvstatus);
      uint32_t recv_uint32;
      MPI_Recv(&recv_uint32, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, &recvstatus);
      uint64_t recv_uint64;
      MPI_Recv(&recv_uint64, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &recvstatus);

      float _Complex recv_complex;
      MPI_Recv(&recv_complex, 1, MPI_C_COMPLEX, 0, 0, MPI_COMM_WORLD, &recvstatus);
      double _Complex recv_double_complex;
      MPI_Recv(&recv_double_complex, 1, MPI_C_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD, &recvstatus);
      long double _Complex recv_long_double_complex;
      MPI_Recv(&recv_long_double_complex, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD, &recvstatus);

      char recv_byte;
      MPI_Recv(&recv_byte, 1, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &recvstatus);
   }
   MPI_Finalize();

   return 0;
}
