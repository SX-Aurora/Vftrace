#include "mpi_util_types.h"

//#include <stdbool.h>
//#include <stdio.h>
//
//#include "vftr_functions.h"
//#include "vftr_stacks.h"
//#include "vftr_environment.h"
//#include "vftr_setup.h"
//#include "vftr_hwcounters.h"
//#include "vftr_filewrite.h"
//#include "vftr_pause.h"
//#include "vftr_mpi_utils.h"
//#include "pcontrol.h"

// set of all C-MPI defined datatype including the derived type
#define NVFTR_TYPES 38

#ifdef _MPI
   #include <mpi.h>
#endif

// define all mpi types if mpi is not used to allow
// for the usage of types in programs like tracedump
#ifndef _MPI
   #define MPI_CHAR                  1
   #define MPI_SHORT                 1
   #define MPI_INT                   1
   #define MPI_LONG                  1
   #define MPI_LONG_LONG_INT         1
   #define MPI_LONG_LONG             1
   #define MPI_SIGNED_CHAR           1
   #define MPI_UNSIGNED_CHAR         1
   #define MPI_UNSIGNED_SHORT        1
   #define MPI_UNSIGNED              1
   #define MPI_UNSIGNED_LONG         1
   #define MPI_UNSIGNED_LONG_LONG    1
   #define MPI_FLOAT                 1
   #define MPI_DOUBLE                1
   #define MPI_LONG_DOUBLE           1
   #define MPI_WCHAR                 1
   #define MPI_C_BOOL                1
   #define MPI_INT8_T                1
   #define MPI_INT16_T               1
   #define MPI_INT32_T               1
   #define MPI_INT64_T               1
   #define MPI_UINT8_T               1
   #define MPI_UINT16_T              1
   #define MPI_UINT32_T              1
   #define MPI_UINT64_T              1
   #define MPI_C_COMPLEX             1
   #define MPI_C_FLOAT_COMPLEX       1
   #define MPI_C_DOUBLE_COMPLEX      1
   #define MPI_C_LONG_DOUBLE_COMPLEX 1
   #define MPI_INTEGER               1
   #define MPI_LOGICAL               1
   #define MPI_REAL                  1
   #define MPI_DOUBLE_PRECISION      1
   #define MPI_COMPLEX               1
   #define MPI_CHARACTER             1
   #define MPI_BYTE                  1
   #define MPI_PACKED                1
#endif

struct vftr_mpi_type_t all_mpi_types[NVFTR_TYPES] = {
   {.mpi_type = MPI_CHAR,                  .name = "MPI_CHAR" },
   {.mpi_type = MPI_SHORT,                 .name = "MPI_SHORT" },
   {.mpi_type = MPI_INT,                   .name = "MPI_INT" },
   {.mpi_type = MPI_LONG,                  .name = "MPI_LONG" },
   {.mpi_type = MPI_LONG_LONG_INT,         .name = "MPI_LONG_LONG_INT" },
   {.mpi_type = MPI_LONG_LONG,             .name = "MPI_LONG_LONG" },
   {.mpi_type = MPI_SIGNED_CHAR,           .name = "MPI_SIGNED_CHAR" },
   {.mpi_type = MPI_UNSIGNED_CHAR,         .name = "MPI_UNSIGNED_CHAR" },
   {.mpi_type = MPI_UNSIGNED_SHORT,        .name = "MPI_UNSIGNED_SHORT" },
   {.mpi_type = MPI_UNSIGNED,              .name = "MPI_UNSIGNED" },
   {.mpi_type = MPI_UNSIGNED_LONG,         .name = "MPI_UNSIGNED_LONG" },
   {.mpi_type = MPI_UNSIGNED_LONG_LONG,    .name = "MPI_UNSIGNED_LONG_LONG" },
   {.mpi_type = MPI_FLOAT,                 .name = "MPI_FLOAT" },
   {.mpi_type = MPI_DOUBLE,                .name = "MPI_DOUBLE" },
   {.mpi_type = MPI_LONG_DOUBLE,           .name = "MPI_LONG_DOUBLE" },
   {.mpi_type = MPI_WCHAR,                 .name = "MPI_WCHAR" },
   {.mpi_type = MPI_C_BOOL,                .name = "MPI_C_BOOL" },
   {.mpi_type = MPI_INT8_T,                .name = "MPI_INT8_T" },
   {.mpi_type = MPI_INT16_T,               .name = "MPI_INT16_T" },
   {.mpi_type = MPI_INT32_T,               .name = "MPI_INT32_T" },
   {.mpi_type = MPI_INT64_T,               .name = "MPI_INT64_T" },
   {.mpi_type = MPI_UINT8_T,               .name = "MPI_UINT8_T" },
   {.mpi_type = MPI_UINT16_T,              .name = "MPI_UINT16_T" },
   {.mpi_type = MPI_UINT32_T,              .name = "MPI_UINT32_T" },
   {.mpi_type = MPI_UINT64_T,              .name = "MPI_UINT64_T" },
   {.mpi_type = MPI_C_COMPLEX,             .name = "MPI_C_COMPLEX" },
   {.mpi_type = MPI_C_FLOAT_COMPLEX,       .name = "MPI_C_FLOAT_COMPLEX" },
   {.mpi_type = MPI_C_DOUBLE_COMPLEX,      .name = "MPI_C_DOUBLE_COMPLEX" },
   {.mpi_type = MPI_C_LONG_DOUBLE_COMPLEX, .name = "MPI_C_LONG_DOUBLE_COMPLEX" },
   {.mpi_type = MPI_INTEGER,               .name = "MPI_INTEGER" },
   {.mpi_type = MPI_LOGICAL,               .name = "MPI_LOGICAL" },
   {.mpi_type = MPI_REAL,                  .name = "MPI_REAL" },
   {.mpi_type = MPI_DOUBLE_PRECISION,      .name = "MPI_DOUBLE_PRECISION" },
   {.mpi_type = MPI_COMPLEX,               .name = "MPI_COMPLEX" },
   {.mpi_type = MPI_CHARACTER,             .name = "MPI_CHARACTER" },
   {.mpi_type = MPI_BYTE,                  .name = "MPI_BYTE" },
   {.mpi_type = MPI_PACKED,                .name = "MPI_PACKED" }
};

#ifdef _MPI
// Translates an MPI-Datatype into the vftr type index
int vftr_get_mpitype_idx(MPI_Datatype mpi_type) {
   for (int i=0; i<NVFTR_TYPES; i++) {
      if (mpi_type == all_mpi_types[i].mpi_type) {
         return i;
      }
   }
   // if no datatype could be found return the derived datatype idx
   return -1;
}

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string(MPI_Datatype mpi_type) {
   int idx = vftr_get_mpitype_idx(mpi_type);
   if (idx >= 0 && idx < NVFTR_TYPES) {
      // search for MPI_Type string was successful
      return all_mpi_types[idx].name;
   } else if (idx == -1) {
      // search failed.
      // MPI_Type can not be a standard type
      return "MPI_DERIVED_TYPE";
   } else {
      return "MPI_UNDEFINED_TYPE";
   }
}
#endif

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string_from_idx(int mpi_type_idx) {
   if (mpi_type_idx >= 0 && mpi_type_idx < NVFTR_TYPES) {
      return all_mpi_types[mpi_type_idx].name;
   } else if (mpi_type_idx == -1) {
      return "MPI_DERIVED_TYPE";
   }
   return "MPI_UNDEFINED_TYPE";
}
