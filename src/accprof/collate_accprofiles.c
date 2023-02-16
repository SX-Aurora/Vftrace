#include <string.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include "collated_stack_types.h"
#include "accprofiling_types.h"

void vftr_collate_accprofiles_root_self (collated_stacktree_t *collstacktree_ptr,
                                          stacktree_t *stacktree_ptr) {
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
      int i_collstack = stack->gid;
      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;

      accprofile_t copy_accprof = stack->profiling.profiles[0].accprof;
      collated_accprofile_t *collaccprof = &(collstack->profile.accprof);
      
      collaccprof->region_id = copy_accprof.region_id;
      collaccprof->event_type = copy_accprof.event_type;
      collaccprof->copied_bytes = copy_accprof.copied_bytes;
      collaccprof->overhead_nsec = copy_accprof.overhead_nsec;

   }
}

#ifdef _MPI
static void vftr_collate_accprofiles_on_root (collated_stacktree_t *collstacktree_ptr,
                                              stacktree_t *stacktree_ptr,
                                              int myrank, int nranks, int *nremote_profiles) {
   typedef struct {
     int gid;
     int event_type;
     int line_start;
     int line_end;
     int len_source_file;
     int len_func_name;
     int len_var_name;
     int len_kernel_name;
     long long region_id;
     long long copied_bytes;
     long long overhead_nsec;
     char *source_file;
     char *func_name;
     char *var_name;
     char *kernel_name;
   } accprofile_transfer_t;

   int max_profiles = 0;
   for (int irank = 0; irank < nranks; irank++) {
      max_profiles = nremote_profiles[irank] > max_profiles ? nremote_profiles[irank] : max_profiles;
   } 

   int max_len_source_file = 0;
   int max_len_func_name = 0; 
   int max_len_var_name = 0;
   int max_len_kernel_name = 0;

   int *len_source_files = (int*)malloc(nranks * sizeof(int)); 
   int *len_func_names = (int*)malloc(nranks * sizeof(int)); 
   int *len_var_names = (int*)malloc(nranks * sizeof(int)); 
   int *len_kernel_names = (int*)malloc(nranks * sizeof(int)); 
   memset (len_source_files, 0, sizeof(int) * nranks);
   memset (len_func_names, 0, sizeof(int) * nranks);
   memset (len_var_names, 0, sizeof(int) * nranks);
   memset (len_kernel_names, 0, sizeof(int) * nranks);

   for (int i = 0; i < max_profiles; i++) {
      int len_sf, len_fn, len_vn, len_kn;
      if (i >= nremote_profiles[i]) {
         len_sf = 0;
         len_fn = 0;
         len_vn = 0;
         len_kn = 0;
      } else {
         vftr_stack_t *mystack = stacktree_ptr->stacks + i;
         accprofile_t accprof = mystack->profiling.profiles->accprof;
         len_sf = accprof.source_file != NULL ? strlen(accprof.source_file) : 0;
         len_fn = accprof.func_name != NULL ? strlen(accprof.func_name): 0;
         len_vn = accprof.var_name != NULL ? strlen(accprof.var_name): 0;
         len_kn = accprof.kernel_name != NULL ? strlen(accprof.kernel_name): 0;
      }
      PMPI_Gather (&len_sf, 1, MPI_INT, len_source_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
      PMPI_Gather (&len_fn, 1, MPI_INT, len_func_names, 1, MPI_INT, 0, MPI_COMM_WORLD);
      PMPI_Gather (&len_vn, 1, MPI_INT, len_var_names, 1, MPI_INT, 0, MPI_COMM_WORLD);
      PMPI_Gather (&len_kn, 1, MPI_INT, len_kernel_names, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (myrank == 0) {
        for (int irank = 0; irank < nranks; irank++) {
           max_len_source_file = len_source_files[irank] > max_len_source_file ? len_source_files[irank] : max_len_source_file;
           max_len_func_name = len_func_names[irank] > max_len_func_name ? len_func_names[irank] : max_len_func_name;
           max_len_var_name = len_var_names[irank] > max_len_var_name ? len_var_names[irank] : max_len_var_name;
           max_len_kernel_name = len_kernel_names[irank] > max_len_kernel_name ? len_kernel_names[irank] : max_len_kernel_name;
        }
        for (int irank = 0; irank < nranks; irank++) {
           PMPI_Send (&max_len_source_file, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
           PMPI_Send (&max_len_func_name, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
           PMPI_Send (&max_len_var_name, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
           PMPI_Send (&max_len_kernel_name, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
        }
      } else {
         MPI_Status status;
         PMPI_Recv (&max_len_source_file, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
         PMPI_Recv (&max_len_func_name, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
         PMPI_Recv (&max_len_var_name, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
         PMPI_Recv (&max_len_kernel_name, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      }
   }

   int nblocks = 6;
   int *blocklengths = (int*)malloc(nblocks * sizeof(int));
   blocklengths[0] = 8;
   blocklengths[1] = 3;
   blocklengths[2] = max_len_source_file;
   blocklengths[3] = max_len_func_name;
   blocklengths[4] = max_len_var_name;
   blocklengths[5] = max_len_kernel_name;
   //const int blocklengths[] = {8,3,4};
   //const MPI_Aint displacements[] = {MPI_INT, MPI_LONG_LONG_INT,
   //                                  MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR};
   MPI_Aint *displacements = (MPI_Aint*)malloc(nblocks * sizeof(MPI_Aint));
   displacements[0] = 0;
   displacements[1] = 8 * sizeof(int);
   displacements[2] = displacements[1] + 3 * sizeof(long long);
   displacements[3] = displacements[2] + max_len_source_file * sizeof(char);
   displacements[4] = displacements[3] + max_len_func_name * sizeof(char);
   displacements[5] = displacements[4] + max_len_var_name * sizeof(char);
   displacements[6] = displacements[5] + max_len_kernel_name * sizeof(char);

   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT,
                                 MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR};
   MPI_Datatype accprofile_transfer_mpi_t;
   PMPI_Type_create_struct (nblocks, blocklengths, displacements, types,
                            &accprofile_transfer_mpi_t);
   PMPI_Type_commit (&accprofile_transfer_mpi_t);

   if (myrank > 0) {
      int nprofiles = stacktree_ptr->nstacks;
      accprofile_transfer_t *sendbuf = (accprofile_transfer_t*)malloc(nprofiles * sizeof(accprofile_transfer_t));
      memset (sendbuf, 0, nprofiles * sizeof(accprofile_transfer_t));
      for (int istack = 0; istack < nprofiles; istack++) {
         vftr_stack_t *mystack = stacktree_ptr->stacks + istack;
         accprofile_t accprof = mystack->profiling.profiles->accprof;
         sendbuf[istack].gid = mystack->gid;
         sendbuf[istack].event_type = accprof.event_type;
         sendbuf[istack].line_start = accprof.line_start;
         sendbuf[istack].line_end = accprof.line_end;
         sendbuf[istack].copied_bytes = accprof.copied_bytes;
         sendbuf[istack].overhead_nsec = accprof.overhead_nsec;  
      }
      PMPI_Send (sendbuf, nprofiles, accprofile_transfer_mpi_t, 0, myrank, MPI_COMM_WORLD);
      free(sendbuf);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_profiles[irank] > maxprofiles ? nremote_profiles[irank] : maxprofiles;
      } 

      accprofile_transfer_t *recvbuf = (accprofile_transfer_t*)malloc(maxprofiles * sizeof(accprofile_transfer_t));
      memset (recvbuf, 0, maxprofiles * sizeof(accprofile_transfer_t));
      
      for (int irank = 1; irank < nranks; irank++) {
         int nprofiles = nremote_profiles[irank];
         MPI_Status status;
         PMPI_Recv (recvbuf, nprofiles, accprofile_transfer_mpi_t, irank, irank, MPI_COMM_WORLD, &status);
         for (int iprof = 0; iprof < nprofiles; iprof++) {
            int gid = recvbuf[iprof].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
            collated_accprofile_t *collaccprof = &(collstack->profile.accprof);
         
            collaccprof->event_type = recvbuf[iprof].event_type;
            collaccprof->copied_bytes += recvbuf[iprof].copied_bytes;
            collaccprof->overhead_nsec += recvbuf[iprof].overhead_nsec;
         }
      }
      free(recvbuf);
   }
   PMPI_Type_free(&accprofile_transfer_mpi_t);
}
#endif
 

// Currently, OpenACC profiling is only supported for one MPI process
// and one OMP thread. Therefore, collating the profiles just comes
// down to copying the profile from the one stack which exists.
//void vftr_collate_accprofiles (collated_stacktree_t *collstacktree_ptr,
//                               stacktree_t *stacktree_ptr,
//                               int myrank, int nranks, int *nremote_profiles) {
//   (void)myrank;
//   (void)nranks;
//   (void)nremote_profiles;
//
//   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
//      vftr_stack_t *stack = stacktree_ptr->stacks + istack;
//      int i_collstack = stack->gid;
//      collated_stack_t *collstack = collstacktree_ptr->stacks + i_collstack;
//      // OpenACC is only supported for one thread (i_prof = 0).
//      accprofile_t copy_accprof = stack->profiling.profiles[0].accprof;
//      accprofile_t *collaccprof = &(collstack->profile.accprof);
//      
//      collaccprof->event_type = copy_accprof.event_type;
//      collaccprof->line_start = copy_accprof.line_start;
//      collaccprof->line_end = copy_accprof.line_end;
//      collaccprof->copied_bytes = copy_accprof.copied_bytes;
//      collaccprof->source_file = copy_accprof.source_file;
//      collaccprof->var_name = copy_accprof.var_name;
//      collaccprof->func_name = copy_accprof.func_name;
//      collaccprof->kernel_name = copy_accprof.kernel_name;
//      collaccprof->overhead_nsec = copy_accprof.overhead_nsec;
//      collaccprof->region_id = copy_accprof.region_id;
//   }
//}

void vftr_collate_accprofiles (collated_stacktree_t *collstacktree_ptr,
                               stacktree_t *stacktree_ptr,
                               int myrank, int nranks, int *nremote_profiles) {
  vftr_collate_accprofiles_root_self (collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
  int mpi_initialized;
  PMPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    vftr_collate_accprofiles_on_root (collstacktree_ptr, stacktree_ptr, myrank, nranks, nremote_profiles);
  }
#endif
}
