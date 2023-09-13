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

      accprofile_t local_accprof = stack->profiling.profiles[0].accprof;
      collated_accprofile_t *collaccprof = &(collstack->profile.accprof);
      
      collaccprof->region_id = local_accprof.region_id;
      collaccprof->event_type = local_accprof.event_type;
      collaccprof->copied_bytes = local_accprof.copied_bytes;
      collaccprof->source_file = local_accprof.source_file != NULL ?
                                 strdup(local_accprof.source_file) : NULL;
      collaccprof->func_name = local_accprof.func_name != NULL ?
                                 strdup(local_accprof.func_name) : NULL;
      collaccprof->var_name = local_accprof.var_name != NULL ?
                                 strdup(local_accprof.var_name) : NULL;
      collaccprof->kernel_name = local_accprof.kernel_name != NULL ?
                                 strdup(local_accprof.kernel_name) : NULL;
      collaccprof->overhead_nsec = local_accprof.overhead_nsec;

      collaccprof->on_nranks = 1;
      collated_callprofile_t callprof = collstack->profile.callprof;
      if (collaccprof->event_type == acc_ev_enqueue_upload_start ||
          collaccprof->event_type == acc_ev_enqueue_upload_end) {
        collaccprof->ncalls[0] = callprof.calls; 
        collaccprof->ncalls[1] = 0;
      } else if (collaccprof->event_type == acc_ev_enqueue_download_start ||
                 collaccprof->event_type == acc_ev_enqueue_download_end) {
        collaccprof->ncalls[0] = 0;
        collaccprof->ncalls[1] = callprof.calls; 
      } else {
        collaccprof->ncalls[0] = 0;
        collaccprof->ncalls[1] = 0;
      }

      for (int dir = 0; dir < 2; dir++) {
        collaccprof->max_on_rank[dir] = 0;
        collaccprof->min_on_rank[dir] = 0;
        collaccprof->avg_ncalls[dir] = collaccprof->ncalls[dir];
        collaccprof->min_ncalls[dir] = collaccprof->ncalls[dir];
        collaccprof->max_ncalls[dir] = collaccprof->ncalls[dir];
      }
   }
}

#ifdef _MPI
static void vftr_collate_accprofiles_on_root (collated_stacktree_t *collstacktree_ptr,
                                              stacktree_t *stacktree_ptr,
                                              int myrank, int nranks, int *nremote_stacks) {
#define TRANSFER_BUFSIZE 1024
   typedef struct {
     int gid;
     int event_type;
     int line_start;
     int line_end;
     int len_source_file;
     int len_func_name;
     int len_var_name;
     int len_kernel_name;
     int n_calls[2];
     long long region_id;
     long long copied_bytes;
     long long overhead_nsec;
     char source_file[TRANSFER_BUFSIZE];
     char func_name[TRANSFER_BUFSIZE];
     char var_name[TRANSFER_BUFSIZE];
     char kernel_name[TRANSFER_BUFSIZE];
   } accprofile_transfer_t;

   int max_profiles = 0;
   MPI_Status status;
   if (myrank == 0) {
      for (int irank = 0; irank < nranks; irank++) {
         max_profiles = nremote_stacks[irank] > max_profiles ? nremote_stacks[irank] : max_profiles;
      }
      for (int irank = 1; irank < nranks; irank++) {
         PMPI_Send (&max_profiles, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
      }
   } else {
      PMPI_Recv (&max_profiles, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
   }

   int nblocks = 3;
   int *blocklengths = (int*)malloc(nblocks * sizeof(int));
   blocklengths[0] = 10;
   blocklengths[1] = 3;
   blocklengths[2] = 4 * TRANSFER_BUFSIZE;

   MPI_Aint *displacements = (MPI_Aint*)malloc(nblocks * sizeof(MPI_Aint));
   displacements[0] = 0;
   displacements[1] = 10 * sizeof(int);
   displacements[2] = displacements[1] + 3 * sizeof(long long);

   const MPI_Datatype types[] = {MPI_INT, MPI_LONG_LONG_INT, MPI_CHAR};
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
         accprofile_t local_accprof = mystack->profiling.profiles[0].accprof;
         sendbuf[istack].gid = mystack->gid;
         sendbuf[istack].region_id = local_accprof.region_id;
         sendbuf[istack].event_type = local_accprof.event_type;
         sendbuf[istack].line_start = local_accprof.line_start;
         sendbuf[istack].line_end = local_accprof.line_end;
         sendbuf[istack].copied_bytes = local_accprof.copied_bytes;
         sendbuf[istack].overhead_nsec = local_accprof.overhead_nsec;  
         if (local_accprof.event_type == acc_ev_enqueue_upload_start ||
             local_accprof.event_type == acc_ev_enqueue_upload_end) {
            sendbuf[istack].n_calls[0] = mystack->profiling.profiles[0].callprof.calls;
            sendbuf[istack].n_calls[1] = 0;
         } else if (local_accprof.event_type == acc_ev_enqueue_download_start ||
                    local_accprof.event_type == acc_ev_enqueue_download_end) {
            sendbuf[istack].n_calls[0] = 0;
            sendbuf[istack].n_calls[1] = mystack->profiling.profiles[0].callprof.calls;
         } else {
            sendbuf[istack].n_calls[0] = 0;
            sendbuf[istack].n_calls[1] = 0;
         }
         if (local_accprof.source_file != NULL) {
           sendbuf[istack].len_source_file = strlen(local_accprof.source_file);
           strncpy (sendbuf[istack].source_file, local_accprof.source_file,
                    sendbuf[istack].len_source_file);
         } else {
           sendbuf[istack].len_source_file = 1;
           strncpy (sendbuf[istack].source_file, "", 1);
         }
         if (local_accprof.func_name != NULL) {
           sendbuf[istack].len_func_name = strlen(local_accprof.func_name);
           strncpy (sendbuf[istack].func_name, local_accprof.func_name,
                    sendbuf[istack].len_func_name);
         } else {
           sendbuf[istack].len_func_name = 1;
           strncpy (sendbuf[istack].func_name, "", 1);
         }
         if (local_accprof.var_name != NULL) {
           sendbuf[istack].len_var_name = strlen(local_accprof.var_name);
           strncpy (sendbuf[istack].var_name, local_accprof.var_name,
                    sendbuf[istack].len_var_name);
         } else {
           sendbuf[istack].len_var_name = 1;
           strncpy (sendbuf[istack].var_name, "", 1);
         }
         if (local_accprof.kernel_name != NULL) {
            sendbuf[istack].len_kernel_name = strlen(local_accprof.kernel_name);
            strncpy (sendbuf[istack].kernel_name, local_accprof.kernel_name,
                     sendbuf[istack].len_kernel_name);
         } else {
            sendbuf[istack].len_kernel_name = 1;
            strncpy (sendbuf[istack].kernel_name, "", 1);
         }
      }
      PMPI_Send (sendbuf, nprofiles, accprofile_transfer_mpi_t, 0, myrank, MPI_COMM_WORLD);
      free(sendbuf);
   } else {
      int maxprofiles = 0;
      for (int irank = 1; irank < nranks; irank++) {
         maxprofiles = nremote_stacks[irank] > maxprofiles ? nremote_stacks[irank] : maxprofiles;
      } 

      accprofile_transfer_t *recvbuf = (accprofile_transfer_t*)malloc(maxprofiles * sizeof(accprofile_transfer_t));
      memset (recvbuf, 0, maxprofiles * sizeof(accprofile_transfer_t));
      
      for (int irank = 1; irank < nranks; irank++) {
         int nprofiles = nremote_stacks[irank];
         MPI_Status status;
         PMPI_Recv (recvbuf, nprofiles, accprofile_transfer_mpi_t, irank, irank, MPI_COMM_WORLD, &status);
         for (int iprof = 0; iprof < nprofiles; iprof++) {
            int gid = recvbuf[iprof].gid;
            collated_stack_t *collstack = collstacktree_ptr->stacks + gid;
            collated_accprofile_t *collaccprof = &(collstack->profile.accprof);
         
            collaccprof->region_id = recvbuf[iprof].region_id;
            collaccprof->event_type = recvbuf[iprof].event_type;
            collaccprof->line_start = recvbuf[iprof].line_start;
            collaccprof->line_end = recvbuf[iprof].line_end;
            collaccprof->copied_bytes += recvbuf[iprof].copied_bytes;
            collaccprof->overhead_nsec += recvbuf[iprof].overhead_nsec;
            if (recvbuf[iprof].len_source_file > 1)
               collaccprof->source_file = strdup(recvbuf[iprof].source_file);
            if (recvbuf[iprof].len_func_name > 1)
               collaccprof->func_name = strdup(recvbuf[iprof].func_name);
            if (recvbuf[iprof].len_var_name > 1)
               collaccprof->var_name = strdup(recvbuf[iprof].var_name);
            if (recvbuf[iprof].len_kernel_name > 1)
               collaccprof->kernel_name = strdup(recvbuf[iprof].kernel_name);

            if (recvbuf[iprof].n_calls[0] > 0 || recvbuf[iprof].n_calls[1] > 0) {
               collaccprof->on_nranks++;
               for (int dir = 0; dir < 2; dir++) {
                  int n_calls = recvbuf[iprof].n_calls[dir];
                  if (n_calls > 0) {
                     collaccprof->avg_ncalls[dir] += n_calls;
                     if (n_calls > collaccprof->max_ncalls[dir]) {
                        collaccprof->max_on_rank[dir] = irank;
                        collaccprof->max_ncalls[dir] = n_calls;
                     } else if (n_calls < collaccprof->min_ncalls[dir]) {
                        collaccprof->min_on_rank[dir] = irank;
                        collaccprof->min_ncalls[dir] = n_calls;
                     }
                  }
               }
            }
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
//                               int myrank, int nranks, int *nremote_stacks) {
//   (void)myrank;
//   (void)nranks;
//   (void)nremote_stacks;
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
                               int myrank, int nranks, int *nremote_stacks) {
  vftr_collate_accprofiles_root_self (collstacktree_ptr, stacktree_ptr);
#ifdef _MPI
  int mpi_initialized;
  PMPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    vftr_collate_accprofiles_on_root (collstacktree_ptr, stacktree_ptr, myrank, nranks, nremote_stacks);
  }
#endif
}
