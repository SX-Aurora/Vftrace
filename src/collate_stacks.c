#include <stdlib.h>

#include <string.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include "realloc_consts.h"
#include "self_profile.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "collated_hash_types.h"
#include "collated_stack_types.h"

#include "hashing.h"
#include "collate_hashes.h"
#include "search.h"
#include "profiling.h"
#include "collated_profiling.h"

gid_list_t vftr_new_empty_gid_list() {
   gid_list_t gid_list;
   gid_list.ngids = 0;
   gid_list.maxgids = 0;
   gid_list.gids = NULL;
   return gid_list;
}

void vftr_gid_list_realloc(gid_list_t *gid_list_ptr) {
   gid_list_t gid_list = *gid_list_ptr;
   while (gid_list.ngids > gid_list.maxgids) {
      int maxgids = gid_list.maxgids*vftr_realloc_rate+vftr_realloc_add;
      gid_list.gids = (int*) realloc(gid_list.gids, maxgids*sizeof(int));
      gid_list.maxgids = maxgids;
   }
   *gid_list_ptr = gid_list;
}

void vftr_gid_list_free(gid_list_t *gid_list_ptr) {
   if (gid_list_ptr->ngids > 0) {
      free(gid_list_ptr->gids);
      gid_list_ptr->gids = NULL;
      gid_list_ptr->ngids = 0;
      gid_list_ptr->maxgids = 0;
   }
}

collated_stacktree_t vftr_new_empty_collated_stacktree() {
   SELF_PROFILE_START_FUNCTION;
   collated_stacktree_t stacktree;
   stacktree.nstacks = 0;
   stacktree.maxstacks = 0;
   stacktree.stacks = NULL;
   stacktree.namegrouped = false;
   SELF_PROFILE_END_FUNCTION;
   return stacktree;
}

collated_stacktree_t vftr_new_collated_stacktree(hashlist_t hashlist) {
   SELF_PROFILE_START_FUNCTION;
   // create empty collated stacktree
   collated_stacktree_t coll_stacktree;
   coll_stacktree.nstacks = hashlist.nhashes;
   coll_stacktree.maxstacks = coll_stacktree.nstacks;
   coll_stacktree.stacks = (collated_stack_t*)
      malloc(coll_stacktree.nstacks*sizeof(collated_stack_t));
   for (int istack=0; istack<coll_stacktree.nstacks; istack++) {
      coll_stacktree.stacks[istack].local_stack = NULL;
      coll_stacktree.stacks[istack].gid = istack;
      coll_stacktree.stacks[istack].gid_list = vftr_new_empty_gid_list();
      coll_stacktree.stacks[istack].precise = false;
      coll_stacktree.stacks[istack].caller = -1;
      coll_stacktree.stacks[istack].name = NULL;
      coll_stacktree.stacks[istack].hash = hashlist.hashes[istack];
      coll_stacktree.stacks[istack].profile = vftr_new_collated_profile();
   }
   coll_stacktree.namegrouped = false;
   SELF_PROFILE_END_FUNCTION;
   return coll_stacktree;
}

void vftr_collated_stacktree_realloc(collated_stacktree_t *stacktree_ptr) {
   collated_stacktree_t stacktree = *stacktree_ptr;
   while (stacktree.nstacks > stacktree.maxstacks) {
      int maxstacks = stacktree.maxstacks*vftr_realloc_rate+vftr_realloc_add;
      stacktree.stacks = (collated_stack_t*)
         realloc(stacktree.stacks, maxstacks*sizeof(collated_stack_t));
      stacktree.maxstacks = maxstacks;
   }
   *stacktree_ptr = stacktree;
}

#ifdef _MPI
void vftr_broadcast_collated_stacktree_root(collated_stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int nranks;
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   int nstacks = stacktree_ptr->nstacks;
   // broadcasting the caller ids
   int *tmpintarr = (int*) malloc((nstacks+1)*sizeof(int));
   for (int istack = 0; istack < nstacks; istack++) {
      tmpintarr[istack] = stacktree_ptr->stacks[istack].caller;
   }
   PMPI_Bcast(tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   // communicate the precise-tracing status for each function
   for (int istack = 0; istack < nstacks; istack++) {
      tmpintarr[istack] = stacktree_ptr->stacks[istack].precise ? 1 : 0;
   }
   PMPI_Bcast(tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcasting the function names.
   // first the length of each name
   int totallen = 0;
   for (int istack = 0; istack < nstacks; istack++) {
      tmpintarr[istack] = strlen(stacktree_ptr->stacks[istack].name);
      totallen += tmpintarr[istack];
      totallen += 1; // null terminator between all strings
   }
   // the total length is appended to the tmpintarr
   tmpintarr[nstacks] = totallen;
   PMPI_Bcast(tmpintarr, nstacks+1, MPI_INT, 0, MPI_COMM_WORLD);
   char *tmpchararr = (char*) malloc(totallen*sizeof(char));
   char *charptr = tmpchararr;
   for (int istack = 0; istack < nstacks; istack++) {
      strcpy(charptr, stacktree_ptr->stacks[istack].name);
      charptr += tmpintarr[istack];
      charptr++; // null terminator should be copied by strcpy
   }
   PMPI_Bcast(tmpchararr, totallen, MPI_CHAR, 0, MPI_COMM_WORLD);
  
   int n_callees_tot = 0;
   for (int istack = 0; istack < nstacks; istack++) {
      tmpintarr[istack] = stacktree_ptr->stacks[istack].ncallees;
      n_callees_tot += stacktree_ptr->stacks[istack].ncallees;
   }
   PMPI_Bcast (tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   PMPI_Bcast (&n_callees_tot, 1, MPI_INT, 0, MPI_COMM_WORLD);

   int *all_callees = (int*)malloc(n_callees_tot * sizeof(int));
   int idx = 0;
   for (int istack = 0; istack < nstacks; istack++) {
     for (int icallee = 0; icallee < stacktree_ptr->stacks[istack].ncallees; icallee++) {
        all_callees[idx++] = stacktree_ptr->stacks[istack].callees[icallee];
     }
   }
   PMPI_Bcast (all_callees, n_callees_tot, MPI_INT, 0, MPI_COMM_WORLD);

   free(tmpintarr);
   free(tmpchararr);
   free(all_callees);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_broadcast_collated_stacktree_receivers(collated_stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int nranks;
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   int nstacks = stacktree_ptr->nstacks;
   // receiving the caller ids
   int *tmpintarr = (int*) malloc((nstacks+1)*sizeof(int));
   PMPI_Bcast(tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   for (int istack = 0; istack < nstacks; istack++) {
      stacktree_ptr->stacks[istack].caller = tmpintarr[istack];
   }
   // recive the precise-tracing status for each function
   PMPI_Bcast(tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   for (int istack = 0; istack < nstacks; istack++) {
      stacktree_ptr->stacks[istack].precise = tmpintarr[istack] == 1 ? true : false;
   }
   // receiving the function names;
   // first the length of each name;
   int totallen = 0;
   PMPI_Bcast(tmpintarr, nstacks+1, MPI_INT, 0, MPI_COMM_WORLD);
   // the total length is appended to the tmpintarr
   totallen = tmpintarr[nstacks];
   char *tmpchararr = (char*) malloc(totallen*sizeof(char));
   char *charptr = tmpchararr;
   PMPI_Bcast(tmpchararr, totallen, MPI_CHAR, 0, MPI_COMM_WORLD);
   for (int istack = 0; istack < nstacks; istack++) {
      stacktree_ptr->stacks[istack].name = strdup(charptr);
      charptr += tmpintarr[istack];
      charptr++; // null terminator should be copied by strcpy
   }

   PMPI_Bcast (tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   for (int istack = 0; istack < nstacks; istack++) {
      stacktree_ptr->stacks[istack].ncallees = tmpintarr[istack]; 
   }

   int n_callees_tot;
   PMPI_Bcast (&n_callees_tot, 1, MPI_INT, 0, MPI_COMM_WORLD);
   int *all_callees = (int*)malloc(n_callees_tot * sizeof(int));
   PMPI_Bcast (all_callees, n_callees_tot, MPI_INT, 0, MPI_COMM_WORLD);
   int idx = 0;
   for (int istack = 0; istack < nstacks; istack++) {
      int ncallees = stacktree_ptr->stacks[istack].ncallees;
      if (ncallees > 0) stacktree_ptr->stacks[istack].callees = (int*)malloc(ncallees * sizeof(int));
      for (int icallee = 0; icallee < ncallees; icallee++) {
         stacktree_ptr->stacks[istack].callees[icallee] = all_callees[idx++]; 
      }
   }
   free(tmpintarr);
   free(tmpchararr);
   free(all_callees);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_broadcast_collated_stacktree(collated_stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int nranks;
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   if (nranks > 1) {
      int myrank;
      PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      if (myrank == 0) {
         vftr_broadcast_collated_stacktree_root(stacktree_ptr);
      } else {
         vftr_broadcast_collated_stacktree_receivers(stacktree_ptr);
      }
   }
   SELF_PROFILE_END_FUNCTION;
}
#endif

collated_stacktree_t vftr_collate_stacks(stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   // first compute the hashes for all stacks
   vftr_compute_stack_hashes(stacktree_ptr);

   // collate hashes between processes
   hashlist_t hashlist = vftr_collate_hashes(stacktree_ptr);

   // create empty collated stacktree
   collated_stacktree_t coll_stacktree = vftr_new_collated_stacktree(hashlist);

   // build a lookup table each to translate local2global and global2local
   int *local2global_ID = (int*) malloc(stacktree_ptr->nstacks*sizeof(int));
   int *global2local_ID = (int*) malloc(coll_stacktree.nstacks*sizeof(int));
   for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
      // -1 in the lookup table means that the local stack does not exist
      local2global_ID[istack] = -1;
   }
   for (int istack=0; istack<coll_stacktree.nstacks; istack++) {
      // -1 in the lookup table means that the local stack does not exist
      global2local_ID[istack] = -1;
   }
   // assign every function its global ID
   // the global ID is the index in the hash table
   // fill the looup tables
   for (int istack = 0; istack < stacktree_ptr->nstacks; istack++) {
      int idx = vftr_binary_search_uint64(hashlist.nhashes,
                                          hashlist.hashes,
                                          stacktree_ptr->stacks[istack].hash);
      if (idx >= 0) {
         stacktree_ptr->stacks[istack].gid = idx;
         local2global_ID[istack] = idx;
         global2local_ID[idx] = istack;
      }
   }

   // hashtable is no longer needed
   vftr_collated_hashlist_free(&hashlist);

   // fill in the locally known information
   // special treatment for init because it does not have a caller
   int myrank = 0;
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   }
#endif

   typedef struct {
      int gid;
      int return_id;
      int name_length;
      int precise;
      int ncallees;
   } missing_stack_transfer_t;
   const int blocklengths[] = {5};
   const MPI_Aint displacements[] = {0};
   const MPI_Datatype types[] = {MPI_INT};
   MPI_Datatype missing_stack_transfer_mpi_t;
   PMPI_Type_create_struct (1, blocklengths,
                           displacements, types,
                           &missing_stack_transfer_mpi_t); 
   PMPI_Type_commit (&missing_stack_transfer_mpi_t);

   if (myrank == 0) {
      {
         int lid = 0;
         vftr_stack_t *local_stack = stacktree_ptr->stacks+lid;
         int gid = local2global_ID[lid];
         collated_stack_t *global_stack = coll_stacktree.stacks+gid;
         global_stack->local_stack = local_stack;
         global_stack->gid = gid;
         global_stack->precise = local_stack->precise;
         global_stack->caller = -1;
         global_stack->ncallees = local_stack->ncallees;
         global_stack->callees = (int*)malloc(local_stack->ncallees * sizeof(int));
         for (int i = 0; i < local_stack->ncallees; i++) {
            global_stack->callees[i] = local2global_ID[local_stack->callees[i]];
         }
         global_stack->name = strdup(local_stack->cleanname);
      }
      for (int lid = 1; lid < stacktree_ptr->nstacks; lid++) {
         vftr_stack_t *local_stack = stacktree_ptr->stacks+lid;
         int gid = local2global_ID[lid];
         collated_stack_t *global_stack = coll_stacktree.stacks+gid;
         global_stack->local_stack = local_stack;
         global_stack->gid = gid;
         global_stack->precise = local_stack->precise;
         global_stack->caller = local2global_ID[local_stack->caller];
         global_stack->ncallees = local_stack->ncallees;
         global_stack->callees = (int*)malloc(local_stack->ncallees * sizeof(int));
         for (int i = 0; i < local_stack->ncallees; i++) {
            global_stack->callees[i] = local2global_ID[local_stack->callees[i]];
         }
         global_stack->name = strdup(local_stack->cleanname);
      }
   }

#ifdef _MPI
   if (mpi_initialized) {
      if (myrank == 0) {
         // if there are multiple processes the table might still be missing entries
         int *missingstacks = (int*) malloc(coll_stacktree.nstacks*sizeof(int));
         // loop over all ranks and collect the missing information
         int nranks;
         PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
         for (int irank = 1; irank < nranks; irank++) {
            int nmissing = 0;
            for (int istack = 0; istack < coll_stacktree.nstacks; istack++) {
               if (coll_stacktree.stacks[istack].name == NULL) {
                  missingstacks[nmissing] = istack;
                  nmissing++;
               }
            }

            // Send to the selected process how many entries are still missing
            PMPI_Send(&nmissing, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
            // if at least one entry is missing proceed
            if (nmissing > 0) {
               // Send the missing IDs
               PMPI_Send(missingstacks, nmissing, MPI_INT, irank, 0, MPI_COMM_WORLD);
               // Receive how many missing stacks the other process can fill in
               MPI_Status mystat;
               int hasnmissing;
               PMPI_Recv(&hasnmissing, 1, MPI_INT, irank, 0, MPI_COMM_WORLD, &mystat);
               // only proceed if the number of stacks is positive
               if (hasnmissing > 0) {
                   missing_stack_transfer_t *missing_stack_info =
                      (missing_stack_transfer_t*) malloc(hasnmissing * sizeof(missing_stack_transfer_t));
                   // Receive the found information from remote process
                   PMPI_Recv (missing_stack_info, hasnmissing,
                              missing_stack_transfer_mpi_t, irank, 0,
                              MPI_COMM_WORLD, &mystat);

                   // Create a buffer that contains all stack names in contatenated form
                   int sumlength = 0;
                   for (int istack = 0; istack < hasnmissing; istack++) {
                      sumlength += missing_stack_info[istack].name_length;
                   }
                   char *concatNames = (char*) malloc(sumlength*sizeof(char));

                   // Receive the concatenated String
                   PMPI_Recv(concatNames, sumlength, MPI_CHAR,
                             irank, 0, MPI_COMM_WORLD, &mystat);

                   int n_callees_tot = 0;
                   for (int istack = 0; istack < hasnmissing; istack++) {
                       n_callees_tot += missing_stack_info[istack].ncallees;
                   }
                   int *all_callees = (int*)malloc(n_callees_tot * sizeof(int));
                   PMPI_Recv (all_callees, n_callees_tot, MPI_INT, irank, 0,
                              MPI_COMM_WORLD, &mystat);

                   // Write all the gathered info to the global stackinfo
                   char *tmpstrptr = concatNames;
                   int idx = 0;
                   for (int istack = 0; istack < hasnmissing; istack++) {
                      int glob_id = missing_stack_info[istack].gid;
                      coll_stacktree.stacks[glob_id].caller = missing_stack_info[istack].return_id;
                      coll_stacktree.stacks[glob_id].name = strdup(tmpstrptr);
                      // next string
                      tmpstrptr += missing_stack_info[istack].name_length;
                      coll_stacktree.stacks[glob_id].precise = missing_stack_info[istack].precise != 0;
                      int ncallees = missing_stack_info[istack].ncallees;
                      coll_stacktree.stacks[glob_id].ncallees = ncallees; 
                      coll_stacktree.stacks[glob_id].callees = (int*)malloc(ncallees * sizeof(int));
                      for (int icallee = 0; icallee < ncallees; icallee++) {
                         coll_stacktree.stacks[glob_id].callees[icallee] = all_callees[idx++];
                      }
                   }

                   free(concatNames);
                   free(all_callees);
                   free(missing_stack_info);

               }
            }
         }
         free(missingstacks);
      } else {
         // not process 0
         MPI_Status mystat;
         // receive how many entries process 0 is missing
         int nmissing;
         PMPI_Recv(&nmissing, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &mystat);
         // if at least one entry is missing proceed
         if (nmissing > 0) {
            // allocate space to hold the missing ids
            int *missingstacks = (int*) malloc(nmissing*sizeof(int));
            // receiving the missing stacks
            PMPI_Recv(missingstacks, nmissing, MPI_INT, 0, 0, MPI_COMM_WORLD, &mystat);

            // check how many of the missing stacks this process has infos about
            int hasnmissing = 0;
            for (int imissing = 0; imissing < nmissing; imissing++) {
               int globID = missingstacks[imissing];
               if (global2local_ID[globID] >= 0) {
                  hasnmissing++;
               }
            }
            // Report back how many missing stacks this process can fill in
            PMPI_Send(&hasnmissing, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            // only proceed if the number of stacks is positive
            if (hasnmissing > 0) {
               missing_stack_transfer_t *missing_stack_info =
                   (missing_stack_transfer_t*)malloc(hasnmissing * sizeof(missing_stack_transfer_t));
               // Go through the stacks and record the needed information
               int imatch = 0;
               for (int istack = 0; istack < nmissing; istack++) {
                  int globID = missingstacks[istack];
                  int locID = global2local_ID[globID];
                  if (locID >= 0) {
                     missing_stack_info[imatch].gid = globID;
                     vftr_stack_t *local_stack = stacktree_ptr->stacks+locID;
                     missing_stack_info[imatch].return_id = local2global_ID[local_stack->caller];
                     // add one to length due to null terminator
                     missing_stack_info[imatch].name_length = strlen(local_stack->cleanname) + 1;
                     missing_stack_info[imatch].precise = local_stack->precise ? 1 : 0;
                     missing_stack_info[imatch].ncallees = local_stack->ncallees;
                     imatch++;
                  }
               }
               // Communicate the found information to process 0;
               PMPI_Send(missing_stack_info, hasnmissing, missing_stack_transfer_mpi_t,
                         0, 0, MPI_COMM_WORLD);
               // Create a buffer that contains all stack names in contatenated form
               int sumlength = 0;
               for (int istack = 0; istack < hasnmissing; istack++) {
                  sumlength += missing_stack_info[istack].name_length;
               }
               char *concatNames = (char*) malloc(sumlength*sizeof(char));
               // concatenate the names into one string
               char *tmpstrptr = concatNames;
               for (int istack = 0; istack < hasnmissing; istack++) {
                  int globID = missing_stack_info[istack].gid;
                  int locID = global2local_ID[globID];
                  strcpy(tmpstrptr, stacktree_ptr->stacks[locID].cleanname);
                  tmpstrptr += missing_stack_info[istack].name_length-1;
                  // add null terminator
                  *tmpstrptr = '\0';
                  tmpstrptr++;
               }
               // communicate the concatenated string to process 0
               PMPI_Send(concatNames, sumlength, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

               int n_callees_tot = 0;
               for (int istack = 0; istack < hasnmissing; istack++) {
                  n_callees_tot += missing_stack_info[istack].ncallees;
               }

               int *all_callees = (int*)malloc(n_callees_tot * sizeof(int));
               int idx = 0;
               for (int istack = 0; istack < hasnmissing; istack++) {
                  int globID = missingstacks[istack];
                  int locID = global2local_ID[globID];
                  if (locID >= 0) {
                     vftr_stack_t *local_stack = stacktree_ptr->stacks+locID;
                     for (int icallee = 0; icallee < local_stack->ncallees; icallee++) {
                        all_callees[idx++] = local2global_ID[local_stack->callees[icallee]];
                     }
                  }
               }

               PMPI_Send (all_callees, n_callees_tot, MPI_INT, 0, 0, MPI_COMM_WORLD);

               // free everything. This should be all on the remote processes
               free(concatNames);
               free(all_callees);
               free(missing_stack_info);
            }
            free(missingstacks);
         }
      }

      vftr_broadcast_collated_stacktree(&coll_stacktree);
   }
#endif

   free(local2global_ID);
   free(global2local_ID);

   SELF_PROFILE_END_FUNCTION;
   return coll_stacktree;
}

int vftr_collated_stacktree_insert_namegroup(collated_stacktree_t *stacktree_ptr,
                                              collated_stack_t *stack_ptr) {
   int idx = stacktree_ptr->nstacks;
   stacktree_ptr->nstacks++;
   vftr_collated_stacktree_realloc(stacktree_ptr);
   // shift all entries up until the right spot to insert is found
   while (idx > 0 && strcmp(stacktree_ptr->stacks[idx-1].name, stack_ptr->name) > 0) {
      stacktree_ptr->stacks[idx] = stacktree_ptr->stacks[idx-1];
      idx--;
   }
   stacktree_ptr->stacks[idx] = *stack_ptr;
   return idx;
}

void vftr_collated_stacktree_insert_gid(gid_list_t *gid_list_ptr, int gid) {
   int idx = gid_list_ptr->ngids;
   gid_list_ptr->ngids++;
   vftr_gid_list_realloc(gid_list_ptr);
   // shift all entries up until the right spot to insert is found
   while (idx > 0 && gid < gid_list_ptr->gids[idx-1]) {
      gid_list_ptr->gids[idx] = gid_list_ptr->gids[idx-1];
      idx--;
   }
   gid_list_ptr->gids[idx] = gid;
}

collated_stacktree_t vftr_collated_stacktree_group_by_name(
   collated_stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   collated_stacktree_t grouped_stacktree = vftr_new_empty_collated_stacktree();
   grouped_stacktree.namegrouped = true;

   for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
      collated_stack_t *stack = stacktree_ptr->stacks+istack;

      // check if stack is already present
      int idx = vftr_binary_search_collated_stacks_name(grouped_stacktree, stack->name);
      collated_stack_t *grouped_stack = NULL;
      if (idx < 0) {
         idx = vftr_collated_stacktree_insert_namegroup(&grouped_stacktree, stack);
         grouped_stack = grouped_stacktree.stacks+idx;
      } else {
         grouped_stack = grouped_stacktree.stacks+idx;
         grouped_stack->profile = vftr_add_collated_profiles(grouped_stack->profile,
                                                             stack->profile);
      }

      vftr_collated_stacktree_insert_gid(&(grouped_stack->gid_list), stack->gid);
   }
   SELF_PROFILE_END_FUNCTION;
   return grouped_stacktree;
}

void vftr_collated_stacktree_free(collated_stacktree_t *stacktree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   if (stacktree_ptr->nstacks > 0) {
      for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
         // the grouped stacks only contain a copy of the reference to one of the
         // stacks with the name
         if (!stacktree_ptr->namegrouped) {
            free(stacktree_ptr->stacks[istack].name);
         }
         vftr_collated_profile_free(&(stacktree_ptr->stacks[istack].profile));
         vftr_gid_list_free(&(stacktree_ptr->stacks[istack].gid_list));
      }
      free(stacktree_ptr->stacks);
      stacktree_ptr->stacks = NULL;
      stacktree_ptr->nstacks = 0;
   }
   SELF_PROFILE_END_FUNCTION;
}

char *vftr_get_collated_stack_string(collated_stacktree_t stacktree,
                                     int stackid, bool show_precise) {
   SELF_PROFILE_START_FUNCTION;
   int stringlen = 0;
   int tmpstackid = stackid;
   stringlen += strlen(stacktree.stacks[stackid].name);
   stringlen ++; // function seperating character "<", or null terminator
   if (stacktree.stacks[tmpstackid].precise) {
      stringlen ++; // '*' for indicating precise functions
   }
   while (stacktree.stacks[tmpstackid].caller >= 0) {
      tmpstackid = stacktree.stacks[tmpstackid].caller;
      stringlen += strlen(stacktree.stacks[tmpstackid].name);
      stringlen ++; // function seperating character "<", or null terminator
      if (show_precise && stacktree.stacks[tmpstackid].precise) {
         stringlen ++; // '*' for indicating precise functions
      }
   }
   char *stackstring = (char*) malloc(stringlen*sizeof(char));
   // copy the chars one by one so there is no need to call strlen again.
   // thus minimizing reading the same memory locations over and over again.
   tmpstackid = stackid;
   char *tmpname_ptr = stacktree.stacks[tmpstackid].name;
   char *tmpstackstring_ptr = stackstring;
   while (*tmpname_ptr != '\0') {
      *tmpstackstring_ptr = *tmpname_ptr;
      tmpstackstring_ptr++;
      tmpname_ptr++;
   }
   if (show_precise && stacktree.stacks[tmpstackid].precise) {
      *tmpstackstring_ptr = '*';
      tmpstackstring_ptr++;
   }
   while (stacktree.stacks[tmpstackid].caller >= 0) {
      // add function name separating character
      *tmpstackstring_ptr = '<';
      tmpstackstring_ptr++;
      tmpstackid = stacktree.stacks[tmpstackid].caller;
      char *tmpname_ptr = stacktree.stacks[tmpstackid].name;
      while (*tmpname_ptr != '\0') {
         *tmpstackstring_ptr = *tmpname_ptr;
         tmpstackstring_ptr++;
         tmpname_ptr++;
      }
      if (show_precise && stacktree.stacks[tmpstackid].precise) {
         *tmpstackstring_ptr = '*';
         tmpstackstring_ptr++;
      }
   }
   // replace last char with a null terminator
   *tmpstackstring_ptr = '\0';
   SELF_PROFILE_END_FUNCTION;
   return stackstring;
}

bool vftr_collstack_is_init (collated_stack_t stack) {
   return stack.local_stack != NULL && stack.local_stack->lid == 0;
}

void vftr_print_name_grouped_collated_stack(FILE *fp, 
                                            collated_stacktree_t stacktree,
                                            int stackid) {
   collated_stack_t stack = stacktree.stacks[stackid];
   fprintf(fp, "%s: %d", stack.name, stack.gid_list.gids[0]);
   for (int igid=1; igid<stack.gid_list.ngids; igid++) {
      fprintf(fp, ",%d", stack.gid_list.gids[igid]);
   }
}

void vftr_print_collated_stack(FILE *fp, collated_stacktree_t stacktree, int stackid) {
   char *stackstr = vftr_get_collated_stack_string(stacktree, stackid, false);
   fprintf(fp, "%s", stackstr);
   free(stackstr);
}

void vftr_print_collated_stacklist(FILE *fp, collated_stacktree_t stacktree) {
   if (stacktree.namegrouped) {
      for (int istack=0; istack<stacktree.nstacks; istack++) {
         fprintf(fp, "%u: ", istack);
         vftr_print_name_grouped_collated_stack(fp, stacktree, istack);
         fprintf(fp, "\n");
      }
   } else {
      for (int istack=0; istack<stacktree.nstacks; istack++) {
         fprintf(fp, "%u: ", istack);
         vftr_print_collated_stack(fp, stacktree, istack);
         fprintf(fp, "\n");
      }
   }
}
