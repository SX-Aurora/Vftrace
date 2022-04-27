#include <stdlib.h>

#include <string.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include "stack_types.h"
#include "collated_hash_types.h"
#include "collated_stack_types.h"

#include "hashing.h"
#include "collate_hashes.h"
#include "search.h"

collated_stacktree_t vftr_new_collated_stacktree() {
   collated_stacktree_t stacktree;
   stacktree.nstacks = 0;
   stacktree.stacks = NULL;
   return stacktree;
}

#ifdef _MPI
void vftr_broadcast_collated_stacktree_root(collated_stacktree_t *stacktree_ptr) {
   int nranks;
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   int nstacks = stacktree_ptr->nstacks;
   // broadcasting the caller ids
   int *tmpintarr = (int*) malloc((nstacks+1)*sizeof(int));
   for (int istack=0; istack<nstacks; istack++) {
      tmpintarr[istack] = stacktree_ptr->stacks[istack].caller;
   }
   PMPI_Bcast(tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcasting the function names.
   // first the length of each name
   int totallen = 0;
   for (int istack=0; istack<nstacks; istack++) {
      tmpintarr[istack] = strlen(stacktree_ptr->stacks[istack].name);
      totallen += tmpintarr[istack];
      totallen += 1; // null terminator between all strings
   }
   // the total length is appended to the tmpintarr 
   tmpintarr[nstacks] = totallen;
   PMPI_Bcast(tmpintarr, nstacks+1, MPI_INT, 0, MPI_COMM_WORLD);
   char *tmpchararr = (char*) malloc(totallen*sizeof(char));
   char *charptr = tmpchararr;
   for (int istack=0; istack<nstacks; istack++) {
      strcpy(charptr, stacktree_ptr->stacks[istack].name);
      charptr += tmpintarr[istack];
      charptr++; // null terminator should be copied by strcpy
   }
   PMPI_Bcast(tmpchararr, totallen, MPI_CHAR, 0, MPI_COMM_WORLD);
   free(tmpintarr);
   free(tmpchararr);
}

void vftr_broadcast_collated_stacktree_receivers(collated_stacktree_t *stacktree_ptr) {
   int nranks;
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   int nstacks = stacktree_ptr->nstacks;
   // receiving the caller ids
   int *tmpintarr = (int*) malloc(nstacks+1*sizeof(int));
   PMPI_Bcast(tmpintarr, nstacks, MPI_INT, 0, MPI_COMM_WORLD);
   for (int istack=0; istack<nstacks; istack++) {
      stacktree_ptr->stacks[istack].caller = tmpintarr[istack];
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
   for (int istack=0; istack<nstacks; istack++) {
      stacktree_ptr->stacks[istack].name = strdup(charptr);
      charptr += tmpintarr[istack];
      charptr++; // null terminator should be copied by strcpy
   }
   free(tmpintarr);
   free(tmpchararr);
}

void vftr_broadcast_collated_stacktree(collated_stacktree_t *stacktree_ptr) {
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
}
#endif

collated_stacktree_t vftr_collate_stacks(stacktree_t *stacktree_ptr) {
   collated_stacktree_t coll_stacktree;

   // first compute the hashes for all stacks
   vftr_compute_stack_hashes(stacktree_ptr);

   // collate hashes between processes
   hashlist_t hashlist = vftr_collate_hashes(stacktree_ptr);
   
   // create empty collated stacktree
   coll_stacktree.nstacks = hashlist.nhashes;
   coll_stacktree.stacks = (collated_stack_t*)
      malloc(coll_stacktree.nstacks*sizeof(collated_stack_t));
   for (int istack=0; istack<coll_stacktree.nstacks; istack++) {
      coll_stacktree.stacks[istack].local_stack = NULL;
      coll_stacktree.stacks[istack].gid = istack;
      coll_stacktree.stacks[istack].caller = -1;
      coll_stacktree.stacks[istack].name = NULL;
      coll_stacktree.stacks[istack].hash = hashlist.hashes[istack];
   }

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
   for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
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
   if (myrank == 0) {
      {
         int lid = 0;
         stack_t *local_stack = stacktree_ptr->stacks+lid;
         int gid = local2global_ID[lid];
         collated_stack_t *global_stack = coll_stacktree.stacks+gid;
         global_stack->local_stack = local_stack;
         global_stack->gid = gid;
         global_stack->caller = -1;
         global_stack->name = strdup(local_stack->name);
      }
      for (int lid=1; lid<stacktree_ptr->nstacks; lid++) {
         stack_t *local_stack = stacktree_ptr->stacks+lid;
         int gid = local2global_ID[lid];
         collated_stack_t *global_stack = coll_stacktree.stacks+gid;
         global_stack->local_stack = local_stack;
         global_stack->gid = gid;
         global_stack->caller = local2global_ID[local_stack->caller];
         global_stack->name = strdup(local_stack->name);
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
         for (int irank=1; irank<nranks; irank++) {
            int nmissing = 0;
            for (int istack=0; istack<coll_stacktree.nstacks; istack++) {
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
                   // Allocate space for the Stack information
                   // globalIDs  -> 3*istack+0
                   // returnID   -> 3*istack+1
                   // NameLength -> 3*istack+2
                   int *missingStackInfo = (int*) malloc(3*hasnmissing*sizeof(int));
                   // Receive the found information from remote process
                   PMPI_Recv(missingStackInfo, 3*hasnmissing, MPI_INT,
                             irank, 0, MPI_COMM_WORLD, &mystat);
   
                   // Create a buffer that contains all stack names in contatenated form
                   int sumlength = 0;
                   for (int istack=0; istack<hasnmissing; istack++) {
                      sumlength += missingStackInfo[3*istack+2];
                   }
                   char *concatNames = (char*) malloc(sumlength*sizeof(char));
   
                   // Receive the concatenated String
                   PMPI_Recv(concatNames, sumlength, MPI_CHAR,
                             irank, 0, MPI_COMM_WORLD, &mystat);
   
                   // Write all the gathered info to the global stackinfo
                   char *tmpstrptr = concatNames;
                   for (int istack=0; istack<hasnmissing; istack++) {
                      int globID = missingStackInfo[3*istack+0];
                      coll_stacktree.stacks[globID].caller = missingStackInfo[3*istack+1];
                      coll_stacktree.stacks[globID].name = strdup(tmpstrptr);
                      // next string
                      tmpstrptr += missingStackInfo[3*istack+2];
                   }
   
                   free(concatNames);
                   free(missingStackInfo);
   
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
            for (int imissing=0; imissing<nmissing; imissing++) {
               int globID = missingstacks[imissing];
               if (global2local_ID[globID] >= 0) {
                  hasnmissing++;
               }
            }
            // Report back how many missing stacks this process can fill in
            PMPI_Send(&hasnmissing, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            // only proceed if the number of stacks is positive
            if (hasnmissing > 0) {
               // Allocate space for the global IDs, return ID, and the name lengths
               // globalIDs  -> 3*istack+0
               // returnID   -> 3*istack+1
               // NameLength -> 3*istack+2
               int *missingStackInfo = (int*) malloc(3*hasnmissing*sizeof(int));
               // Go through the stacks and record the needed information
               int imatch = 0;
               for (int istack=0; istack<nmissing; istack++) {
                  int globID = missingstacks[istack];
                  int locID = global2local_ID[globID];
                  if (locID >= 0) {
                     missingStackInfo[3*imatch+0] = globID;
                     stack_t *local_stack = stacktree_ptr->stacks+locID;
                     missingStackInfo[3*imatch+1] = local2global_ID[local_stack->caller];
                     // add one to length due to null terminator
                     missingStackInfo[3*imatch+2] = strlen(local_stack->name) + 1;
                     imatch++;
                  }
               }
               // Communicate the found information to process 0;
               PMPI_Send(missingStackInfo, 3*hasnmissing, MPI_INT, 0, 0, MPI_COMM_WORLD);
               // Create a buffer that contains all stack names in contatenated form
               int sumlength = 0;
               for (int istack=0; istack<hasnmissing; istack++) {
                  sumlength += missingStackInfo[3*istack+2];
               }
               char *concatNames = (char*) malloc(sumlength*sizeof(char));
               // concatenate the names into one string
               char *tmpstrptr = concatNames;
               for (int istack=0; istack<hasnmissing; istack++) {
                  int globID = missingStackInfo[3*istack+0];
                  int locID = global2local_ID[globID];
                  strcpy(tmpstrptr, stacktree_ptr->stacks[locID].name);
                  tmpstrptr += missingStackInfo[3*istack+2]-1;
                  // add null terminator
                  *tmpstrptr = '\0';
                  tmpstrptr++;
               }
               // communicate the concatenated string to process 0
               PMPI_Send(concatNames, sumlength, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
   
               // free everything. This should be all on the remote processes
               free(concatNames);
               free(missingStackInfo);
            }
            free(missingstacks);
         }
      }
   
      vftr_broadcast_collated_stacktree(&coll_stacktree);
   }
#endif

   free(local2global_ID);
   free(global2local_ID);

   return coll_stacktree;
}

void vftr_collated_stacktree_free(collated_stacktree_t *stacktree_ptr) {
   if (stacktree_ptr->nstacks > 0) {
      for (int istack=0; istack<stacktree_ptr->nstacks; istack++) {
         free(stacktree_ptr->stacks[istack].name);
      }
      free(stacktree_ptr->stacks);
      stacktree_ptr->stacks = NULL;
      stacktree_ptr->nstacks = 0;
   }
}

void vftr_print_collated_stack(FILE *fp, collated_stacktree_t stacktree, int stackid) {
   fprintf(fp, "%s", stacktree.stacks[stackid].name);
   if (stacktree.stacks[stackid].caller >= 0) {
      fprintf(fp, "<");
      vftr_print_collated_stack(fp, stacktree, stacktree.stacks[stackid].caller);
   }
}

void vftr_print_collated_stacklist(FILE *fp, collated_stacktree_t stacktree) {
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      fprintf(fp, "%u: ", istack);
      vftr_print_collated_stack(fp, stacktree, istack);
      fprintf(fp, "\n");
   }
}
