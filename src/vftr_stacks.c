/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "vftr_symbols.h"
#include "vftr_stacks.h"
#include "vftr_hashing.h"
#include "vftr_environment.h"
#include "vftr_timer.h"
#include "vftr_setup.h"
#include "vftr_hooks.h"

#include "vftr_output_macros.h"

// Maximum time in a call tree, searched for in vftr_finalize
long long vftr_maxtime;

// number of locally unique stacks
int   vftr_stackscount = 0;
// number of globally unique stacks
int   vftr_gStackscount = 0;

// Collective information about all stacks across processes
gstackinfo_t  *vftr_gStackinfo = NULL;

// Table of all functions defined
function_t **vftr_func_table = NULL;
// Size of vftr_func_table (grows as needed)
unsigned int  vftr_func_table_size = 5000;

// Function call stack
function_t *vftr_fstack = NULL;
// Function call stack roots
function_t *vftr_froots = NULL;
// Profile data sample
profdata_t vftr_prof_data;

// initialize stacks only called from vftr_initialize
void vftr_initialize_stacks() {
   // Allocate stack tables for each thread
   vftr_fstack = (function_t*) malloc(sizeof(function_t));
   vftr_froots = (function_t*) malloc(sizeof(function_t));

   // Initialize stack tables 
   char *s = "init";
   function_t *func = vftr_new_function (NULL, strdup (s), NULL, 0, true);
   func->next_in_level = func; /* Close circular linked list to itself */
   vftr_fstack = func;
   vftr_samplecount = 0;
   vftr_maxtime = 0;
   vftr_froots = func;
}

// Write the stacks out
void vftr_write_stacks (FILE *fp, int level, function_t *func) {
   int len = strlen(func->name);
   char *name = (char*) malloc(len + 2);
   strncpy(name, func->name, len);

   if (func->precise) {
      name[len++] = '*';
   }

   int levels = level + 1;
   int ret = func->return_to ? func->return_to->id : 0;
   fwrite(&func->id, sizeof(int), 1, fp);
   //fwrite(&func->gid, sizeof(int), 1, fp);
   fwrite(&levels, sizeof(int), 1, fp);
   fwrite(&ret, sizeof(int), 1, fp);
   fwrite(&len, sizeof(int), 1, fp);
   fwrite(name, sizeof(char), len, fp);

   // Recursive print of callees
   function_t *f = func->first_in_level;
   for(int i=0; i<func->levels; i++,f=f->next_in_level) {
      vftr_write_stacks(fp, level+1, f);
   }
}

// synchronise the global stack IDs among different processes
int vftr_normalize_stacks() {
    // fill the local hashtable for stacks
    vftr_gStackscount = vftr_stackscount;
    uint64_t *stackhashtable = (uint64_t*) malloc(vftr_stackscount*sizeof(uint64_t));
    for (int istack=0; istack<vftr_stackscount; istack++) {
       stackhashtable[istack] = vftr_func_table[istack]->stackHash;
    }
    // sort and synchronize the hashes 
    // thus every process has the exact same table of stack hashes
    vftr_synchronise_hashes(&vftr_gStackscount, &stackhashtable);

    // Build a lookup table each to translate local2global and global2local
    int *local2global_ID = (int*) malloc(vftr_stackscount*sizeof(int));
    int *global2local_ID = (int*) malloc(vftr_gStackscount*sizeof(int));
    for (int istack=0; istack<vftr_stackscount; istack++) {
       // -1 in the lookup table means that the local stack does not exist
       local2global_ID[istack] = -1;
    }
    for (int istack=0; istack<vftr_gStackscount; istack++) {
       // -1 in the lookup table means that the local stack does not exist
       global2local_ID[istack] = -1;
    }
    // assign every function its global ID
    // the global ID is the index in the hash table
    // fill the looup tables
    // TODO: implement binary search to speed up globalID assignment
    for (int istack=0; istack<vftr_stackscount; istack++) {
       for (int ihash =0; ihash<vftr_gStackscount; ihash++) {
          if (vftr_func_table[istack]->stackHash == stackhashtable[ihash]) {
             vftr_func_table[istack]->gid = ihash;
             int globID = vftr_func_table[istack]->gid;
             local2global_ID[istack] = globID;
             global2local_ID[globID] = istack;
             break;
          }
       }
    }

    // the hashtable is nolonger needed
    free(stackhashtable);
    stackhashtable = NULL;

    // process zero needs to collect the function names
    // of every function called on all processes
    if (vftr_mpirank == 0) {
       vftr_gStackinfo = (gstackinfo_t*) malloc(vftr_gStackscount*sizeof(gstackinfo_t));
       // init the global stack info
       for (int istack=0; istack<vftr_gStackscount; istack++) {
          vftr_gStackinfo[istack].ret = -2;
          vftr_gStackinfo[istack].name = NULL;
          vftr_gStackinfo[istack].locID = -1;
       }
       // fill in global info process 0 knows
       for (int istack=0; istack<vftr_stackscount; istack++) {
          int globID = local2global_ID[istack];
          vftr_gStackinfo[globID].name = strdup(vftr_func_table[istack]->name);
          if (strcmp(vftr_gStackinfo[globID].name, "init")) {
             // not the init function
             vftr_gStackinfo[globID].ret = vftr_func_table[istack]->return_to->gid;
             vftr_gStackinfo[globID].locID = istack;
          } else {
             vftr_gStackinfo[globID].ret = -1;
             vftr_gStackinfo[globID].locID = 0;
          }

       }
#ifdef _MPI
       // if there are multiple processes the table might still be missing entries

       // see which entries are missing
       int nmissing = 0;
       for (int istack=0; istack<vftr_gStackscount; istack++) {
          if (vftr_gStackinfo[istack].ret == -2) {
             nmissing++;
          }
       }
       int *missingStacks = (int*) malloc(nmissing*sizeof(int));

       // loop over all other ranks and collect the missing information
       for (int irank=1; irank<vftr_mpisize; irank++) {
          // count how many are still missing
          int nmissing= 0;
          for (int istack=0; istack<vftr_gStackscount; istack++) {
             if (vftr_gStackinfo[istack].ret == -2) {
                nmissing++;
             }
          }
          // collect the missing ids
          int imissing = 0;
          for (int istack=0; istack<vftr_gStackscount; istack++) {
             if (vftr_gStackinfo[istack].ret == -2) {
                missingStacks[imissing] = istack;
                imissing++;
             }
          }
          // Send to the selected process how many entries are still missing
          PMPI_Send(&nmissing, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
          // if at least one entry is missing proceed
          if (nmissing > 0) {
             // Send the missing IDs

             PMPI_Send(missingStacks, nmissing, MPI_INT, irank, 0, MPI_COMM_WORLD);
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
                   vftr_gStackinfo[globID].ret = missingStackInfo[3*istack+1];
                   vftr_gStackinfo[globID].name = strdup(tmpstrptr);
                   // next string
                   tmpstrptr += missingStackInfo[3*istack+2];
                }

                free(concatNames);
                free(missingStackInfo);
             }
          }
       }

       free(missingStacks);
    } else {
       // not process 0
       MPI_Status mystat;
       // receive how many entries process 0 is missing
       int nmissing;
       PMPI_Recv(&nmissing, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &mystat);
       // if at least one entry is missing proceed
       if (nmissing > 0) {
          // allocate space to hold the missing ids
          int *missingStacks = (int*) malloc(nmissing*sizeof(int));
          // receiving the missing stacks
          PMPI_Recv(missingStacks, nmissing, MPI_INT, 0, 0, MPI_COMM_WORLD, &mystat);

          // check how many of the missing stacks this process has infos about
          int hasnmissing = 0;
          for (int imissing=0; imissing<nmissing; imissing++) {
             int globID = missingStacks[imissing];
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
                int globID = missingStacks[istack];
                int locID = global2local_ID[globID];
                if (locID >= 0) {
                   missingStackInfo[3*imatch+0] =
                      globID;
                   missingStackInfo[3*imatch+1] =
                      vftr_func_table[locID]->return_to->gid;
                   // add one to length due to null terminator
                   missingStackInfo[3*imatch+2] =
                      strlen(vftr_func_table[locID]->name)+1;
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
                strcpy(tmpstrptr, vftr_func_table[locID]->name);
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
          free(missingStacks);
          missingStacks = NULL;
       }

#endif
    }

    free(local2global_ID);
    free(global2local_ID);

    return 0;
}

/**********************************************************************/

void vftr_print_stack (double time0, function_t *func, char *label, int timeToSample) {
    function_t   *f;
    char         *mark;

    if (func->new) {
        func->new = false;
        mark = "*";
    } else {
        mark = "";
    }

    fprintf (vftr_log, "%s%12.6lf %4d %s %s", 
             timeToSample ? "+" : " ", time0, func->id, label, mark );

    for (f = func; f; f = f->return_to) fprintf (vftr_log, "%s<", f->name);
    fprintf (vftr_log, "\n");
}

/**********************************************************************/

void vftr_print_local_stacklist (function_t **funcTable, FILE *pout, int ntop) {
    char *fmtFid;
    int  i, fidp, namep, tableWidth, maxID;
    int  useGid = (pout != stdout && (vftr_mpisize > 1));
    
    if (!vftr_profile_wanted) return;

    /* Compute column and table widths */

    for (i = 0,namep = 0,maxID = 0; i < ntop; i++) {
        function_t *func = funcTable[i];
        if (func == NULL || !func->return_to) continue;
        int width, id;
        id = useGid ? func->gid : func->id;
        for (width = 0; func; func = func->return_to) {
            width += strlen (func->name) + 1;
	}
        if (namep < width) namep = width;
        if (maxID < id) maxID = id;
    }

    COMPUTE_COLWIDTH( maxID, fidp, 2, fmtFid, " %%%dd "  )
    tableWidth = 1 + fidp+1 + namep;

    /* Print headers */

    fputs( "Call stacks\n", pout );

    OUTPUT_DASHES_NL( tableWidth, pout )

    fputs( " ", pout );
    OUTPUT_HEADER( "ID", fidp, pout )
    fputs( "Function call stack\n", pout );

    fputs( " ", pout );
    OUTPUT_DASHES_SP_2( fidp, namep, pout )
    fputs( "\n", pout );

    /* Print table */

    for (i = 0; i < ntop; i++) {
        char *sep; 
        int  id;
        function_t *func = funcTable[i];
        if (func == NULL || !func->return_to) continue; /* If not defined or no caller */
	id = useGid ? func->gid : func->id;
        fprintf( pout, fmtFid, id );
        for( sep=""; func; func=func->return_to, sep="<")
            fprintf( pout, "%s%s", sep, func->name  );
        fprintf( pout, "\n" );
    }
    OUTPUT_DASHES_NL( tableWidth, pout )
    fputs( "\n", pout );
}

/**********************************************************************/

void vftr_print_local_demangled (function_t **funcTable, FILE *pout, int ntop) {
    char *fmtFid;
    int  i, fidp, namep, tableWidth, maxID;
    int  useGid = pout!=stdout && ( vftr_mpisize > 1 );
    
    if (!vftr_profile_wanted) return;

    /* Compute column and table widths */

    for( i=0,namep=0,maxID=0; i<ntop; i++ ) {
        function_t *func = funcTable[i];
        int        width, id;
        if( func == NULL || !func->return_to ||              /* If not defined or no caller */
            func->full == NULL            ) continue;  /* or no full demangled name */
        id = useGid ? func->gid : func->id;
        width = strlen(func->full);
        if( namep < width ) namep = width;
        if( maxID < id    ) maxID = id;
    }
    if( namep == 0 ) return; /* If no demangled names */

    COMPUTE_COLWIDTH( maxID, fidp, 2, fmtFid, " %%%dd "  )
    tableWidth = 1 + fidp+1 + namep;

    /* Print headers */

    OUTPUT_DASHES_NL( tableWidth, pout )

    fputs( " ", pout );
    OUTPUT_HEADER( "ID", fidp, pout )
    fputs( "Full demangled name\n", pout );

    fputs( " ", pout );
    OUTPUT_DASHES_SP_2( fidp, namep, pout )
    fputs( "\n", pout );

    /* Print table */

    for( i=0; i<ntop; i++ ) {
        int  id;
        function_t *func = funcTable[i];
        if( func == NULL || !func->return_to ||              /* If not defined or no caller */
            func->full == NULL            ) continue;  /* or no full demangled name */
	id = useGid ? func->gid : func->id;
        fprintf( pout, fmtFid, id );
        fprintf( pout, "%s\n", func->full  );
    }
    OUTPUT_DASHES_NL( tableWidth, pout )
    fputs( "\n", pout );
}

/**********************************************************************/

void vftr_print_global_stacklist (FILE *pout) {

   // Compute column and table widths
   // loop over all stacks to find the longest one
   int maxstrlen = 0;
   for (int istack=0; istack<vftr_gStackscount; istack++) {
      int jstack = istack;
      // follow the functions until they reach the bottom of the stack
      int stackstrlength = 0;
      while (vftr_gStackinfo[jstack].locID && vftr_gStackinfo[jstack].ret >= 0) {
         stackstrlength += strlen(vftr_gStackinfo[jstack].name);
         stackstrlength ++;
         jstack = vftr_gStackinfo[jstack].ret;
      } 
      stackstrlength += strlen(vftr_gStackinfo[jstack].name);
      if (stackstrlength > maxstrlen) maxstrlen = stackstrlength;
   }
   int maxID = vftr_gStackscount;
   maxstrlen--; // Chop trailing space
   char *fmtFid;
   int fidp;
   COMPUTE_COLWIDTH( maxID, fidp, 2, fmtFid, " %%%dd "  )
   int tableWidth = 1 + fidp+1 + maxstrlen;

   /* Print headers */

   fputs( "Call stacks\n", pout );

   OUTPUT_DASHES_NL( tableWidth, pout )

   fputs( " ", pout );
   OUTPUT_HEADER( "ID", fidp, pout )
   fputs( "Function call stack\n", pout );

   fputs( " ", pout );
   OUTPUT_DASHES_SP_2( fidp, maxstrlen, pout )
   fputs( "\n", pout );

   /* Print table */

   for(int istack=0; istack<vftr_gStackscount; istack++) {
      int jstack = istack;
      fprintf( pout, fmtFid, istack);
      while (vftr_gStackinfo[jstack].locID && vftr_gStackinfo[jstack].ret >= 0) {
         fprintf(pout, "%s", vftr_gStackinfo[jstack].name);
         fprintf(pout, "<");
         jstack = vftr_gStackinfo[jstack].ret;
      }
      fprintf(pout, "%s", vftr_gStackinfo[jstack].name);
      fprintf(pout, "\n");
   }

   OUTPUT_DASHES_NL( tableWidth, pout )
   fputs( "\n", pout );
}
