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
#include "vftr_html.h"
#include "vftr_filewrite.h"
#include "vftr_fileutils.h"

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

const char *vftr_stacktree_headers[6] = {"T[s]", "Calls", "Imbalance[%]", "Total send", "Total recv.", "Stack ID"};

/**********************************************************************/

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

/**********************************************************************/

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
	  vftr_gStackinfo[istack].print_profile = false;
       }
       // fill in global info process 0 knows
       for (int istack=0; istack<vftr_stackscount; istack++) {
          int globID = local2global_ID[istack];
          vftr_gStackinfo[globID].name = strdup(vftr_func_table[istack]->name);
	  if (vftr_environment.print_stack_profile->set) {
		if (vftr_pattern_match (vftr_environment.print_stack_profile->value, 
				        vftr_func_table[istack]->name)) { 
			vftr_gStackinfo[globID].print_profile = true;
	        } // else if match stack id
   	  }
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

#ifdef _MPI
    // If the logfile is supposed to be available for all ranks,
    // the global stack info needs to be communicated to all ranks.
    // We also need to communicate if stack profiles with imbalances are to be printed,
    // because identical function stacks can be located at different positions in the
    // function table or not be present at all. 
    if (vftr_environment.logfile_all_ranks->value || vftr_environment.print_stack_profile->value) {
       // The amount of unique stacks is know due to the hash synchronisation earlier
       // allocate memory on all but 0th rank
       if (vftr_mpirank != 0) {
          vftr_gStackinfo = (gstackinfo_t*) malloc(vftr_gStackscount*sizeof(gstackinfo_t));
       }

       // temporary arrays to hold information to be distributed
       int *tmpStackInfos = (int*) malloc(vftr_gStackscount*sizeof(int));

       // distribute the return values of the individual stacks
       if (vftr_mpirank == 0) {
          // rank 0 prepares the data
          for (int i=0; i<vftr_gStackscount; i++) {
             tmpStackInfos[i] = vftr_gStackinfo[i].ret;
          }
          PMPI_Bcast(tmpStackInfos,
                     vftr_gStackscount,
                     MPI_INT,
                     0,
                     MPI_COMM_WORLD);
       } else {
          // all other ranks receive and write the data in the local copy of the global stackinfo
          PMPI_Bcast(tmpStackInfos,
                     vftr_gStackscount,
                     MPI_INT,
                     0,
                     MPI_COMM_WORLD);
          for (int i=0; i<vftr_gStackscount; i++) {
             vftr_gStackinfo[i].ret = tmpStackInfos[i];
          }
       }

       // distribute the names
       // first the length of names
       if (vftr_mpirank == 0) {
          // rank 0 prepares the data
          for (int i=0; i<vftr_gStackscount; i++) {
             tmpStackInfos[i] = strlen(vftr_gStackinfo[i].name)+1;
          }
          PMPI_Bcast(tmpStackInfos,
                     vftr_gStackscount,
                     MPI_INT,
                     0,
                     MPI_COMM_WORLD);
       } else {
          // all other ranks receive the length of the names
          PMPI_Bcast(tmpStackInfos,
                     vftr_gStackscount,
                     MPI_INT,
                     0,
                     MPI_COMM_WORLD);
       }
       // compute the total length of the strings
       int totalstrlenght = 0;
       for (int i=0; i<vftr_gStackscount; i++) {
          totalstrlenght += tmpStackInfos[i];
       }
       // Allocate buffer to hold total string
       char *concatNames = (char*) malloc(totalstrlenght*sizeof(char));
       if (vftr_mpirank == 0){
          // rank 0 concats names into one string
          char *tmpstrptr = concatNames;
          for (int istack=0; istack<vftr_gStackscount; istack++) {
             strcpy(tmpstrptr, vftr_gStackinfo[istack].name);
             tmpstrptr += tmpStackInfos[istack]-1;
             // add null terminator
             *tmpstrptr = '\0';
             tmpstrptr++;
          }

          // Distribute total string
          PMPI_Bcast(concatNames,
                     totalstrlenght,
                     MPI_CHAR,
                     0,
                     MPI_COMM_WORLD);
       } else {

          // all other ranks receive the total string
          PMPI_Bcast(concatNames,
                     totalstrlenght,
                     MPI_CHAR,
                     0,
                     MPI_COMM_WORLD);

          // take apart concatenated string and write it into the gStackinfo
          char *tmpstrptr = concatNames;
          for (int istack=0; istack<vftr_gStackscount; istack++) {
             vftr_gStackinfo[istack].name = strdup(tmpstrptr);
             // next string
             tmpstrptr += tmpStackInfos[istack];
          }
       }
       // free remaining buffers
       free(tmpStackInfos);
       tmpStackInfos = NULL;
       free(concatNames);
       concatNames = NULL;

       // All but rank 0 need to assign their local stack-IDs to the global info
       if (vftr_mpirank != 0 ) {
          for (int istack=0; istack<vftr_gStackscount; istack++) {
             vftr_gStackinfo[istack].locID = global2local_ID[istack];
          }
       }
    }
#endif

    free(local2global_ID);
    free(global2local_ID);

    return 0;
}

/**********************************************************************/

void vftr_write_stack_ascii (FILE *fp, double time0,
			     function_t *func, char *label, int timeToSample) {
    function_t *f;
    char *mark;

    if (func->new) {
        func->new = false;
        mark = "*";
    } else {
        mark = "";
    }

    fprintf (fp, "%s%12.6lf %4d %s %s", 
             timeToSample ? "+" : " ", time0, func->id, label, mark );

    for (f = func; f; f = f->return_to) {
	fprintf (fp, "%s<", f->name);
    }
    fprintf (fp, "\n");
}

/**********************************************************************/

void vftr_write_stacks_vfd (FILE *fp, int level, function_t *func) {
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
   for (int i = 0; i < func->levels; i++,f = f->next_in_level) {
      vftr_write_stacks_vfd (fp, level + 1, f);
   }
}

/**********************************************************************/

void vftr_print_local_stacklist (function_t **funcTable, FILE *pout, int ntop) {
    char *fmtFid;
    int  fidp, tableWidth;
    int  useGid = (pout != stdout && (vftr_mpisize > 1));
    
    if (!vftr_profile_wanted) return;

    /* Compute column and table widths */
    int namep = 0;
    int maxID = 0;
    for (int i = 0; i < ntop; i++) {
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

    for (int i = 0; i < ntop; i++) {
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

/**********************************************************************/

int vftr_stack_length (int stack_id0) {
	int n = 0;
	int stack_id = stack_id0;
	for (; stack_id > 0; stack_id = vftr_gStackinfo[stack_id].ret) {
		n++;
	}
	return n;
}

/**********************************************************************/

enum new_leaf_type {ORIGIN, NEXT, CALLEE};

void vftr_create_new_leaf (stack_leaf_t **new_leaf, int stack_id, int func_id, enum new_leaf_type leaf_type) {
	if (leaf_type == ORIGIN) {
		*new_leaf = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
		(*new_leaf)->stack_id = stack_id;
		(*new_leaf)->func_id = func_id;
	 	(*new_leaf)->final_id = -1;
		(*new_leaf)->next_in_level = NULL;
		(*new_leaf)->callee = NULL;
		(*new_leaf)->origin = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
		(*new_leaf)->origin = *new_leaf;
	} else if (leaf_type == NEXT) {
		(*new_leaf)->next_in_level = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
		(*new_leaf)->next_in_level->stack_id = stack_id;
		(*new_leaf)->next_in_level->func_id = func_id;
	 	(*new_leaf)->final_id = -1;
		(*new_leaf)->next_in_level->next_in_level = NULL;	
		(*new_leaf)->next_in_level->callee = NULL;
		(*new_leaf)->next_in_level->origin = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
		(*new_leaf)->next_in_level->origin = (*new_leaf)->origin;
	} else if (leaf_type == CALLEE) {
			(*new_leaf)->callee = (stack_leaf_t*) malloc (sizeof(stack_leaf_t));
			(*new_leaf)->callee->stack_id = stack_id;	
			(*new_leaf)->callee->func_id = func_id;	
	 		(*new_leaf)->final_id = -1;
			(*new_leaf)->callee->next_in_level = NULL;
			(*new_leaf)->callee->callee = NULL;
			(*new_leaf)->callee->origin = (stack_leaf_t*)malloc (sizeof(stack_leaf_t));
			(*new_leaf)->callee->origin = (*new_leaf)->origin;
	}
}

/**********************************************************************/

void vftr_fill_into_stack_tree (stack_leaf_t **this_leaf, int n_stack_ids,
			   int *stack_ids, int func_id) {
	int stack_id = stack_ids[n_stack_ids - 1];
	if (*this_leaf) {
		*this_leaf = (*this_leaf)->origin;
	} else {
		vftr_create_new_leaf (this_leaf, stack_id, func_id, ORIGIN);
	}
	for (int level = n_stack_ids - 2; level >= 0; level--) {
		stack_id = stack_ids[level];
		if ((*this_leaf)->callee) {
			*this_leaf = (*this_leaf)->callee;
			while ((*this_leaf)->stack_id != stack_id) { 
				if ((*this_leaf)->next_in_level) {
					*this_leaf = (*this_leaf)->next_in_level;
				} else {
					vftr_create_new_leaf (this_leaf, stack_id, func_id, NEXT);
					*this_leaf = (*this_leaf)->next_in_level;
					break;
				}
			}
		} else {
			vftr_create_new_leaf (this_leaf, stack_id, func_id, CALLEE);
			*this_leaf = (*this_leaf)->callee;
		}
	}	
}

/**********************************************************************/

void vftr_scan_for_final_values (stack_leaf_t *leaf, int this_n_spaces, double *imbalances,
				   int *n_spaces_max, int *n_final, double **t_final, int **n_calls_final, double **imba_final) {
	if (!leaf) return;
	if (leaf->callee) {
		int new_n_spaces = this_n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name);
		if (this_n_spaces > 0) new_n_spaces++;
		vftr_scan_for_final_values (leaf->callee, new_n_spaces, imbalances,
					      n_spaces_max, n_final, t_final, n_calls_final, imba_final);
	} else {
		int new_n_spaces = this_n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name) + 1; // + 1 for the colon at the end
		if (this_n_spaces > 0) new_n_spaces++;
		if (new_n_spaces > *n_spaces_max) *n_spaces_max = new_n_spaces;
		if (leaf->func_id > 0) {
		   double this_t = (double)vftr_func_table[leaf->func_id]->prof_current.timeIncl * 1e-6;
		   (*t_final)[*n_final] = this_t;
		   int this_n_calls = vftr_func_table[leaf->func_id]->prof_current.calls;
		   (*n_calls_final)[*n_final] = this_n_calls;
		   (*imba_final)[*n_final] = imbalances[leaf->func_id];
		   *n_final = *n_final + 1;
		}
	}
	if (leaf->next_in_level) {
		vftr_scan_for_final_values (leaf->next_in_level, this_n_spaces, imbalances,
					      n_spaces_max, n_final, t_final, n_calls_final, imba_final);
	}
}

/**********************************************************************/

void vftr_stacktree_assign_positions (stack_leaf_t *leaf, int *pos, int *indices) {
   if (!leaf) return;
   if (leaf->callee) {
      vftr_stacktree_assign_positions (leaf->callee, pos, indices);
   } else if (leaf->func_id > 0) {
      //if (vftr_mpirank == 0) printf ("%d, %d\n", *pos, indices[*pos]);
      leaf->final_id = indices[*pos];
      *pos = *pos + 1;	
   }
   if (leaf->next_in_level) {
      vftr_stacktree_assign_positions (leaf->next_in_level, pos, indices); 
   }
}
   
/**********************************************************************/

void vftr_print_stacktree_header (FILE *fp, int n_stacks, char *func_name,
				  int n_spaces_max, int fmt_calls, int fmt_t, int fmt_imba,
				  int fmt_send_bytes, int fmt_recv_bytes, int fmt_stackid) {
	int n_char_tot = n_spaces_max + fmt_calls + fmt_t + fmt_imba + fmt_send_bytes + fmt_recv_bytes + fmt_stackid + 18;
	char title[64];
	sprintf (title, "Function stacks leading to %s: %d", func_name, n_stacks);
	fprintf (fp, "%s", title);
	for (int i = 0; i < n_spaces_max - strlen(title); i++) fprintf (fp, " ");
	fprintf (fp, "   %*s   %*s   %*s   %*s   %*s   %*s   %*s\n", fmt_t, vftr_stacktree_headers[TIME],
		 fmt_calls, vftr_stacktree_headers[CALLS], fmt_imba, vftr_stacktree_headers[IMBA],
		 fmt_send_bytes, vftr_stacktree_headers[SEND_BYTES], fmt_recv_bytes, vftr_stacktree_headers[RECV_BYTES],
		 fmt_stackid, vftr_stacktree_headers[STACK_ID], strlen("position"), "position");
	vftr_print_dashes (fp, n_char_tot);
}

/**********************************************************************/

void vftr_print_stacktree (FILE *fp, stack_leaf_t *leaf, int n_spaces, double *imbalances, 
			   int n_spaces_max, int fmt_calls, int fmt_t, int fmt_imba,
		           int fmt_send_bytes, int fmt_recv_bytes, int fmt_stackid,
			   long long *total_time) {
	if (!leaf) return;
	fprintf (fp, vftr_gStackinfo[leaf->stack_id].name);
	if (leaf->callee) {
		fprintf (fp, ">");
		int new_n_spaces = n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name);
		if (n_spaces > 0) new_n_spaces++;
		vftr_print_stacktree (fp, leaf->callee, new_n_spaces, imbalances,
				      n_spaces_max, fmt_calls, fmt_t, fmt_imba,
			              fmt_send_bytes, fmt_recv_bytes, fmt_stackid,
				      total_time);
	} else {
		if (leaf->func_id < 0) {
			fprintf (fp, "[not on this rank]\n");
		} else {
			*total_time += vftr_func_table[leaf->func_id]->prof_current.timeIncl;
		        fprintf (fp, ":");
			for (int i = n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name) + 2; i < n_spaces_max; i++) {
			   fprintf (fp, " ");
			}
			char *send_unit_str, *recv_unit_str;
		        double mpi_tot_send_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_send_bytes;
		        double mpi_tot_recv_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_recv_bytes;
			vftr_memory_unit (&mpi_tot_send_bytes, &send_unit_str);	
			vftr_memory_unit (&mpi_tot_recv_bytes, &recv_unit_str);	
			fprintf (fp, "   %*.6f   %*d   %*.2f   %*.lf %s   %*.lf %s   %*d   %*d\n",
				 fmt_t, (double)vftr_func_table[leaf->func_id]->prof_current.timeIncl * 1e-6,
				 fmt_calls, vftr_func_table[leaf->func_id]->prof_current.calls,
				 fmt_imba, imbalances[leaf->func_id],
			 	 fmt_send_bytes - 4, mpi_tot_send_bytes, send_unit_str,
				 fmt_recv_bytes - 4, mpi_tot_recv_bytes, recv_unit_str,
				 fmt_stackid, leaf->stack_id, strlen("position"), leaf->final_id);
		}
	}
	if (leaf->next_in_level) {
		for (int i = 0; i < n_spaces; i++) fprintf (fp, " ");
		fprintf (fp, ">");
		vftr_print_stacktree (fp, leaf->next_in_level, n_spaces, imbalances,
				      n_spaces_max, fmt_calls, fmt_t, fmt_imba,
				      fmt_send_bytes, fmt_recv_bytes, fmt_stackid,
				      total_time);
	}
}

/**********************************************************************/

void vftr_print_function_stack (FILE *fp, int rank, char *func_name,
		           int n_final_stack_ids, int n_final_func_ids,
			   int *final_stack_ids, int *final_func_ids) {
	long long all_times [vftr_mpisize];
	double imbalances [vftr_func_table_size];
#ifdef _MPI
	for (int fsid = 0; fsid < n_final_stack_ids; fsid++) {
		int function_idx = vftr_gStackinfo[final_stack_ids[fsid]].locID;
		long long t = function_idx >= 0 ? vftr_func_table[function_idx]->prof_current.timeIncl : -1.0;
		PMPI_Allgather (&t, 1, MPI_LONG_LONG_INT,
				all_times, 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);

		if (function_idx >= 0) {
			imbalances[function_idx] = compute_mpi_imbalance (all_times, -1.0);
		}
	}
#else
	for (int i  = 0; i < vftr_func_table_size; i++) {
		imbalances[i] = 0;
	}
#endif
	stack_leaf_t *stack_tree = NULL;
	fprintf (fp, "\n");
	if (n_final_stack_ids == 0) {
		fprintf (fp, "No stack IDs for %s registered.\n", func_name);
		return;
	}
	for (int fsid = 0; fsid < n_final_stack_ids; fsid++) {
		int n_functions_in_stack = vftr_stack_length (final_stack_ids[fsid]);
		int *stack_ids = (int*)malloc (n_functions_in_stack * sizeof(int));
		int stack_id = final_stack_ids[fsid];
		int function_id = vftr_gStackinfo[stack_id].locID;
		for (int i = 0; i < n_functions_in_stack; i++) {
			stack_ids[i] = stack_id;
			stack_id = vftr_gStackinfo[stack_id].ret;
		}
		vftr_fill_into_stack_tree (&stack_tree, n_functions_in_stack, stack_ids, function_id);
		free (stack_ids);
	}
	long long total_time = 0;
	int n_spaces_max = 0;
	double *t_final = (double*) malloc (n_final_stack_ids * sizeof(double));
	int *n_calls_final = (int*) malloc (n_final_stack_ids * sizeof(int));	
	double *imba_final = (double*) malloc (n_final_stack_ids * sizeof(double));
        int n_final = 0;
	vftr_scan_for_final_values (stack_tree->origin, 0, imbalances,
				      &n_spaces_max, &n_final, &t_final, &n_calls_final, &imba_final);	
	int *rank_final_ids = (int*) malloc (n_final_stack_ids * sizeof(int));
	double *t_final_sorted = (double*) malloc (n_final_stack_ids * sizeof(double));
        vftr_sort_double_copy (t_final, n_final, false, &t_final_sorted);
        for (int i = 0; i < n_final; i++) {
	   rank_final_ids[i] = -1;
           double search_for = t_final[i];
           for (int j = 0; j < n_final; j++) {
	      if (t_final_sorted[j] == search_for) {
		 rank_final_ids[i] = j + 1;
		 break;
	      }
           }
        }
               
	vftr_sort_integer (&n_calls_final, n_final, false);
	vftr_sort_double (&imba_final, n_final, false);

	double t_max = t_final_sorted[0];
	int n_calls_max = n_calls_final[0];
	double imba_max = imba_final[0];

        int pos0 = 0;
        vftr_stacktree_assign_positions (stack_tree->origin, &pos0, rank_final_ids); 

	free (rank_final_ids);
        free (t_final);
	free (t_final_sorted);
        free (n_calls_final);
        free (imba_final);

	// We have six digits behind the comma for time values. More do not make sense since we have a resolution
	// of microseconds.
	// We add this value of 6 to the number of digits in front of the comma, plus one for the comma itself.
	int fmt_t = (vftr_count_digits_double(t_max) + 7) > strlen(vftr_stacktree_headers[TIME]) ? vftr_count_digits_double(t_max) + 7: strlen(vftr_stacktree_headers[TIME]);
	int fmt_calls = vftr_count_digits(n_calls_max) > strlen(vftr_stacktree_headers[CALLS]) ? vftr_count_digits(n_calls_max) : strlen(vftr_stacktree_headers[CALLS]);;
	// For percentage values, two decimal points are enough.
	int fmt_imba = (vftr_count_digits_double(imba_max) + 3) > strlen(vftr_stacktree_headers[IMBA]) ? vftr_count_digits_double(imba_max) + 3: strlen(vftr_stacktree_headers[IMBA]);
	int fmt_stackid = vftr_count_digits(vftr_gStackscount) > strlen(vftr_stacktree_headers[STACK_ID]) ?
			 vftr_count_digits(vftr_gStackscount) : strlen(vftr_stacktree_headers[STACK_ID]);
	int fmt_mpi_send = strlen (vftr_stacktree_headers[SEND_BYTES]);
	int fmt_mpi_recv = strlen (vftr_stacktree_headers[RECV_BYTES]);
	vftr_print_stacktree_header (fp, n_final_stack_ids, func_name, n_spaces_max, fmt_calls,
				     fmt_t, fmt_imba, fmt_mpi_send, fmt_mpi_recv, fmt_stackid);
	vftr_print_stacktree (fp, stack_tree->origin, 0, imbalances,
			      n_spaces_max, fmt_calls, fmt_t, fmt_imba, fmt_mpi_send, fmt_mpi_recv, fmt_stackid, &total_time);
	if (vftr_mpirank == 0) {
		vftr_print_html_output (NULL, func_name, stack_tree->origin, imbalances);
	}
	free (stack_tree);
	fprintf (fp, "Total(%s): %lf sec. \n", func_name, (double)total_time * 1e-6);
}

/**********************************************************************/

int vftr_stacks_test_1 (FILE *fp_in, FILE *fp_out) {
	unsigned long long addrs[6];
	fprintf (fp_out, "Initial vftr_stackscount: %d\n", vftr_stackscount);
	function_t *func1 = vftr_new_function (NULL, "init", NULL, 0, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, 0, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, 0, false);	
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, 0, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, 0, false);
	function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, 0, false);
	function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, 0, false);
	vftr_normalize_stacks();			
	fprintf (fp_out, "%s: %d %d\n", func1->name, func1->id, func1->gid);
	fprintf (fp_out, "%s: %d %d\n", func2->name, func2->id, func2->gid);
	fprintf (fp_out, "%s: %d %d\n", func3->name, func3->id, func3->gid);
	fprintf (fp_out, "%s: %d %d\n", func4->name, func4->id, func4->gid);
	fprintf (fp_out, "%s: %d %d\n", func5->name, func5->id, func5->gid);
	fprintf (fp_out, "%s: %d %d\n", func6->name, func6->id, func6->gid);
	fprintf (fp_out, "%s: %d %d\n", func7->name, func7->id, func7->gid);
	fprintf (fp_out, "Global stacklist: \n");
	vftr_print_global_stacklist (fp_out);
	return 0;
}

/**********************************************************************/

int vftr_stacks_test_2 (FILE *fp_in, FILE *fp_out) {
#ifdef _MPI
	unsigned long long addrs[6];
	function_t *func0 = vftr_new_function (NULL, "init", NULL, 0, false);	
	if (vftr_mpirank == 0) {
		function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, 0, false);
		function_t *func2 = vftr_new_function ((void*)(addrs + 1), "func2", func1, 0, false);
		function_t *func3 = vftr_new_function ((void*)(addrs + 2), "func3", func2, 0, false);
		function_t *func4 = vftr_new_function ((void*)(addrs + 3), "func4", func3, 0, false);
	} else if (vftr_mpirank == 1) {
		function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, 0, false);
		function_t *func2 = vftr_new_function ((void*)(addrs + 2), "func3", func1, 0, false);
		function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func2", func2, 0, false);
		function_t *func4 = vftr_new_function ((void*)(addrs + 3), "func4", func3, 0, false);
	} else if (vftr_mpirank == 2) {	
		function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, 0, false);
		function_t *func2 = vftr_new_function ((void*)(addrs + 1), "func2", func1, 0, false);
		function_t *func3 = vftr_new_function ((void*)(addrs + 2), "func2", func2, 0, false);
		function_t *func4 = vftr_new_function ((void*)(addrs + 3), "func2", func3, 0, false);
	} else if (vftr_mpirank == 3) {
		function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, 0, false);
		function_t *func2 = vftr_new_function ((void*)(addrs + 3), "func4", func1, 0, false);
	} else {
		fprintf (fp_out, "Error: Invalid MPI rank (%d)!\n", vftr_mpirank);
		return -1;
	}

	vftr_normalize_stacks();

	// Needs to be set for printing the local stacklist
	vftr_profile_wanted = true;
	for (int i = 0; i < vftr_mpisize; i++) {
		if (vftr_mpirank == i) {
			fprintf (fp_out, "Local stacklist for rank %d: \n", i);
			// There is "init" + the four (rank 0 - 2) or two (rank 3) additional functions.
			int n_functions = vftr_mpirank == 3 ? 3 : 5;
			vftr_print_local_stacklist (vftr_func_table, fp_out, n_functions);
		}
		PMPI_Barrier (MPI_COMM_WORLD);
	}


	if (vftr_mpirank == 0) {
		fprintf (fp_out, "Global stacklist: \n");
		vftr_print_global_stacklist (fp_out);
	}
#endif
	return 0;
}

/**********************************************************************/

