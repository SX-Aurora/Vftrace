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
#include "vftr_browse.h"
#include "vftr_filewrite.h"
#include "vftr_fileutils.h"
#include "vftr_sorting.h"

// number of locally unique stacks
int vftr_stackscount = 0;
// number of globally unique stacks
int vftr_gStackscount = 0;

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

stack_string_t *vftr_global_stack_strings;

/**********************************************************************/

// initialize stacks only called from vftr_initialize
void vftr_initialize_stacks() {
   // Allocate stack tables for each thread
   vftr_fstack = (function_t*) malloc(sizeof(function_t));
   vftr_froots = (function_t*) malloc(sizeof(function_t));

   // Initialize stack tables 
   char *s = "init";
   function_t *func = vftr_new_function (NULL, strdup (s), NULL, true);
   func->next_in_level = func; /* Close circular linked list to itself */
   vftr_fstack = func;
   vftr_function_samplecount = 0;
   vftr_message_samplecount = 0;
   vftr_froots = func;
}

/**********************************************************************/

// synchronise the global stack IDs among different processes
void vftr_normalize_stacks() {
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
    for (int istack = 0; istack < vftr_stackscount; istack++) {
       for (int ihash = 0; ihash < vftr_gStackscount; ihash++) {
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
       for (int istack = 0; istack < vftr_gStackscount; istack++) {
          vftr_gStackinfo[istack].ret = -2;
          vftr_gStackinfo[istack].name = NULL;
          vftr_gStackinfo[istack].locID = -1;
	  vftr_gStackinfo[istack].print_profile = false;
       }
       // fill in global info process 0 knows
       for (int istack=0; istack<vftr_stackscount; istack++) {
          int globID = local2global_ID[istack];
          vftr_gStackinfo[globID].name = strdup(vftr_func_table[istack]->name);
          // TODO: is this used?
	  if (vftr_environment.print_stack_profile->set) {
		if (vftr_pattern_match (vftr_environment.print_stack_profile->value, 
				        vftr_func_table[istack]->name)) { 
			vftr_gStackinfo[globID].print_profile = true;
	        } // else if match stack id
   	  }
          if (strcmp(vftr_gStackinfo[globID].name, "init")) {
             // not the init function
             vftr_gStackinfo[globID].ret = vftr_func_table[istack]->return_to->gid;
          } else {
             vftr_gStackinfo[globID].ret = -1;
          }
          vftr_gStackinfo[globID].locID = istack;

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
       for (int irank = 1; irank < vftr_mpisize; irank++) {
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
                for (int istack = 0; istack < hasnmissing; istack++) {
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
             for (int istack = 0; istack<nmissing; istack++) {
                int globID = missingStacks[istack];
                int locID = global2local_ID[globID];
                if (locID >= 0) {
                   missingStackInfo[3 * imatch + 0] = globID;
                   missingStackInfo[3 * imatch + 1] = vftr_func_table[locID]->return_to->gid;
                   // add one to length due to null terminator
                   missingStackInfo[3 * imatch + 2] = strlen(vftr_func_table[locID]->name) + 1;
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
             for (int istack = 0; istack < hasnmissing; istack++) {
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
    if (vftr_env_distribute_gStack()) {
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

void vftr_print_local_stacklist (function_t **funcTable, FILE *fp, int n_ids) {
    bool use_gid = vftr_gStackinfo != NULL;
    
    if (!vftr_profile_wanted) return;

    // Compute column and table widths
    int max_width = 0;
    int max_id = 0;
    for (int i = 0; i < n_ids; i++) {
        function_t *func = funcTable[i];
        if (func == NULL || !func->return_to) continue;
        int width, id;
        id = use_gid ? func->gid : func->id;
        for (width = 0; func; func = func->return_to) {
            width += strlen (func->name) + 1;
	}
        if (max_width < width) max_width = width;
        if (max_id < id) max_id = id;
    }

    if (strlen("Functions") > max_width) max_width = strlen("Functions");
    int max_id_length = strlen("ID") > vftr_count_digits_int(max_id) ? strlen("ID") : vftr_count_digits_int(max_id);
    int table_width = 1 + max_id_length + 1 + max_width;

    // Print headers

    fprintf (fp, "Local call stacks:\n");
    vftr_print_dashes (fp, table_width);
    fprintf (fp, " %*s %*s\n", max_id_length, "ID", max_width, "Functions");
    vftr_print_dashes (fp, table_width);

    // Print table

    for (int i = 0; i < n_ids; i++) {
        char *sep; 
        int  id;
        function_t *func = funcTable[i];
        if (func == NULL || !func->return_to) continue; // If not defined or no caller
	id = use_gid ? func->gid : func->id;
        fprintf (fp, " %*d ", max_id_length, id); 
        for (sep=""; func; func = func->return_to, sep="<") {
           fprintf (fp, "%s%s", sep, func->name);
        }
        fprintf (fp, "\n");
    }
    vftr_print_dashes (fp, table_width);
}

/**********************************************************************/

void vftr_create_global_stack_strings () {
   vftr_global_stack_strings = (stack_string_t*) malloc (vftr_gStackscount * sizeof(stack_string_t));
   int i_stack = 0;
   for (int i = 0; i < vftr_gStackscount; i++) {
      if (vftr_gStackinfo[i].locID >= 0) {
        char *name;
        int len;
        int depth;
        vftr_create_stack_string (i, &name, &len, &depth);
	vftr_global_stack_strings[i_stack].s = strdup(name);
        vftr_global_stack_strings[i_stack].len = len;
        vftr_global_stack_strings[i_stack].id = i;
        vftr_global_stack_strings[i_stack].depth = depth;
        i_stack++;
      } else {
        vftr_global_stack_strings[i_stack++].id = -1;
      }
   }
}

/**********************************************************************/

void vftr_create_stack_string (int i_stack, char **name, int *len, int *depth) {
   // First, count how long the string will be.
   *len = 0;
   int j_stack = i_stack;
   while (vftr_gStackinfo[j_stack].locID >= 0 && vftr_gStackinfo[j_stack].ret >= 0) {
      // In the final string, there will be an additional "<" character. Therefore, add 1.
      *len += strlen(vftr_gStackinfo[j_stack].name) + 1; 
      j_stack = vftr_gStackinfo[j_stack].ret;
   }
   *len += strlen(vftr_gStackinfo[j_stack].name); 
   // extra char for null terminator
   char *stack_string = (char *) malloc ((*len + 1)* sizeof(char));
   *name = stack_string;
   j_stack = i_stack;
   while (vftr_gStackinfo[j_stack].locID >= 0 && vftr_gStackinfo[j_stack].ret >= 0) {
      char *s = vftr_gStackinfo[j_stack].name;
      int n = strlen(s);
      for (int i = 0; i < n; i++) {
        *stack_string = s[i];
        stack_string++;
      }
      *stack_string = '<';
      stack_string++;
      j_stack = vftr_gStackinfo[j_stack].ret;
   }
   char *s = vftr_gStackinfo[j_stack].name;
   int n = strlen(s);
   for (int i = 0; i < n; i++) {
      *stack_string = s[i];
      stack_string++;
   }
   *stack_string = '\0';
}

/**********************************************************************/

void vftr_print_global_stacklist (FILE *fp) {

   // Compute column and table widths
   // loop over all stacks to find the longest one
   int maxstrlen = 0;
   for (int i_stack = 0; i_stack < vftr_gStackscount; i_stack++) {
      if (vftr_global_stack_strings[i_stack].id >= 0) {
         int this_length = vftr_global_stack_strings[i_stack].len;
         if (this_length > maxstrlen) maxstrlen = this_length;
      }
   }
   maxstrlen--; // Chop trailing space
   if (strlen("Functions") > maxstrlen) maxstrlen = strlen("Functions");
   int max_id = vftr_gStackscount;
   // Each stack ID in this list is prefixed with "STID" to allow for an easier search of that line.
   // This has always more characters than the header "ID". Therefore, max_id_length is just
   // the length of "STID" (4) plus the length of the maximal ID.
   int max_id_length = 4 + vftr_count_digits_int(max_id) + 1;
   int table_width = 1 + max_id_length + 1 + maxstrlen;

   // Print headers

   fprintf (fp, "Global call stacks:\n");
   vftr_print_dashes (fp, table_width);
   fprintf (fp, " %*s %*s\n", max_id_length, "ID", maxstrlen, "Functions");
   vftr_print_dashes (fp, table_width);

   // Print table

   for (int i_stack = 0; i_stack < vftr_gStackscount; i_stack++) {
      if (vftr_global_stack_strings[i_stack].id >= 0) {
         char sid[max_id_length];
         snprintf (sid, max_id_length, "STID%d", vftr_global_stack_strings[i_stack].id);
         fprintf (fp, "%*s %s\n", max_id_length, sid, vftr_global_stack_strings[i_stack].s);
      }
   }

   vftr_print_dashes (fp, table_width);
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
				   int *n_spaces_max, int *n_chars_max, int *n_final, double **t_final, int **n_calls_final, double **imba_final) {
	if (!leaf) return;
	if (leaf->callee) {
		int new_n_spaces = this_n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name);
		if (this_n_spaces > 0) new_n_spaces++;
		vftr_scan_for_final_values (leaf->callee, new_n_spaces, imbalances,
					      n_spaces_max, n_chars_max, n_final, t_final, n_calls_final, imba_final);
	} else {
		int new_n_spaces = this_n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name) + 1; // + 1 for the colon at the end
		if (this_n_spaces > 0) new_n_spaces++;
		if (new_n_spaces > *n_spaces_max) *n_spaces_max = new_n_spaces;
		   int this_n_chars = leaf->func_id > 0 ? strlen(vftr_func_table[leaf->func_id]->name) : strlen("[not on this rank]");
		   if (this_n_chars > *n_chars_max) *n_chars_max = this_n_chars;
		if (leaf->func_id > 0) {
		   double this_t = (double)vftr_func_table[leaf->func_id]->prof_current.time_incl * 1e-6;
		   (*t_final)[*n_final] = this_t;
		   int this_n_calls = vftr_func_table[leaf->func_id]->prof_current.calls;
		   (*n_calls_final)[*n_final] = this_n_calls;
		   if (imbalances) (*imba_final)[*n_final] = imbalances[leaf->func_id];
		   *n_final = *n_final + 1;
		}
	}
	if (leaf->next_in_level) {
		vftr_scan_for_final_values (leaf->next_in_level, this_n_spaces, imbalances,
					    n_spaces_max, n_chars_max, n_final, t_final, n_calls_final, imba_final);
	}
}

/**********************************************************************/

void vftr_stacktree_assign_positions (stack_leaf_t *leaf, int *pos, int *indices) {
   if (!leaf) return;
   if (leaf->callee) {
      vftr_stacktree_assign_positions (leaf->callee, pos, indices);
   } else if (leaf->func_id > 0) {
      leaf->final_id = indices[*pos];
      *pos = *pos + 1;	
   }
   if (leaf->next_in_level) {
      vftr_stacktree_assign_positions (leaf->next_in_level, pos, indices); 
   }
}
   
/**********************************************************************/

void vftr_print_stacktree_header (FILE *fp, int n_stacks, char *func_name,
				  int *n_spaces_max, int fmt_calls, int fmt_t, int fmt_imba,
				  int fmt_send_bytes, int fmt_recv_bytes, int fmt_stackid,
				  int *n_char_tot) {
	int fmt_position = strlen("position");
	char title[128];
	sprintf (title, "Function stacks leading to %s: %d", func_name, n_stacks);
	fprintf (fp, "%s", title);
	if (*n_spaces_max < strlen(title)) *n_spaces_max = strlen(title);
	*n_char_tot = *n_spaces_max + fmt_calls + fmt_t + fmt_imba + fmt_send_bytes + fmt_recv_bytes + fmt_stackid + fmt_position + 21;
	for (int i = 0; i < *n_spaces_max - strlen(title); i++) fprintf (fp, " ");
	fprintf (fp, "   %*s   %*s   %*s   %*s   %*s   %*s   %*s\n", fmt_t, vftr_stacktree_headers[TIME],
		 fmt_calls, vftr_stacktree_headers[CALLS], fmt_imba, vftr_stacktree_headers[IMBA],
		 fmt_send_bytes, vftr_stacktree_headers[SEND_BYTES], fmt_recv_bytes, vftr_stacktree_headers[RECV_BYTES],
		 fmt_stackid, vftr_stacktree_headers[STACK_ID], (int)strlen("position"), "position");
}

/**********************************************************************/

void vftr_print_stacktree (FILE *fp, stack_leaf_t *leaf, int n_spaces, double *imbalances, 
			   int n_spaces_max, int fmt_calls, int fmt_t, int fmt_imba,
		           int fmt_send_bytes, int fmt_recv_bytes, int fmt_stackid) {
	if (!leaf) return;
	fprintf (fp, "%s", vftr_gStackinfo[leaf->stack_id].name);
	if (leaf->callee) {
		fprintf (fp, ">");
		int new_n_spaces = n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name);
		if (n_spaces > 0) new_n_spaces++;
		vftr_print_stacktree (fp, leaf->callee, new_n_spaces, imbalances,
				      n_spaces_max, fmt_calls, fmt_t, fmt_imba,
			              fmt_send_bytes, fmt_recv_bytes, fmt_stackid);
	} else {
		if (leaf->func_id < 0) {
			fprintf (fp, "[not on this rank]\n");
		} else {
		        fprintf (fp, ":");
			for (int i = n_spaces + strlen(vftr_gStackinfo[leaf->stack_id].name) + 2; i < n_spaces_max; i++) {
			   fprintf (fp, " ");
			}
			char *send_unit_str, *recv_unit_str;
		        double mpi_tot_send_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_send_bytes;
		        double mpi_tot_recv_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_recv_bytes;
			vftr_memory_unit (&mpi_tot_send_bytes, &send_unit_str);	
			vftr_memory_unit (&mpi_tot_recv_bytes, &recv_unit_str);	
			fprintf (fp, "   %*.6f   %*lld   %*.2f   %*.lf %s   %*.lf %s   %*d   %*d\n",
				 fmt_t, (double)vftr_func_table[leaf->func_id]->prof_current.time_incl * 1e-6,
				 fmt_calls, vftr_func_table[leaf->func_id]->prof_current.calls,
				 fmt_imba, imbalances[leaf->func_id],
			 	 fmt_send_bytes - 4, mpi_tot_send_bytes, send_unit_str,
				 fmt_recv_bytes - 4, mpi_tot_recv_bytes, recv_unit_str,
				 fmt_stackid, leaf->stack_id, (int)strlen("position"), leaf->final_id);
		}
	}
	if (leaf->next_in_level) {
		for (int i = 0; i < n_spaces; i++) fprintf (fp, " ");
		fprintf (fp, ">");
		vftr_print_stacktree (fp, leaf->next_in_level, n_spaces, imbalances,
				      n_spaces_max, fmt_calls, fmt_t, fmt_imba,
				      fmt_send_bytes, fmt_recv_bytes, fmt_stackid);
	}
}

/**********************************************************************/

void vftr_create_stacktree (stack_leaf_t **stack_tree, int n_final_stack_ids, int *final_stack_ids) {
   for (int fsid = 0; fsid < n_final_stack_ids; fsid++) {
   	int n_functions_in_stack = vftr_stack_length (final_stack_ids[fsid]);
   	int *stack_ids = (int*)malloc (n_functions_in_stack * sizeof(int));
   	int stack_id = final_stack_ids[fsid];
   	int function_id = vftr_gStackinfo[stack_id].locID;
   	for (int i = 0; i < n_functions_in_stack; i++) {
   		stack_ids[i] = stack_id;
   		stack_id = vftr_gStackinfo[stack_id].ret;
   	}
   	vftr_fill_into_stack_tree (stack_tree, n_functions_in_stack, stack_ids, function_id);
   	free (stack_ids);
   }
}

/**********************************************************************/

void vftr_stack_compute_imbalances (double *imbalances, int n_final_stack_ids, int *final_stack_ids) {
#if defined(_MPI)
	long long all_times [vftr_mpisize];
	for (int fsid = 0; fsid < n_final_stack_ids; fsid++) {
		int function_idx = vftr_gStackinfo[final_stack_ids[fsid]].locID;
		long long t = function_idx >= 0 ? vftr_func_table[function_idx]->prof_current.time_incl : -1.0;
		PMPI_Allgather (&t, 1, MPI_LONG_LONG_INT,
				all_times, 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);

		if (function_idx >= 0) {
			imbalances[function_idx] = vftr_compute_mpi_imbalance (all_times, -1.0);
		}
	}
#else
	for (int i  = 0; i < vftr_func_table_size; i++) {
		imbalances[i] = 0;
	}
#endif

}

/**********************************************************************/

void vftr_stack_get_total_time (stack_leaf_t *leaf, long long *total_time) {
   if (!leaf) return;
   if (leaf->callee) {
	vftr_stack_get_total_time (leaf->callee, total_time);  
   } else {
	if (leaf->func_id >= 0) {
	   *total_time += vftr_func_table[leaf->func_id]->prof_current.time_incl;
	}
   }
   if (leaf->next_in_level) vftr_stack_get_total_time (leaf->next_in_level, total_time);
}

/**********************************************************************/

void vftr_scan_stacktree (stack_leaf_t *stack_tree, int n_final_stack_ids, double *imbalances,
			  double *t_max, int *n_calls_max, double *imba_max, int *n_spaces_max, int *n_chars_max) {
   *n_spaces_max = 0;
   *n_chars_max = 0;
   double *t_final = (double*) malloc (n_final_stack_ids * sizeof(double));
   int *n_calls_final = (int*) malloc (n_final_stack_ids * sizeof(int));	
   double *imba_final = (double*) malloc (n_final_stack_ids * sizeof(double));
   int n_final = 0;
   vftr_scan_for_final_values (stack_tree->origin, 0, imbalances,
   		               n_spaces_max, n_chars_max, &n_final, &t_final, &n_calls_final, &imba_final);	

   int *rank_final_ids = (int*) malloc (n_final_stack_ids * sizeof(int));
   double *t_final_sorted = (double*) malloc (n_final_stack_ids * sizeof(double));
   vftr_sort_double_copy (t_final, n_final, false, t_final_sorted);
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
          
   vftr_sort_integer (n_calls_final, n_final, false);
   vftr_sort_double (imba_final, n_final, false);

   *t_max = t_final_sorted[0];
   *n_calls_max = n_calls_final[0];
   *imba_max = imba_final[0];

   int pos0 = 0;
   vftr_stacktree_assign_positions (stack_tree->origin, &pos0, rank_final_ids); 

   free (rank_final_ids);
   free (t_final);
   free (t_final_sorted);
   free (n_calls_final);
   free (imba_final);
}

/**********************************************************************/

void vftr_print_function_stack (FILE *fp, char *func_name, int n_final_stack_ids,
			        double *imbalances, long long total_time,
			        double t_max, int n_calls_max, double imba_max, int n_spaces_max, 
			        stack_leaf_t *stack_tree) {
	fprintf (fp, "\n");
	if (n_final_stack_ids == 0) {
		fprintf (fp, "No stack IDs for %s registered.\n", func_name);
		return;
	}
	// We have six digits behind the comma for time values. More do not make sense since we have a resolution
	// of microseconds.
	// We add this value of 6 to the number of digits in front of the comma, plus one for the comma itself.
	int fmt_t = (vftr_count_digits_double(t_max) + 7) > strlen(vftr_stacktree_headers[TIME]) ? vftr_count_digits_double(t_max) + 7: strlen(vftr_stacktree_headers[TIME]);
	int fmt_calls = vftr_count_digits_int(n_calls_max) > strlen(vftr_stacktree_headers[CALLS]) ? vftr_count_digits_int(n_calls_max) : strlen(vftr_stacktree_headers[CALLS]);;
	// For percentage values, two decimal points are enough.
	int fmt_imba = (vftr_count_digits_double(imba_max) + 3) > strlen(vftr_stacktree_headers[IMBA]) ? vftr_count_digits_double(imba_max) + 3: strlen(vftr_stacktree_headers[IMBA]);
	int fmt_stackid = vftr_count_digits_int(vftr_gStackscount) > strlen(vftr_stacktree_headers[STACK_ID]) ?
			 vftr_count_digits_int(vftr_gStackscount) : strlen(vftr_stacktree_headers[STACK_ID]);
	int fmt_mpi_send = strlen (vftr_stacktree_headers[SEND_BYTES]);
	int fmt_mpi_recv = strlen (vftr_stacktree_headers[RECV_BYTES]);
        int n_char_tot;
	vftr_print_stacktree_header (fp, n_final_stack_ids, func_name, &n_spaces_max, fmt_calls,
				     fmt_t, fmt_imba, fmt_mpi_send, fmt_mpi_recv, fmt_stackid, &n_char_tot);
	vftr_print_dashes (fp, n_char_tot);
	vftr_print_stacktree (fp, stack_tree->origin, 0, imbalances,
			      n_spaces_max, fmt_calls, fmt_t, fmt_imba, fmt_mpi_send, fmt_mpi_recv, fmt_stackid);
	vftr_print_dashes (fp, n_char_tot);
	fprintf (fp, "Total(%s): %lf sec. \n", func_name, (double)total_time * 1e-6);
}

/**********************************************************************/
