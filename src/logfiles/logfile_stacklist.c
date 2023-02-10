#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "table_types.h"
#include "collated_stack_types.h"

#include "collate_stacks.h"
#include "tables.h"

void vftr_write_logfile_global_stack_list(FILE *fp, collated_stacktree_t stacktree) {
   SELF_PROFILE_START_FUNCTION;
   fprintf(fp, "\nGlobal call stacks:\n");

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, stacktree.nstacks);
   vftr_table_left_outline(&table, false);
   vftr_table_right_outline(&table, false);
   vftr_table_columns_separating_line(&table, false);

   // first column with the StackIDs
   int *IDs = (int*) malloc(stacktree.nstacks*sizeof(int));
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      IDs[istack] = istack;
   }
   vftr_table_add_column(&table, col_int, "STID", "STID%d", 'r', 'r', (void*) IDs);

   // second column with the stack strings
   char **stacks = (char**) malloc(stacktree.nstacks*sizeof(char*));
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      stacks[istack] = vftr_get_collated_stack_string(stacktree, istack, false);
   }
   vftr_table_add_column(&table, col_string,
                         "Call stack", "%s", 'l', 'l', (void*) stacks);

   vftr_print_table(fp, table);
   fprintf(fp, "\n");

   vftr_table_free(&table);
   free(IDs);
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      free(stacks[istack]);
   }
   free(stacks);
   SELF_PROFILE_END_FUNCTION;
}
