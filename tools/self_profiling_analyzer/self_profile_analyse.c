#include <stdlib.h>
#include <stdio.h>

#include "event_types.h"
#include "events.h"
#include "function_types.h"
#include "functions.h"
#include "stack_types.h"
#include "stacks.h"
#include "timer.h"

int main(int argc, char **argv) {
   if (argc < 2) {
      fprintf(stderr, "%s <filename>\n", argv[0]);
      exit(EXIT_FAILURE);
   }

   char *filename = argv[1];
   FILE *stream = fopen(filename, "r");
   if (stream == NULL) {
      perror("fopen");
      exit(EXIT_FAILURE);
   }

   functionlist_t functionlist = new_functionlist();
   stacktree_t stacktree = new_stacktree(functionlist);
   int current_stack = 0;


   char *line = NULL;
   int nread = 0;
   event_t event;
   while ((nread = get_event(&event, stream)) != -1) {
      // print_event(stdout, event);
      // get function index
      int fidx = function_index_from_functionlist_by_name(event.name, &functionlist);
      // get stack index
      int sidx = -1;
      if (event.action == enter) {
         sidx = search_callee(stacktree.stacks,
                              current_stack,
                              functionlist.functions[fidx].name);
         if (sidx < 0) {
            sidx = new_stack(current_stack,
                             functionlist.functions[fidx].name,
                             &stacktree);
         }
         vftr_stack_t *stack = &(stacktree.stacks[sidx]);
         stack->ncalls++;
         stack->t_enter.tv_sec = event.t_sec;
         stack->t_enter.tv_nsec = event.t_nsec;
      } else if (event.action == leave) {
         sidx = current_stack;
         vftr_stack_t *stack = &(stacktree.stacks[sidx]);
         stack->t_leave.tv_sec = event.t_sec;
         stack->t_leave.tv_nsec = event.t_nsec;
         long long delta_t_nsec = time_diff_nsec(stack->t_enter, stack->t_leave);
         stack->time_nsec += delta_t_nsec;
         sidx =  stacktree.stacks[current_stack].caller;
      }
      free(event.name);
      current_stack = sidx;
   }

   finalize_stacktree(&stacktree);
   vftr_stack_t **sortedstacklist = sort_stacks_by_excl_time(stacktree);

   fprintf(stdout, "\n");
   fprintf(stdout, "Functionlist:\n");
   print_functionlist(stdout, functionlist);
   fprintf(stdout, "\n");
   fprintf(stdout, "Stacktree:\n");
   print_stacktree(stdout, stacktree);
   fprintf(stdout, "\n");
   fprintf(stdout, "Stacklist:\n");
   print_sorted_stacklist(stdout, sortedstacklist, stacktree);

   fclose(stream);
   free(line);

   free(sortedstacklist);
   free_functionlist(&functionlist);
   free_stacktree(&stacktree);
   exit(EXIT_SUCCESS);
}
