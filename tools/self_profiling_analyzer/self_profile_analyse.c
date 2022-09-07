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
   size_t len = 0;
   int nread = 0;
   while ((nread = getline(&line, &len, stream)) != -1) {
      //printf("%s", line);
      event_t event = event_from_line(line);
      print_event(stdout, event);
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
         stack_t *stack = &(stacktree.stacks[sidx]);
         stack->ncalls++;
         stack->t_enter.tv_sec = event.t_sec;
         stack->t_enter.tv_nsec = event.t_nsec;
      } else if (event.action == leave) {
         sidx = current_stack;
         stack_t *stack = &(stacktree.stacks[sidx]);
         stack->t_leave.tv_sec = event.t_sec;
         stack->t_leave.tv_nsec = event.t_nsec;
         long long delta_t_usec = time_diff_usec(stack->t_enter, stack->t_leave);
         stack->time_usec += delta_t_usec;
         sidx =  stacktree.stacks[current_stack].caller;
      }
      current_stack = sidx;
   }

   finalize_stacktree(&stacktree);
   stack_t **sortedstacklist = sort_stacks_by_excl_time(stacktree);

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
