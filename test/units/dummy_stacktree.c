#include <stdlib.h>
#include <string.h>

#include <ctype.h>

#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "callprofiling.h"
#include "hashing.h"

static uintptr_t base_addr = 123456;

stacktree_t vftr_init_dummy_stacktree (uint64_t t_call, uint64_t t_overhead) {
   stacktree_t stacktree = vftr_new_stacktree();
   profile_t *profile = stacktree.stacks[0].profiling.profiles;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, t_call);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), t_overhead);

   return stacktree;
}

int vftr_count_functions_in_stackstr(char *stackstr) {
   int count = 1; // 1 for init, which should always be present.
   char *tmpstackstr = stackstr;
   while (*tmpstackstr != '\0') {
      count += *tmpstackstr == '<';
      tmpstackstr++;
   }
   return count;
}

char **vftr_split_string_in_functions(char *stackstr, int nfuncs) {
   char **funclist = (char**) malloc(nfuncs*sizeof(char*));
   funclist[nfuncs-1] = strtok(stackstr, "<");
   for (int ifkt=nfuncs-2; ifkt>=0; ifkt--) {
      funclist[ifkt] = strtok(NULL, "<");
   }
   return funclist;
}

int vftr_get_offset_from_function_name(char *name) {
   while (!isdigit(*name) && *name != '\0') {name++;}
   if (*name == '\0') {return -1;}
   return atoi(name);
}

int vftr_get_index_from_functionlist(int nfunc, char **funclist,
                                     stacktree_t *stacktree_ptr) {
   if (strcmp(stacktree_ptr->stacks[0].name, funclist[0])) {
      fprintf(stderr, "Wrong Stack start %s! Should be %s\n",
              funclist[0], stacktree_ptr->stacks[0].name);
      return -1;
   }

   int idx = 0;
   for (int ifunc=1; ifunc<nfunc; ifunc++) {
      vftr_stack_t *parent = stacktree_ptr->stacks+idx;
      // check if the callees of the parent functions
      int calleeidx = -1;
      for (int icallee=0; icallee<parent->ncallees; icallee++) {
         vftr_stack_t *callee = stacktree_ptr->stacks+parent->callees[icallee];
         if (!strcmp(callee->name, funclist[ifunc])) {
            calleeidx = callee->lid;
            break;
         }
      }

      // callee not found
      if (calleeidx < 0) {
         int offset = vftr_get_offset_from_function_name(funclist[ifunc]);
         idx = vftr_new_stack(idx, stacktree_ptr,
                              funclist[ifunc], funclist[ifunc],
                              base_addr+offset, false);
      } else {
         idx = calleeidx;
      }
   }
   return idx;
}

void vftr_register_dummy_stack(stacktree_t *stacktree_ptr,
                               char *stackstring,
                               int thread_id,
                               uint64_t t_call,
                               uint64_t t_overhead) {
   char *stackstr = strdup(stackstring);
   // Disect stack into its functions
   int nfuncs = vftr_count_functions_in_stackstr(stackstr);
   char **funclist = vftr_split_string_in_functions(stackstr, nfuncs);

   int idx = vftr_get_index_from_functionlist(nfuncs, funclist, stacktree_ptr);

   free(stackstr);
   free(funclist);

   vftr_stack_t *my_stack = stacktree_ptr->stacks+idx;
   thread_t my_thread = {.threadID = thread_id};

   profile_t *profile = vftr_get_my_profile(my_stack, &my_thread);
   vftr_accumulate_callprofiling(&(profile->callprof), 1, t_call);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), t_overhead);
}
