#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>

#include <link.h>
#include <dlfcn.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "symbol_types.h"
#include "symbols.h"
#include "sorting.h"

static void* (*real_dlopen)(const char *filename, int flag) = NULL;

static void set_real_dlopen() {
   real_dlopen = dlsym(RTLD_NEXT, "dlopen");
   if (real_dlopen == NULL) {
      printf ("dlopen: Internal error\n");
   }
}

void *dlopen(const char *filename, int flag) {
   SELF_PROFILE_START_FUNCTION;
   if (real_dlopen == NULL) {
      set_real_dlopen();
   }
   // load the library
   void *libhandle = real_dlopen(filename, flag);
   // TODO: use dlinfo to get absolute filename from handle
   struct link_map *lib_linkmap_ptr = NULL;
   dlinfo(libhandle, RTLD_DI_LINKMAP, &lib_linkmap_ptr);

   while (lib_linkmap_ptr != NULL) {
      library_t dynlib = {
         .base = lib_linkmap_ptr->l_addr,
         .offset = 0,
         .path = lib_linkmap_ptr->l_name
      };
      symboltable_t dynsymtab = vftr_read_symbols_from_library(dynlib);
      if (dynsymtab.nsymbols > 0) {
         vftr_sort_symboltable(dynsymtab.nsymbols, dynsymtab.symbols);
         vftr_symboltable_determine_preciseness(
            &dynsymtab,
            vftrace.environment.preciseregex.value.regex_val);
         vftr_symboltable_strip_fortran_module_name(
            &dynsymtab,
            vftrace.environment.strip_module_names.value.bool_val);
   #ifdef _LIBERTY
         vftr_symboltable_demangle_cxx_name(
            &dynsymtab,
            vftrace.environment.demangle_cxx.value.bool_val);
   #endif
         vftr_merge_symbol_tables(&(vftrace.symboltable), dynsymtab);
         free(dynsymtab.symbols);
      }
      lib_linkmap_ptr = lib_linkmap_ptr->l_next;
   }
   SELF_PROFILE_END_FUNCTION;
   return libhandle;
}
