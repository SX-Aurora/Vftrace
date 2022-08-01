#define _GNU_SOURCE

#include <stdlib.h>

#include <dlfcn.h>

static void* (*real_dlopen)(const char *filename, int flag) = NULL;

static void set_real_dlopen() {
   real_dlopen = dlsym(RTLD_NEXT, "dlopen");
   if (real_dlopen == NULL) {
      printf ("dlopen: Internal error\n");
   }
}

void *dlopen(const char *filename, int flag) {
   if (real_dlopen == NULL) {
      set_real_dlopen();
   }
   // load the library
   void *libhandle = real_dlopen(filename, flag);
   // TODO: use dlinfo to get absolute filename from handle
   // TODO: read the symboltable
   // TODO: check preciseness
   // TODO: strip fortran module names
   // TODO: cxx demangling
   // TODO: merge with big symbol table
   return libhandle;
}
