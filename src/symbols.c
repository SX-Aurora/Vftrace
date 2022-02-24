#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "symbols.h"

path_t vftr_parse_library_path(char *line) {
   path_t libpath = {
      .base = 0,
      .offset = 0,
      .path = NULL
   };
   // a line has the following format:
   // <baseaddr>-<topaddr> <permissions> <offset> <device> <inode> <libpath>
   // split the string by tokens
   char *base = strtok(line,"-");
   // skip over topaddr
   strtok(NULL," ");
   char *permissions = strtok(NULL," ");
   // only continue parsing if the library is marked executable
   if (permissions[2] != 'x') {return libpath;}
   char *offset = strtok(NULL, " ");
#ifndef __VMAP_OFFSET
   if (strncmp(offset, "00000000", 8)) {return libpath;}
#endif
   // skip over devide
   strtok(NULL," ");
   // skip over inode
   strtok(NULL," ");
   char *path = strtok(NULL," \n");
   // only continue if a valid path is found
   if (path == NULL || path[0] == '[') {return libpath;}
   // Filter out devices and system libraries
   // They are not instrumented and can be discarded
   // test for common path names
   if (!strncmp(path, "/dev/", 5)) {return libpath;}
   if (!strncmp(path, "/usr/lib", 8)) {return libpath;}
   if (!strncmp(path, "/usr/lib64", 10)) {return libpath;}
   if (!strncmp(path, "/lib", 4)) {return libpath;}
#ifdef __ve__
   if (!strncmp(path, "/opt/nec/ve/veos/lib", 20)) {return libpath;}
#endif
   sscanf(base, "%ld", &(libpath.base));
   libpath.path = strdup(path);
#ifdef __VMAP_OFFSET
   libpath.offset = strtoul(offset, NULL, 16);
#else
   libpath.offset = 0L;
#endif
   return libpath;
}

// read the different library paths from the applications map
pathlist_t vftr_read_library_paths() {
   FILE *fmap = fopen("/proc/self/maps", "r");
   if (fmap == NULL) {
      perror ("Opening /proc/self/maps");
      abort();
   }

   pathlist_t pathlist = {
      .npaths = 0,
      .maxpaths = 0,
      .paths = NULL
   };

   char *lineptr = NULL;
   size_t buffsize = 0;
   ssize_t readbytes = 0;
   // read all lines
   while ((readbytes = getline(&lineptr, &buffsize, fmap)) > 0) {
      path_t libpath = vftr_parse_library_path(lineptr);
      if (libpath.path != NULL) {
         // append it to the pathlist
         int idx = pathlist.npaths;
         pathlist.npaths++;
         if (pathlist.npaths > pathlist.maxpaths) {
            pathlist.maxpaths = 1.4*(pathlist.maxpaths+1);
            pathlist.paths = (path_t*)
               realloc(pathlist.paths, pathlist.maxpaths*sizeof(path_t));
         }
         pathlist.paths[idx] = libpath;
#ifdef _DEBUG
         fprintf(stderr, "Found library path b=%lu o=%lu p=%s\n",
                 pathlist.paths[pathlist.npaths-1].base,
                 pathlist.paths[pathlist.npaths-1].offset,
                 pathlist.paths[pathlist.npaths-1].path);
#endif
      }
   }
   free(lineptr);
   fclose(fmap);
   return pathlist;
}

void vftr_free_pathlist(pathlist_t *pathlist_ptr) {
   pathlist_t pathlist = *pathlist_ptr;
   if (pathlist.npaths > 0) {
      for (int ipath=0; ipath<pathlist.npaths; ipath++) {
         free(pathlist.paths[ipath].path);
         pathlist.paths[ipath].path = NULL;
      }
      free(pathlist.paths);
      pathlist.paths = NULL;
   }
}

symbollist_t vftr_read_symbols() {
   symbollist_t symbols;
   pathlist_t pathlist = vftr_read_library_paths();

   vftr_free_pathlist(&pathlist);
   return symbols;
}
