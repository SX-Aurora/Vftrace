#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "symbols.h"

library_t vftr_parse_maps_line(char *line) {
   library_t library = {
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
   if (permissions[2] != 'x') {return library;}
   char *offset = strtok(NULL, " ");
#ifndef __VMAP_OFFSET
   if (strncmp(offset, "00000000", 8)) {return library;}
#endif
   // skip over devide
   strtok(NULL," ");
   // skip over inode
   strtok(NULL," ");
   char *path = strtok(NULL," \n");
   // only continue if a valid path is found
   if (path == NULL || path[0] == '[') {return library;}
   // Filter out devices and system libraries
   // They are not instrumented and can be discarded
   // test for common path names
   if (!strncmp(path, "/dev/", 5)) {return library;}
   if (!strncmp(path, "/usr/lib", 8)) {return library;}
   if (!strncmp(path, "/usr/lib64", 10)) {return library;}
   if (!strncmp(path, "/lib", 4)) {return library;}
#ifdef __ve__
   if (!strncmp(path, "/opt/nec/ve/veos/lib", 20)) {return library;}
#endif
   sscanf(base, "%ld", &(library.base));
   library.path = strdup(path);
#ifdef __VMAP_OFFSET
   library.offset = strtoul(offset, NULL, 16);
#else
   library.offset = 0L;
#endif
   return library;
}

// read the different library paths from the applications map
librarylist_t vftr_read_library_maps() {
   FILE *fmap = fopen("/proc/self/maps", "r");
   if (fmap == NULL) {
      perror ("Opening /proc/self/maps");
      abort();
   }

   librarylist_t librarylist = {
      .nlibraries = 0,
      .maxlibraries = 0,
      .libraries = NULL
   };

   char *lineptr = NULL;
   size_t buffsize = 0;
   ssize_t readbytes = 0;
   // read all lines
   while ((readbytes = getline(&lineptr, &buffsize, fmap)) > 0) {
      library_t library = vftr_parse_maps_line(lineptr);
      if (library.path != NULL) {
         // append it to the librarylist
         int idx = librarylist.nlibraries;
         librarylist.nlibraries++;
         if (librarylist.nlibraries > librarylist.maxlibraries) {
            librarylist.maxlibraries = 1.4*(librarylist.maxlibraries+1);
            librarylist.libraries = (library_t*)
               realloc(librarylist.libraries, librarylist.maxlibraries*sizeof(library_t));
         }
         librarylist.libraries[idx] = library;
#ifdef _DEBUG
         fprintf(stderr, "Found library p=%s (%lu,%lu)\n",
                 librarylist.libraries[librarylist.nlibraries-1].path,
                 librarylist.libraries[librarylist.nlibraries-1].base,
                 librarylist.libraries[librarylist.nlibraries-1].offset);
#endif
      }
   }
   free(lineptr);
   fclose(fmap);
   return librarylist;
}

void vftr_free_librarylist(librarylist_t *librarylist_ptr) {
   librarylist_t librarylist = *librarylist_ptr;
   if (librarylist.nlibraries > 0) {
      for (int ilib=0; ilib<librarylist.nlibraries; ilib++) {
         free(librarylist.libraries[ilib].path);
         librarylist.libraries[ilib].path = NULL;
      }
      free(librarylist.libraries);
      librarylist.libraries = NULL;
   }
}

symbollist_t vftr_read_symbols() {
   symbollist_t symbols;
   // get all library paths that belong to the program
   librarylist_t librarylist = vftr_read_library_maps();

   vftr_free_librarylist(&librarylist);
   return symbols;
}
