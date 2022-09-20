#ifndef SELF_PROFILE_H
#define SELF_PROFILE_H

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#ifdef _SELF_PROFILE_VFTRACE
extern char *vftr_self_prof_filename;
extern FILE *vftr_self_prof_iohandle;
extern char *vftr_self_prof_iobuffer;

#define GET_SELF_PROF_FILENAME \
   do { \
      int pid = getpid(); \
      char *filename_base = "vftrace_self_profile-"; \
      int total_str_len = snprintf(NULL, 0, "%s%d", filename_base, pid) + 1; \
      vftr_self_prof_filename = (char*) malloc(total_str_len*sizeof(char)); \
      snprintf(vftr_self_prof_filename, \
               total_str_len, "%s%d", \
               filename_base, pid); \
   } while(0)

#define GET_SELF_PROF_FILEHANDLE \
   do { \
      vftr_self_prof_iohandle = \
         fopen(vftr_self_prof_filename, "w"); \
      size_t bufsize = 256*1024*1024; \
      vftr_self_prof_iobuffer = (char*) malloc(bufsize); \
      memset((void*) vftr_self_prof_iobuffer, 0, bufsize); \
      int status = setvbuf(vftr_self_prof_iohandle, \
                           vftr_self_prof_iobuffer, \
                           _IOFBF, bufsize); \
   } while(0)

#define INIT_SELF_PROF_VFTRACE \
   do { \
      GET_SELF_PROF_FILENAME; \
      GET_SELF_PROF_FILEHANDLE; \
   } while(0)

#define FREE_SELF_PROF_FILENAME \
   do { \
      if (vftr_self_prof_filename != NULL) { \
         free(vftr_self_prof_filename); \
         vftr_self_prof_filename = NULL; \
      } \
   } while(0)

#define CLOSE_SELF_PROF_FILEHANDLE \
   do { \
      if (vftr_self_prof_iohandle != NULL) { \
         fclose(vftr_self_prof_iohandle); \
         vftr_self_prof_iohandle = NULL; \
         free(vftr_self_prof_iobuffer); \
         vftr_self_prof_iobuffer = NULL; \
      } \
   } while(0)
   
#define FINALIZE_SELF_PROF_VFTRACE \
   do { \
      CLOSE_SELF_PROF_FILEHANDLE; \
      FREE_SELF_PROF_FILENAME; \
   } while(0)

#define GET_TIMESTAMP \
   struct timespec timestamp; \
   clock_gettime(CLOCK_MONOTONIC, &timestamp);

#define SELF_PROFILE_WRITE_FUNCTION_NAME \
   do { \
      char *tmpptr = __FUNCTION__; \
      while (*tmpptr) { \
         putc(*tmpptr, vftr_self_prof_iohandle); \
         tmpptr++; \
      } \
      putc('\0', vftr_self_prof_iohandle); \
   } while(0)

#define SELF_PROFILE_WRITE_TIMESTAMP \
   do { \
      fwrite(&timestamp, \
             sizeof(struct timespec), 1, \
             vftr_self_prof_iohandle); \
   } while(0)

#define SELF_PROFILE_START_FUNCTION \
   do { \
      GET_TIMESTAMP; \
      putc('E', vftr_self_prof_iohandle); \
      SELF_PROFILE_WRITE_FUNCTION_NAME; \
      SELF_PROFILE_WRITE_TIMESTAMP; \
   } while(0)

#define SELF_PROFILE_END_FUNCTION \
   do { \
      GET_TIMESTAMP; \
      putc('L', vftr_self_prof_iohandle); \
      SELF_PROFILE_WRITE_FUNCTION_NAME; \
      SELF_PROFILE_WRITE_TIMESTAMP; \
   } while(0)

#else

#define INIT_SELF_PROF_VFTRACE
#define SELF_PROFILE_START_FUNCTION
#define SELF_PROFILE_END_FUNCTION
#define FINALIZE_SELF_PROF_VFTRACE

#endif // ifdef SELF_PROFILE_VFTRACE

#endif // define SELF_PROFILE_H
