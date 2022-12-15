#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <time.h>

#include "signal_handling.h"
#include "event_types.h"

int get_event(event_t *event, FILE *stream) {
   int nread = 1;
   int character = getc(stream);
   if (character == EOF) {
      return -1;
   }
   if (character == 'E') {
      event->action = enter;
   } else if (character == 'L') {
      event->action = leave;
   } else {
      fprintf(stderr, "Unknown action %c\n", character);
      vftr_abort(0);
   }
   // get current file pointer position
   long int current_pos = ftell(stream);
   int namelen = 1; // one for null pointer
   while ((character = getc(stream)) != '\0') {
      namelen++;
   }
   if (fseek(stream, current_pos, SEEK_SET)) {
      fprintf(stderr, "Cound not adjust filepointer to %ld\n",
              current_pos);
      vftr_abort(0);
   }
   event->name = (char*) malloc(namelen*sizeof(char));
   for (int i=0; i<namelen; i++) {
      event->name[i] = getc(stream);
      nread++;
   }

   struct timespec timestamp;
   nread += fread(&timestamp, 1, sizeof(struct timespec), stream);
   event->t_sec = timestamp.tv_sec;
   event->t_nsec = timestamp.tv_nsec;

   return nread;
}

void print_event(FILE *fp, event_t event) {
   fprintf(fp, "%10ld.%09ld: %s %s\n",
           event.t_sec, event.t_nsec,
           event.action == enter ? "Enter" : "Leave",
           event.name);
}
