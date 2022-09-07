#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "event_types.h"

event_t event_from_line(char *line) {
   event_t event;
   // get the action
   char *token = strtok(line, " ");
   if (strcmp(token, "Enter:") == 0) {
      event.action = enter;
   } else if (strcmp(token, "Leave:") == 0) {
      event.action = leave;
   } else {
      fprintf(stderr, "Unknown action %s\n", token);
      abort();
   }
   // get function name
   token = strtok(NULL, " ");
   event.name = token;
   // skip "at"
   token = strtok(NULL, " ");
   // get seconds
   token = strtok(NULL, " ");
   event.t_sec = atoll(token);
   // skip "s"
   token = strtok(NULL, " ");
   // get nanoseconds
   token = strtok(NULL, " ");
   event.t_nsec = atoll(token);

   return event;
}

void print_event(FILE *fp, event_t event) {
   fprintf(fp, "%10ld.%09ld: %s %s\n",
           event.t_sec, event.t_nsec,
           event.action == enter ? "Enter" : "Leave",
           event.name);
}
