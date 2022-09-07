#ifndef EVENTS_H
#define EVENTS_H

#include "event_types.h"

event_t event_from_line(char *line);

void print_event(FILE *fp, event_t event);

#endif
