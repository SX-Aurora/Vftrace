#ifndef EVENTS_H
#define EVENTS_H

#include <stdio.h>

#include "event_types.h"

int get_event(event_t *event, FILE *stream);

void print_event(FILE *fp, event_t event);

#endif
