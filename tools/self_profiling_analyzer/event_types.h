#ifndef EVENT_TYPES_H
#define EVENT_TYPES_H

typedef enum {
   enter,
   leave
} action_t;

typedef struct {
   action_t action;
   char *name;
   long t_sec;
   long t_nsec;
} event_t;

#endif
