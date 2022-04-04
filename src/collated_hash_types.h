#ifndef COLLATED_HASH_TYPES_H
#define COLLATED_HASH_TYPES_H

#include <stdint.h>

typedef struct {
   int nhashes;
   uint64_t *hashes;
} hashlist_t;
#endif
