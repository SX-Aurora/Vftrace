#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

uint8_t random_byte() {
   return rand()&0xFF;
}


int8_t random_int8() {
   int niter = sizeof(int8_t)/sizeof(uint8_t);
   int8_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

uint8_t random_uint8() {
   int niter = sizeof(uint8_t)/sizeof(uint8_t);
   uint8_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

int16_t random_int16() {
   int niter = sizeof(int16_t)/sizeof(uint8_t);
   int16_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

uint16_t random_uint16() {
   int niter = sizeof(uint16_t)/sizeof(uint8_t);
   uint16_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

int32_t random_int32() {
   int niter = sizeof(int32_t)/sizeof(uint8_t);
   int32_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

uint32_t random_uint32() {
   int niter = sizeof(uint32_t)/sizeof(uint8_t);
   uint32_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

int64_t random_int64() {
   int niter = sizeof(int64_t)/sizeof(uint8_t);
   int64_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

uint64_t random_uint64() {
   int niter = sizeof(uint64_t)/sizeof(uint8_t);
   uint64_t r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

char random_char() {
   int niter = sizeof(char)/sizeof(uint8_t);
   char r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed char random_signedchar() {
   int niter = sizeof(signed char)/sizeof(uint8_t);
   signed char r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned char random_unsignedchar() {
   int niter = sizeof(unsigned char)/sizeof(uint8_t);
   unsigned char r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

short random_short() {
   int niter = sizeof(short)/sizeof(uint8_t);
   short r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

short int random_shortint() {
   int niter = sizeof(short int)/sizeof(uint8_t);
   short int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed short random_signedshort() {
   int niter = sizeof(signed short)/sizeof(uint8_t);
   signed short r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed short int random_signedshortint() {
   int niter = sizeof(signed short int)/sizeof(uint8_t);
   signed short int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned short random_unsignedshort() {
   int niter = sizeof(unsigned short)/sizeof(uint8_t);
   unsigned short r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned short int random_unsignedshortint() {
   int niter = sizeof(unsigned short int)/sizeof(uint8_t);
   unsigned short int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

int random_int() {
   int niter = sizeof(int)/sizeof(uint8_t);
   int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed random_signed() {
   int niter = sizeof(signed)/sizeof(uint8_t);
   signed r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed int random_signedint() {
   int niter = sizeof(signed int)/sizeof(uint8_t);
   signed int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned random_unsigned() {
   int niter = sizeof(unsigned)/sizeof(uint8_t);
   unsigned r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned int random_unsignedint() {
   int niter = sizeof(unsigned int)/sizeof(uint8_t);
   unsigned int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

long random_long() {
   int niter = sizeof(long)/sizeof(uint8_t);
   long r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

long int random_longint() {
   int niter = sizeof(long int)/sizeof(uint8_t);
   long int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed long random_signedlong() {
   int niter = sizeof(signed long)/sizeof(uint8_t);
   signed long r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed long int random_signedlongint() {
   int niter = sizeof(signed long int)/sizeof(uint8_t);
   signed long int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned long random_unsignedlong() {
   int niter = sizeof(unsigned long)/sizeof(uint8_t);
   unsigned long r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned long int random_unsignedlongint() {
   int niter = sizeof(unsigned long int)/sizeof(uint8_t);
   unsigned long int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

long long random_longlong() {
   int niter = sizeof(long long)/sizeof(uint8_t);
   long long r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

long long int random_longlongint() {
   int niter = sizeof(long long int)/sizeof(uint8_t);
   long long int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed long long random_signedlonglong() {
   int niter = sizeof(signed long long)/sizeof(uint8_t);
   signed long long r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

signed long long int random_signedlonglongint() {
   int niter = sizeof(signed long long int)/sizeof(uint8_t);
   signed long long int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned long long random_unsignedlonglong() {
   int niter = sizeof(unsigned long long)/sizeof(uint8_t);
   unsigned long long r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

unsigned long long int random_unsignedlonglongint() {
   int niter = sizeof(unsigned long long int)/sizeof(uint8_t);
   unsigned long long int r = 0;
   for (int iter=0; iter<niter; iter++) {
      r = r<<8;
      r += random_byte();
   }
   return r;
}

float random_float() {
   float r;
   do {
      uint32_t ur = random_uint32();
      float *rp = (float*)&ur;
      r = *rp;
   } while (!isfinite(r));
   return r;
}

double random_double() {
   double r;
   do {
      uint64_t ur = random_uint64();
      double *rp = (double*)&ur;
      r = *rp;
   } while (!isfinite(r));
   return r;
}
