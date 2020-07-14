#ifndef VFTR_OUTPUT_MACROS_H
#define VFTR_OUTPUT_MACROS_H

/* COMPUTE_COLUMN_WIDTH(COUNT,WIDTH)
   Compute WIDTH, the number of decimal places needed to represent COUNT
*/

#define COMPUTE_COLUMN_WIDTH(COUNT,WIDTH) \
                { long long count; int k;\
                  for( count=(COUNT),k=0; count; count/=10,k++ );\
                  if( k > WIDTH ) WIDTH = k; }

/* COMPUTE_COLWIDTH(COUNT,WIDTH,MINWIDTH,FMT,MOLD)
   A more advanced form, taking a minimum MINWIDTH into account and also 
   computing the column's format FMT using WIDTH and format MOLD.
*/

#define COMPUTE_COLWIDTH(COUNT,WIDTH,MINWIDTH,FMT,MOLD) \
                { long long count; int k;\
                  for( count=(COUNT),k=0; count; count/=10,k++ );\
                  WIDTH = k > MINWIDTH ? k : MINWIDTH;\
                  FMT = (char *) malloc( 10 );\
                  sprintf( FMT, MOLD, WIDTH );\
                }

/* OUTPUT_HEADER(HEADER,LENGTH,FILE)  Print HEADER in max LENGTH character positions to FILE */

#define OUTPUT_HEADER(HEADER,LENGTH,FILE)\
                { char *sh = HEADER;\
                  int ns = strlen(HEADER),\
                      nc = ns < LENGTH ? ns : LENGTH,\
                      nb = ns < LENGTH ? LENGTH-ns+1 : 1, is;\
                  for( is=0;is<nc;is++) fputc( *sh++, FILE );\
                  for( is=0;is<nb;is++) fputc( ' ',   FILE ); }

/* OUTPUT_DASHES_SP(LENGTH,FILE)   Print LENGTH dashes, followed by a space */

#define OUTPUT_DASHES_SP(LENGTH,FILE)\
                { int j; \
                  for( j=0; j<LENGTH; j++ ) \
                    fputc( '-', FILE ); \
                  fputc( ' ', FILE ); }

/* OUTPUT_DASHES_NL(LENGTH,FILE)   Print LENGTH dashes, followed by a newline */

#define OUTPUT_DASHES_NL(LENGTH,FILE)\
                { int j; \
                  for( j=0; j<LENGTH; j++ ) \
                    fputc( '-', FILE ); \
                  fputc( '\n', FILE ); }

/* OUTPUT_DASHES_SP_n(...,FILE)   Print n dashed lines, each followed by a space */

#define OUTPUT_DASHES_SP_2(LENGTH1,LENGTH2,FILE)\
                OUTPUT_DASHES_SP(LENGTH1,FILE)\
                OUTPUT_DASHES_SP(LENGTH2,FILE)

#define OUTPUT_DASHES_SP_3(LENGTH1,LENGTH2,LENGTH3,FILE)\
                OUTPUT_DASHES_SP(LENGTH1,FILE)\
                OUTPUT_DASHES_SP(LENGTH2,FILE)\
                OUTPUT_DASHES_SP(LENGTH3,FILE)

#define OUTPUT_DASHES_SP_4(LENGTH1,LENGTH2,LENGTH3,LENGTH4,FILE)\
                OUTPUT_DASHES_SP(LENGTH1,FILE)\
                OUTPUT_DASHES_SP(LENGTH2,FILE)\
                OUTPUT_DASHES_SP(LENGTH3,FILE)\
                OUTPUT_DASHES_SP(LENGTH4,FILE)

#define OUTPUT_DASHES_SP_5(LENGTH1,LENGTH2,LENGTH3,LENGTH4,LENGTH5,FILE)\
                OUTPUT_DASHES_SP(LENGTH1,FILE)\
                OUTPUT_DASHES_SP(LENGTH2,FILE)\
                OUTPUT_DASHES_SP(LENGTH3,FILE)\
                OUTPUT_DASHES_SP(LENGTH4,FILE)\
                OUTPUT_DASHES_SP(LENGTH5,FILE)

/* OUTPUT_DASHES_SP_3_NL(...,FILE)   Print 3 dashed lines, separated by spaces, followed by a newline */

#define OUTPUT_DASHES_SP_3_NL(LENGTH1,LENGTH2,LENGTH3,FILE)\
                OUTPUT_DASHES_SP(LENGTH1,FILE)\
                OUTPUT_DASHES_SP(LENGTH2,FILE)\
                OUTPUT_DASHES_NL(LENGTH3,FILE)


#endif
