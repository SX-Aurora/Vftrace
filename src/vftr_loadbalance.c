/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#define _GNU_SOURCE

#ifdef _MPI
#include <mpi.h>
#endif

#include <float.h>
#include <limits.h>
#include <string.h>

#include "vftr_setup.h"
#include "vftr_timer.h"
#include "vftr_functions.h"
#include "vftr_stacks.h"
#include "vftr_output_macros.h"

// Comparator functions

int vftr_compare_mpi_times (const void *a1, const void *a2) {
    callsTimeRange_t *f1 = *(callsTimeRange_t **)a1;
    callsTimeRange_t *f2 = *(callsTimeRange_t **)a2;
    if (!f2) return -1; /* Order important to work around SX qsort problem */
    if (!f1) return  1;
    if (f2->maxMPItime > f1->maxMPItime) return  1;
    if (f2->maxMPItime < f1->maxMPItime) {
	return -1;
    } else {
           return  0;
    }
}

int vftr_compare_omp_times (const void *a1, const void *a2) {
    callsTimeRange_t *f1 = *(callsTimeRange_t **)a1;
    callsTimeRange_t *f2 = *(callsTimeRange_t **)a2;
    if( !f2 ) return -1; /* Order important to work around SX qsort problem */
    if( !f1 ) return  1;
    if( f2->maxOMPtime > f1->maxOMPtime ) return  1;
    if( f2->maxOMPtime < f1->maxOMPtime ) return -1;
    else           return  0;
}

/**********************************************************************/

/*
   Get global load balance info: returns an array of pointers
   loadbalInfo to the load balance info of each MPI rank.
*/

callsTime_t **vftr_get_loadbalance_info (function_t **funcTable)
{
    double         rtime;
    callsTime_t    *callsTime, **loadbalInfo;
    int            multiTask = ( vftr_mpisize  > 1 ),
                   nCallsTime, nCallsTimeBytes,
                   i, tid, fid;
#ifdef _MPI
    callsTime_t    *globCallsTime;
    int            totalSize, offset;
#endif

    loadbalInfo = (callsTime_t **) malloc( vftr_mpisize * sizeof(callsTime_t *) );

    /* Build array of all stack calls and times */

    nCallsTime = vftr_gStackscount * vftr_omp_threads;
    nCallsTimeBytes = nCallsTime * sizeof(callsTime_t);
    callsTime = (callsTime_t *) malloc( nCallsTimeBytes );
    memset( callsTime, 0, nCallsTimeBytes );

    for (i = 0; i < vftr_stackscount; i++) {
        if( funcTable[i] == NULL ) continue;
        for( tid=0; tid<vftr_omp_threads; tid++ ) {
            profdata_t *prof_current  = &funcTable[i]->prof_current [tid];
            profdata_t *prof_previous = &funcTable[i]->prof_previous[tid];
            int calls = prof_current->calls - prof_previous->calls;
            unsigned long long cycles = prof_current->cycles - prof_previous->cycles;
            unsigned long long timeExcl = prof_current->timeExcl - prof_previous->timeExcl;
	    rtime  = calls ? timeExcl * 1.0e-6 : 0;
            fid = funcTable[i]->gid;
            callsTime[fid*vftr_omp_threads+tid].calls = calls;
            callsTime[fid*vftr_omp_threads+tid].time  = rtime;
        }
    }
    if( multiTask ) {

#ifdef _MPI

        /* Gather all timecalls arrays */

        if( vftr_mpirank == 0 ) {

            totalSize = vftr_mpisize * nCallsTimeBytes;
            globCallsTime = (callsTime_t *) malloc( vftr_mpisize * nCallsTimeBytes );
            memset( globCallsTime, 0, totalSize );

            loadbalInfo = (callsTime_t **) malloc( vftr_mpisize * sizeof(callsTime_t *) );

            /* Fill load balance info pointer array to be returned */
            offset = 0;
            for( i=0; i<vftr_mpisize; i++ ) {
                loadbalInfo[i] = &globCallsTime[offset];
                offset += nCallsTime;
            }
        }

        PMPI_Gather( callsTime,     nCallsTimeBytes, MPI_BYTE,
                     globCallsTime, nCallsTimeBytes, MPI_BYTE,
                     0, MPI_COMM_WORLD);

        free( callsTime );
#endif
    } else {
        loadbalInfo = (callsTime_t **) malloc( sizeof(callsTime_t *) );
        loadbalInfo[0] = callsTime; 
    }
    return loadbalInfo;
}

void	
vftr_print_loadbalance( callsTime_t **gCallsTime,
                        int groupBase, int groupSize, FILE *pout,
                        int *loadIDs, int *nLoadIDs ) {
    callsTimeRange_t
                   *callsTimeRange, **cTRarray, cTRmax;
    int            multiTask   = ( vftr_mpisize  > 1 ),
                   multiThread = ( vftr_omp_threads  > 1 ),
                   i, j, k, rank, tid, fidp, nctr,
                   flen, clen, rankp, threadp, jpar,
                   minMPIcallsp,maxMPIcallsp,avgMPIcallsp,
                   minOMPcallsp,maxOMPcallsp,avgOMPcallsp,
                   minMPItimep, maxMPItimep, avgMPItimep,
                   minOMPtimep, maxOMPtimep, avgOMPtimep,
                   minMPIindxcp,maxMPIindxcp,minOMPindxcp,maxOMPindxcp,
                   minMPIindxtp,maxMPIindxtp,minOMPindxtp,maxOMPindxtp;
    char           fmtFuncname[10], fmtCaller[10];
    char           *fmtFid,         *fmtRank,        *fmtThread,      
                   *fmtMinMPIcalls, *fmtMaxMPIcalls, *fmtAvgMPIcalls, 
                   *fmtMinMPIindxc, *fmtMaxMPIindxc, 
                   *fmtMinMPItime,  *fmtMaxMPItime,  *fmtAvgMPItime, 
                   *fmtMinMPIindxt, *fmtMaxMPIindxt, 
                   *fmtMinOMPcalls, *fmtMaxOMPcalls, *fmtAvgOMPcalls, 
                   *fmtMinOMPindxc, *fmtMaxOMPindxc, 
                   *fmtMinOMPtime,  *fmtMaxOMPtime,  *fmtAvgOMPtime, 
                   *fmtMinOMPindxt, *fmtMaxOMPindxt;
    double         maxTime, truncTime;

    *nLoadIDs = 0;
    if( groupBase > vftr_mpisize ) return; /*** Problem workaround: should not happen */

    /* Get min/max calls, times and indices for both MPI and OMP */
    
    callsTimeRange  = (callsTimeRange_t *)  malloc( vftr_gStackscount *
                                                          sizeof(callsTimeRange_t  ));
    cTRarray        = (callsTimeRange_t **) malloc( vftr_gStackscount *
                                                          sizeof(callsTimeRange_t *));

    memset( callsTimeRange, 0, vftr_gStackscount * sizeof(callsTimeRange_t) );
    memset( &cTRmax,        0,                     sizeof(callsTimeRange_t) );

#define STORE(x,y) { callsTimeRange[nctr].x = y; if(cTRmax.x<y) cTRmax.x=y; }

    flen = 8;
    clen = 6;
    nctr = 0;

    for( i=0; i<vftr_gStackscount; i++ ) {
        long long mincalls,maxcalls,avgcalls,calls;
        float     mintime, maxtime, avgtime, time;
        int       minindxc,maxindxc,minindxt,maxindxt;

	cTRarray[nctr] = &callsTimeRange[i]; /* Array of pointers for sorting */
        callsTimeRange[nctr].stackIndex = i;

        k = strlen(vftr_gStackinfo[i].name);
        if(flen < k) flen = k;
        j = vftr_gStackinfo[i].ret;
	if (j >= 0) {
           k = strlen(vftr_gStackinfo[j].name); 
           if(clen < k) clen = k;
        }

        avgcalls = 0;
        avgtime  = 0.;
        
        if( multiTask ) {
            /* Compute calls and max thread times */
            mincalls = LONG_MAX;
            maxcalls = 0;
            minindxc = vftr_mpisize;
            maxindxc = 0;
            mintime  = FLT_MAX;
            maxtime  = 0.;
            minindxt = vftr_mpisize;
            maxindxt = 0;
            avgcalls = 0;
            avgtime  = 0.;
            for( rank=groupBase; rank<groupBase+groupSize; rank++ ) {
                callsTime_t *callsTime = gCallsTime[rank];
                j = i*vftr_omp_threads;
                calls = callsTime[j].calls;
                time  = callsTime[j].time;
                for( tid=1; tid<vftr_omp_threads; tid++ ) {
                    j = i*vftr_omp_threads+tid;
                    if( calls < callsTime[j].calls ) calls = callsTime[j].calls;
                    if( time  < callsTime[j].time  ) time  = callsTime[j].time;
                }
                if( mincalls > calls ) { mincalls = calls; minindxc = rank; }
                if( maxcalls < calls ) { maxcalls = calls; maxindxc = rank; }
                if( mintime  > time  ) { mintime  = time ; minindxt = rank; }
                if( maxtime  < time  ) { maxtime  = time ; maxindxt = rank; }
                avgcalls += calls;
                avgtime  += time;
            }
            avgcalls /= groupSize;
            avgtime  /= groupSize;
            STORE(minMPIcalls,mincalls)
            STORE(maxMPIcalls,maxcalls)
            STORE(avgMPIcalls,avgcalls)
            STORE(minMPItime ,mintime )
            STORE(maxMPItime ,maxtime )
            STORE(avgMPItime ,avgtime )
            STORE(minMPIindxc,minindxc)
            STORE(maxMPIindxc,maxindxc)
            STORE(minMPIindxt,minindxt)
            STORE(maxMPIindxt,maxindxt)
        }

        if( multiThread ) {
            /* Compute calls and max rank times */
            mincalls = LONG_MAX;
            maxcalls = 0;
            avgcalls = 0;
            minindxc = vftr_omp_threads;
            maxindxc = 0;
            mintime  = FLT_MAX;
            maxtime  = 0;
            avgtime  = 0.;
            minindxt = vftr_omp_threads;
            maxindxt = 0;
            for( tid=0; tid<vftr_omp_threads; tid++ ) {
                rank=groupBase;
                callsTime_t *callsTime = gCallsTime[rank];
                j = i*vftr_omp_threads+tid;
                calls = callsTime[j].calls;
                time  = callsTime[j].time;
                for( rank=groupBase+1; rank<groupBase+groupSize; rank++ ) {
                    callsTime_t *callsTime = gCallsTime[rank];
                    if( calls < callsTime[j].calls ) calls = callsTime[j].calls;
                    if( time  < callsTime[j].time  ) time  = callsTime[j].time;
                }
                if( mincalls > calls ) { mincalls = calls; minindxc = tid; }
                if( maxcalls < calls ) { maxcalls = calls; maxindxc = tid; }
                if( mintime  > time  ) { mintime  = time ; minindxt = tid; }
                if( maxtime  < time  ) { maxtime  = time ; maxindxt = tid; }
                avgcalls += calls;
                avgtime  += time;
            }
            avgcalls /= vftr_omp_threads;
            avgtime  /= vftr_omp_threads;
            STORE(minOMPcalls,mincalls)
            STORE(maxOMPcalls,maxcalls)
            STORE(avgOMPcalls,avgcalls)
            STORE(minOMPtime ,mintime )
            STORE(maxOMPtime ,maxtime )
            STORE(avgOMPtime ,avgtime )
            STORE(minOMPindxc,minindxc)
            STORE(maxOMPindxc,maxindxc)
            STORE(minOMPindxt,minindxt)
            STORE(maxOMPindxt,maxindxt)
        }
        nctr++;
    }
#undef STORE

    /* Sort the callsTimeRange array */
    qsort( (void *)cTRarray, (size_t)nctr, sizeof( callsTimeRange_t *),
	   multiTask ? vftr_compare_mpi_times : vftr_compare_omp_times );    

    fprintf( pout, "\n" );

    sprintf( fmtFuncname, " %%-%ds", flen );
    sprintf( fmtCaller,   " %%-%ds", clen );
    COMPUTE_COLWIDTH( vftr_stackscount        , fidp        , 2, fmtFid         , " %%%dd"  )
    COMPUTE_COLWIDTH( vftr_mpisize            , rankp       , 2, fmtRank        , " %%%dd"  )
    COMPUTE_COLWIDTH( vftr_omp_threads            , threadp     , 2, fmtThread      , " %%%dd"  )
    if( multiTask ) {
      COMPUTE_COLWIDTH( cTRmax.minMPIcalls      , minMPIcallsp, 2, fmtMinMPIcalls , " %%%dld" )
      COMPUTE_COLWIDTH( cTRmax.maxMPIcalls      , maxMPIcallsp, 2, fmtMaxMPIcalls , " %%%dld" )
      COMPUTE_COLWIDTH( cTRmax.avgMPIcalls      , avgMPIcallsp, 2, fmtAvgMPIcalls , " %%%dld" )
      COMPUTE_COLWIDTH( cTRmax.minMPIindxc      , minMPIindxcp, 2, fmtMinMPIindxc , " %%%dd"  )
      COMPUTE_COLWIDTH( cTRmax.maxMPIindxc      , maxMPIindxcp, 2, fmtMaxMPIindxc , " %%%dd"  )
      COMPUTE_COLWIDTH( cTRmax.minMPItime*10000., minMPItimep , 5, fmtMinMPItime  , " %%%d.3f")
      COMPUTE_COLWIDTH( cTRmax.maxMPItime*10000., maxMPItimep , 5, fmtMaxMPItime  , " %%%d.3f")
      COMPUTE_COLWIDTH( cTRmax.avgMPItime*10000., avgMPItimep , 5, fmtAvgMPItime  , " %%%d.3f")
      COMPUTE_COLWIDTH( cTRmax.minMPIindxt      , minMPIindxtp, 2, fmtMinMPIindxt , " %%%dd"  )
      COMPUTE_COLWIDTH( cTRmax.maxMPIindxt      , maxMPIindxtp, 2, fmtMaxMPIindxt , " %%%dd"  )
    }
    if( multiThread ) {
      COMPUTE_COLWIDTH( cTRmax.minOMPcalls      , minOMPcallsp, 2, fmtMinOMPcalls , " %%%dld" )
      COMPUTE_COLWIDTH( cTRmax.maxOMPcalls      , maxOMPcallsp, 2, fmtMaxOMPcalls , " %%%dld" )
      COMPUTE_COLWIDTH( cTRmax.avgOMPcalls      , avgOMPcallsp, 2, fmtAvgOMPcalls , " %%%dld" )
      COMPUTE_COLWIDTH( cTRmax.minOMPindxc      , minOMPindxcp, 2, fmtMinOMPindxc , " %%%dd"  )
      COMPUTE_COLWIDTH( cTRmax.maxOMPindxc      , maxOMPindxcp, 2, fmtMaxOMPindxc , " %%%dd"  )
      COMPUTE_COLWIDTH( cTRmax.minOMPtime*10000., minOMPtimep , 5, fmtMinOMPtime  , " %%%d.3f")
      COMPUTE_COLWIDTH( cTRmax.maxOMPtime*10000., maxOMPtimep , 5, fmtMaxOMPtime  , " %%%d.3f")
      COMPUTE_COLWIDTH( cTRmax.avgOMPtime*10000., avgOMPtimep , 5, fmtAvgOMPtime  , " %%%d.3f")
      COMPUTE_COLWIDTH( cTRmax.minOMPindxt      , minOMPindxtp, 2, fmtMinOMPindxt , " %%%dd"  )
      COMPUTE_COLWIDTH( cTRmax.maxOMPindxt      , maxOMPindxtp, 2, fmtMaxOMPindxt , " %%%dd"  )
    }

    fprintf( pout, "Parallel Performance Overview" );

    if( multiTask   )
        fprintf( pout, "; MPI ranks %d-%d",
                       groupBase, groupBase+groupSize-1 );
    if( multiThread )
        fprintf( pout, "; OpenMP threads 0-%d", vftr_omp_threads-1 );

    fprintf( pout, "\n\n ");

    if( multiTask ) {
        OUTPUT_HEADER( "Calls MPI",      maxMPIcallsp+1+maxMPIindxcp+1+
                                         minMPIcallsp+1+minMPIindxcp+1+
                                         avgMPIcallsp, pout )
        OUTPUT_HEADER( "Time[s] MPI",    maxMPItimep +1+maxMPIindxtp+1+
                                         minMPItimep +1+minMPIindxtp+1+
                                         avgMPItimep,  pout )
    }
    if( multiThread ) {
        OUTPUT_HEADER( "Calls OpenMP",   maxOMPcallsp+1+maxOMPindxcp+1+
                                         minOMPcallsp+1+minOMPindxcp+1+
                                         avgOMPcallsp, pout )
        OUTPUT_HEADER( "Time[s] OpenMP", maxOMPtimep +1+maxOMPindxtp+1+
                                         minOMPtimep +1+minOMPindxtp+1+
                                         avgOMPtimep,  pout )
    }

    jpar = 0;
    if( multiTask   ) jpar +=  5;
    if( multiThread ) jpar +=  5;
    if( jpar )
        OUTPUT_HEADER( "Load Balance", jpar-1, pout )
    fputs( "\n ", pout );

    if( multiTask ) {
        OUTPUT_DASHES_SP( maxMPIcallsp+maxMPIindxcp+\
                          minMPIcallsp+minMPIindxcp+\
                          avgMPIcallsp+4, pout )
        OUTPUT_DASHES_SP( maxMPItimep+maxMPIindxtp+\
                          minMPItimep+minMPIindxtp+\
                          avgMPItimep+4,  pout )
    }                     
    if( multiThread ) {
        OUTPUT_DASHES_SP( maxOMPcallsp+maxOMPindxcp+\
                          minOMPcallsp+minOMPindxcp+\
                          avgOMPcallsp+4, pout )
        OUTPUT_DASHES_SP( maxOMPtimep+maxOMPindxtp+\
                          minOMPtimep+minOMPindxtp+\
                          avgOMPtimep+4,  pout )
    }

    OUTPUT_DASHES_SP( jpar-1, pout )
    OUTPUT_DASHES_SP( flen+clen+fidp+2, pout )

    fputs( "\n ", pout );

    /* Rank info, thread 0 */
    if( multiTask ) {
        OUTPUT_HEADER( "Maximum", maxMPIcallsp, pout )
        OUTPUT_HEADER( "Rank",    maxMPIindxcp, pout )
        OUTPUT_HEADER( "Minimum", minMPIcallsp, pout )
        OUTPUT_HEADER( "Rank",    minMPIindxcp, pout )
        OUTPUT_HEADER( "Average", avgMPIcallsp, pout )
        OUTPUT_HEADER( "Maximum", maxMPItimep,  pout )
        OUTPUT_HEADER( "Rank",    maxMPIindxtp, pout )
        OUTPUT_HEADER( "Minimum", minMPItimep,  pout )
        OUTPUT_HEADER( "Rank",    minMPIindxtp, pout )
        OUTPUT_HEADER( "Average", avgMPItimep,  pout )
    }

    if( multiThread ) {
        /* Thread info */
        OUTPUT_HEADER( "Maximum", maxOMPcallsp, pout )
        OUTPUT_HEADER( "Thread",  maxOMPindxcp, pout )
        OUTPUT_HEADER( "Minimum", minOMPcallsp, pout )
        OUTPUT_HEADER( "Thread",  minOMPindxcp, pout )
        OUTPUT_HEADER( "Average", avgOMPcallsp, pout )
        OUTPUT_HEADER( "Maximum", maxOMPtimep,  pout )
        OUTPUT_HEADER( "Thread",  maxOMPindxtp, pout )
        OUTPUT_HEADER( "Minimum", minOMPtimep,  pout )
        OUTPUT_HEADER( "Thread",  minOMPindxtp, pout )
        OUTPUT_HEADER( "Average", avgOMPtimep,  pout )
    }

    if( multiTask ) {
        OUTPUT_HEADER( "%MPI", 4, pout )
    }
    if( multiThread ) {
        OUTPUT_HEADER( "%OMP", 4, pout )
    }

    OUTPUT_HEADER ( "Function", flen, pout )
    OUTPUT_HEADER ( "Caller",   clen, pout )
    fprintf( pout, "ID\n" );
    fputs( " ", pout );

    /* Horizontal lines (collection of dashes) */

    /* Rank info, thread 0 */
    if( multiTask ) {
        OUTPUT_DASHES_SP_5( maxMPIcallsp, maxMPIindxcp,\
                            minMPIcallsp, minMPIindxcp,\
                            avgMPIcallsp,                pout )
        OUTPUT_DASHES_SP_5( maxMPItimep,  maxMPIindxtp,\
                            minMPItimep,  minMPIindxtp,\
                            avgMPItimep,                 pout )
    }                       
    if( multiThread ) {
        OUTPUT_DASHES_SP_5( maxOMPcallsp, maxOMPindxcp,\
                            minOMPcallsp, minOMPindxcp,\
                            avgOMPcallsp,                pout )
        OUTPUT_DASHES_SP_5( maxOMPtimep,  maxOMPindxtp,\
                            minOMPtimep,  minOMPindxtp,\
                            avgOMPtimep,                 pout )
    }

    if( vftr_mpisize>1 ) { OUTPUT_DASHES_SP( 4, pout ) }
    if( vftr_omp_threads>1 ) { OUTPUT_DASHES_SP( 4, pout ) }

    OUTPUT_DASHES_SP_3_NL( flen, clen, fidp, pout )

    /* All headers printed. Next: compute max runtime */
    
    maxTime = 0.;
    for( k=0; k<nctr; k++ ) {
        callsTimeRange_t *ctr = cTRarray[k];
        if( multiTask ) {
            maxTime += ctr->avgMPItime;
        } else
            maxTime += ctr->avgOMPtime;
    }
    truncTime = 0.99 * maxTime;

    /* All headers printed. Next: the table itself */
    
    maxTime = 0.;
    for( k=0; k<nctr; k++ ) {
        float pareffmpi, pareffomp;
        callsTimeRange_t *ctr = cTRarray[k];
        i = ctr->stackIndex;
        loadIDs[k] = i;
        (*nLoadIDs)++;   

        if( multiTask ) {
            /* Skip if no calls or no time */
            if( ctr->minMPIcalls + ctr->maxMPIcalls == 0 ||
                ctr->maxMPItime == 0.                       ) continue;
            /* Print sum of calls and time for threads */
            pareffmpi = 100. * ctr->avgMPItime / ctr->maxMPItime;
            fprintf( pout, fmtMaxMPIcalls, ctr->maxMPIcalls );
            fprintf( pout, fmtMaxMPIindxc, ctr->maxMPIindxc );
            fprintf( pout, fmtMinMPIcalls, ctr->minMPIcalls );
            fprintf( pout, fmtMinMPIindxc, ctr->minMPIindxc );
            fprintf( pout, fmtAvgMPIcalls, ctr->avgMPIcalls );
            fprintf( pout, fmtMaxMPItime , ctr->maxMPItime  );
            fprintf( pout, fmtMaxMPIindxt, ctr->maxMPIindxt );
            fprintf( pout, fmtMinMPItime , ctr->minMPItime  );
            fprintf( pout, fmtMinMPIindxt, ctr->minMPIindxt );
            fprintf( pout, fmtAvgMPItime , ctr->avgMPItime  );
        }

        if( multiThread ) {
            /* Print calls and time for all threads of current rank */
            pareffomp = ctr->maxOMPtime > 0. ?
                        100. * ctr->avgOMPtime / ctr->maxOMPtime : 0.;
            fprintf( pout, fmtMaxOMPcalls, ctr->maxOMPcalls );
            fprintf( pout, fmtMaxOMPindxc, ctr->maxOMPindxc );
            fprintf( pout, fmtMinOMPcalls, ctr->minOMPcalls );
            fprintf( pout, fmtMinOMPindxc, ctr->minOMPindxc );
            fprintf( pout, fmtAvgOMPcalls, ctr->avgOMPcalls );
            fprintf( pout, fmtMaxOMPtime , ctr->maxOMPtime  );
            fprintf( pout, fmtMaxOMPindxt, ctr->maxOMPindxt );
            fprintf( pout, fmtMinOMPtime , ctr->minOMPtime  );
            fprintf( pout, fmtMinOMPindxt, ctr->minOMPindxt );
            fprintf( pout, fmtAvgOMPtime , ctr->avgOMPtime  );
        }

        if( multiTask   )
            fprintf( pout, pareffmpi < 99.95 ? " %4.1f" : " 100.", pareffmpi );
        if( multiThread )
            fprintf( pout, pareffomp < 99.95 ? " %4.1f" : " 100.", pareffomp );
        
	fprintf( pout, fmtFuncname, vftr_gStackinfo[i].name );
        j = vftr_gStackinfo[i].ret;
        if (j >= 0) fprintf( pout, fmtCaller,   vftr_gStackinfo[j].name );
        fprintf( pout, fmtFid,      i );
        fprintf( pout, "\n" );
        maxTime += multiTask ? ctr->avgMPItime : ctr->avgOMPtime;        
        if( maxTime >= truncTime ) break;
    }
    fprintf( pout, "\n" );

    free(callsTimeRange);
    free(cTRarray);
}
