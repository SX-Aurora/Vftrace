AM_CFLAGS = -O3 -finstrument-functions 
AM_CFLAGS += -I$(top_srcdir)/src/
AM_FCFLAGS = -O3 -finstrument-functions 
AM_FCFLAGS += -I$(top_builddir)/src/

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if MPI_PROFILING
  AM_CFLAGS += -D_MPI
  AM_FCFLAGS += -D_MPI
endif
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif


################
# System Tests #
################
check_PROGRAMS = cregions1 \
		 cregions2 \
		 cregions3
		
cregions1_SOURCES = $(srcdir)/cregions1.c
cregions1_CFLAGS  = $(AM_CFLAGS)
cregions1_LDFLAGS = $(AM_LDFLAGS)

cregions2_SOURCES = $(srcdir)/cregions2.c
cregions2_CFLAGS  = $(AM_CFLAGS)
cregions2_LDFLAGS = $(AM_LDFLAGS)

cregions3_SOURCES = $(srcdir)/cregions3.c
cregions3_CFLAGS  = $(AM_CFLAGS)
cregions3_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = cregions1.sh \
		      cregions2.sh \
		      cregions3.sh

check_PROGRAMS += fregions1 \
		  fregions2 \
		  fregions3 \
		  fregions4

fregions1_SOURCES = $(srcdir)/fregions1.F90
fregions1_CFLAGS  = $(AM_CFLAGS)
fregions1_LDFLAGS = $(AM_LDFLAGS)

fregions2_SOURCES = $(srcdir)/fregions2.F90
fregions2_CFLAGS  = $(AM_CFLAGS)
fregions2_LDFLAGS = $(AM_LDFLAGS)

fregions3_SOURCES = $(srcdir)/fregions3.F90
fregions3_CFLAGS  = $(AM_CFLAGS)
fregions3_LDFLAGS = $(AM_LDFLAGS)

fregions4_SOURCES = $(srcdir)/fregions4.F90
fregions4_CFLAGS  = $(AM_CFLAGS)
fregions4_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += fregions1.sh \
		       fregions2.sh \
		       fregions3.sh \
		       fregions4.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
