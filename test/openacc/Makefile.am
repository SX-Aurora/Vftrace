if ENABLE_MPI
if USES_OPEN_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_openmpi.sh;
else
if USES_NEC_MPI
if ON_VECTOR_ENGINE
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necvhmpi.sh;
endif
else
if USES_INTEL_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_intelmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_genericmpi.sh;
endif
endif
endif
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_nompi.sh;
endif

AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/unset_vftr_env.sh;

AM_CFLAGS = -O3 $(FINSTRUMENT_FLAG) -acc -acc=gpu -g -Mnodwarf
AM_CFLAGS += -I$(top_srcdir)/src/
AM_FCFLAGS = -O3 $(FINSTRUMENT_FLAG) 
AM_FCFLAGS += -I$(top_builddir)/src/

AM_CFLAGS += -D_CUDA

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm


check_PROGRAMS = acc_region1 \
                 acc_region2 \
		 acc_region3 \
		 acc_region4

if ENABLE_MPI
check_PROGRAMS += collate_acc1
endif

acc_region1_SOURCES = $(srcdir)/acc_region1.c
acc_region1_CFLAGS = $(AM_CFLAGS)
acc_region1_LDFLAGS = $(AM_LDFLAGS)

acc_region2_SOURCES = $(srcdir)/acc_region2.c
acc_region2_CFLAGS = $(AM_CFLAGS)
acc_region2_LDFLAGS = $(AM_LDFLAGS)

acc_region3_SOURCES = $(srcdir)/acc_region3.c
acc_region3_CFLAGS = $(AM_CFLAGS)
acc_region3_LDFLAGS = $(AM_LDFLAGS)

acc_region4_SOURCES = $(srcdir)/acc_region4.c
acc_region4_CFLAGS = $(AM_CFLAGS)
acc_region4_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
collate_acc1_SOURCES = $(srcdir)/collate_acc1.c
collate_acc1_CFLAGS = $(AM_CFLAGS)
collate_acc1_LDFLAGS = $(AM_LDFLAGS)
endif


dist_noinst_SCRIPTS = acc_region1.sh \
		      acc_region2.sh \
 		      acc_region3.sh \
 		      acc_region4.sh \
	              acc_ranklogfile.sh

if ENABLE_MPI
dist_noinst_SCRIPTS += collate_acc1.sh
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)
TESTS = $(dist_noinst_SCRIPTS)
