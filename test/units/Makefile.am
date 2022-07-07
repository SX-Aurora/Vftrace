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

AM_CFLAGS = -I$(top_srcdir)/src/ -I$(top_srcdir)/src/mpi/utils/

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if ENABLE_MPI
  AM_CFLAGS += -D_MPI
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif
endif

if ENABLE_OMP
  AM_CFLAGS += -D_OMP
endif

check_PROGRAMS = 
check_PROGRAMS += sort_int8 \
		  sort_perm_int8 \
		  sort_uint8 \
		  sort_perm_uint8 \
		  sort_float \
		  sort_perm_float \
		  sort_int8 \
                  sort_perm_int8 \
                  sort_uint8 \
                  sort_perm_uint8 \
                  sort_float \
                  sort_perm_float \
		  sort_uint64 \
		  sort_longlong \
		  sort_double
check_PROGRAMS += chop_chars
check_PROGRAMS += symbols_1 \
		  symbols_2 \
		  symbols_precise \
		  pauseresume_precise
check_PROGRAMS += environment_1 \
		  environment_2
check_PROGRAMS += hashing \
		  stackhashing \
		  collatehashes
if ENABLE_MPI
check_PROGRAMS += collatehashes_parallel
endif
check_PROGRAMS += collatestacks
if ENABLE_MPI
check_PROGRAMS += collatestacks_parallel_1 \
		  collatestacks_parallel_2_1 \
		  collatestacks_parallel_2_2 \
		  collatestacks_parallel_3
endif
check_PROGRAMS += stacktree \
		  stacklist \
		  getstackstring

check_PROGRAMS += profilelist \
		  exclusive_times \
		  total_hook_overhead
if ENABLE_MPI
check_PROGRAMS += total_mpi_overhead
endif
if ENABLE_OMP
check_PROGRAMS += total_omp_overhead
endif
check_PROGRAMS += sorting_stacks
check_PROGRAMS += print_prof_table
if ENABLE_MPI
check_PROGRAMS += print_mpi_table
endif

check_PROGRAMS += threadtree

check_PROGRAMS += filenames
#
#                 scenario_1 \
#                 scenario_2 \
#                 scenario_3 \
#		 tables \
#                 browse_1 \
#		 stacks_1 \
#		 no_instrument_attribute

if ENABLE_HWC
if ON_VECTOR_ENGINE
check_PROGRAMS += sxhwc_1
endif
endif


sort_int8_SOURCES = $(srcdir)/sort_int8.c \
			 $(srcdir)/bad_rng.c
sort_int8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_int8_LDFLAGS = $(AM_LDFLAGS)

sort_perm_int8_SOURCES = $(srcdir)/sort_perm_int8.c \
			      $(srcdir)/bad_rng.c
sort_perm_int8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_perm_int8_LDFLAGS = $(AM_LDFLAGS)

sort_uint8_SOURCES = $(srcdir)/sort_uint8.c \
			  $(srcdir)/bad_rng.c
sort_uint8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_uint8_LDFLAGS = $(AM_LDFLAGS)

sort_perm_uint8_SOURCES = $(srcdir)/sort_perm_uint8.c \
			       $(srcdir)/bad_rng.c
sort_perm_uint8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_perm_uint8_LDFLAGS = $(AM_LDFLAGS)

sort_float_SOURCES = $(srcdir)/sort_float.c \
			  $(srcdir)/bad_rng.c
sort_float_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_float_LDFLAGS = $(AM_LDFLAGS)

sort_perm_float_SOURCES = $(srcdir)/sort_perm_float.c \
			       $(srcdir)/bad_rng.c
sort_perm_float_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_perm_float_LDFLAGS = $(AM_LDFLAGS)

sort_uint64_SOURCES = $(srcdir)/sort_uint64.c \
			   $(srcdir)/bad_rng.c
sort_uint64_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_uint64_LDFLAGS = $(AM_LDFLAGS)

sort_longlong_SOURCES = $(srcdir)/sort_longlong.c \
			     $(srcdir)/bad_rng.c
sort_longlong_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_longlong_LDFLAGS = $(AM_LDFLAGS)

sort_double_SOURCES = $(srcdir)/sort_double.c \
			   $(srcdir)/bad_rng.c
sort_double_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_double_LDFLAGS = $(AM_LDFLAGS)

chop_chars_SOURCES = $(srcdir)/chop_chars.c
chop_chars_CFLAGS = $(AM_CFLAGS)
chop_chars_LDFLAGS = $(AM_LDFLAGS)

symbols_1_SOURCES = $(srcdir)/symbols_1.c \
		    $(srcdir)/protofuncts.c
symbols_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_1_LDFLAGS = $(AM_LDFLAGS)

symbols_2_SOURCES = $(srcdir)/symbols_2.c
symbols_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_2_LDFLAGS = $(AM_LDFLAGS)

symbols_precise_SOURCES = $(srcdir)/symbols_precise.c \
	           $(srcdir)/protofuncts.c
symbols_precise_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_precise_LDFLAGS = $(AM_LDFLAGS)

pauseresume_precise_SOURCES = $(srcdir)/pauseresume_precise.c
pauseresume_precise_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
pauseresume_precise_LDFLAGS = $(AM_LDFLAGS)

environment_1_SOURCES = $(srcdir)/environment_1.c
environment_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
environment_1_LDFLAGS = $(AM_LDFLAGS)

environment_2_SOURCES = $(srcdir)/environment_2.c
environment_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
environment_2_LDFLAGS = $(AM_LDFLAGS)

hashing_SOURCES = $(srcdir)/hashing.c
hashing_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
hashing_LDFLAGS = $(AM_LDFLAGS)

stackhashing_SOURCES = $(srcdir)/stackhashing.c \
		       $(srcdir)/dummysymboltable.c 
stackhashing_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
stackhashing_LDFLAGS = $(AM_LDFLAGS)

collatehashes_SOURCES = $(srcdir)/collatehashes.c \
		       $(srcdir)/dummysymboltable.c 
collatehashes_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatehashes_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
collatehashes_parallel_SOURCES = $(srcdir)/collatehashes_parallel.c \
				 $(srcdir)/dummysymboltable.c 
collatehashes_parallel_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatehashes_parallel_LDFLAGS = $(AM_LDFLAGS)
endif

collatestacks_SOURCES = $(srcdir)/collatestacks.c \
			$(srcdir)/dummysymboltable.c
collatestacks_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
collatestacks_parallel_1_SOURCES = $(srcdir)/collatestacks_parallel_1.c \
			       $(srcdir)/dummysymboltable.c
collatestacks_parallel_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_1_LDFLAGS = $(AM_LDFLAGS)

collatestacks_parallel_2_1_SOURCES = $(srcdir)/collatestacks_parallel_2_1.c \
				 $(srcdir)/dummysymboltable.c
collatestacks_parallel_2_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_2_1_LDFLAGS = $(AM_LDFLAGS)

collatestacks_parallel_2_2_SOURCES = $(srcdir)/collatestacks_parallel_2_2.c \
				 $(srcdir)/dummysymboltable.c
collatestacks_parallel_2_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_2_2_LDFLAGS = $(AM_LDFLAGS)

collatestacks_parallel_3_SOURCES = $(srcdir)/collatestacks_parallel_3.c \
			       $(srcdir)/dummysymboltable.c
collatestacks_parallel_3_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_3_LDFLAGS = $(AM_LDFLAGS)
endif

filenames_SOURCES = $(srcdir)/filenames.c
filenames_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
filenames_LDFLAGS = $(AM_LDFLAGS)

stacktree_SOURCES = $(srcdir)/stacktree.c \
		    $(srcdir)/dummysymboltable.c
stacktree_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
stacktree_LDFLAGS = $(AM_LDFLAGS)

stacklist_SOURCES = $(srcdir)/stacklist.c \
		    $(srcdir)/dummysymboltable.c
stacklist_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
stacklist_LDFLAGS = $(AM_LDFLAGS)

getstackstring_SOURCES = $(srcdir)/getstackstring.c \
			 $(srcdir)/dummysymboltable.c
getstackstring_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
getstackstring_LDFLAGS = $(AM_LDFLAGS)

profilelist_SOURCES = $(srcdir)/profilelist.c
profilelist_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
profilelist_LDFLAGS = $(AM_LDFLAGS)

exclusive_times_SOURCES = $(srcdir)/exclusive_times.c \
			  $(srcdir)/dummysymboltable.c
exclusive_times_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
exclusive_times_LDFLAGS = $(AM_LDFLAGS)

total_hook_overhead_SOURCES = $(srcdir)/total_hook_overhead.c \
			      $(srcdir)/dummysymboltable.c
total_hook_overhead_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
total_hook_overhead_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
total_mpi_overhead_SOURCES = $(srcdir)/total_mpi_overhead.c \
			     $(srcdir)/dummysymboltable.c
total_mpi_overhead_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
total_mpi_overhead_LDFLAGS = $(AM_LDFLAGS)
endif

if ENABLE_OMP
total_omp_overhead_SOURCES = $(srcdir)/total_omp_overhead.c \
			     $(srcdir)/dummysymboltable.c
total_omp_overhead_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
total_omp_overhead_LDFLAGS = $(AM_LDFLAGS)
endif

sorting_stacks_SOURCES = $(srcdir)/sorting_stacks.c \
                         $(srcdir)/dummysymboltable.c
sorting_stacks_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/src/sorting/
sorting_stacks_LDFLAGS = $(AM_LDFLAGS)

print_prof_table_SOURCES = $(srcdir)/print_prof_table.c \
			   $(srcdir)/dummysymboltable.c
print_prof_table_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
print_prof_table_LDFLAGS = $(AM_LDFLAGS)

print_mpi_table_SOURCES = $(srcdir)/print_mpi_table.c \
			  $(srcdir)/dummysymboltable.c
print_mpi_table_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
print_mpi_table_LDFLAGS = $(AM_LDFLAGS)

threadtree_SOURCES = $(srcdir)/threadtree.c \
		     $(srcdir)/dummysymboltable.c
threadtree_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
threadtree_LDFLAGS = $(AM_LDFLAGS)

#scenario_1_SOURCES = $(srcdir)/scenario_1.c
#scenario_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
#scenario_1_LDFLAGS = $(AM_LDFLAGS)
#
#scenario_2_SOURCES = $(srcdir)/scenario_2.c
#scenario_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
#scenario_2_LDFLAGS = $(AM_LDFLAGS)
#
#scenario_3_SOURCES = $(srcdir)/scenario_3.c
#scenario_3_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
#scenario_3_LDFLAGS = $(AM_LDFLAGS)
#
#tables_SOURCES = $(srcdir)/tables.c
#tables_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#tables_LDFLAGS = $(AM_LDFLAGS)
#
#stacks_1_SOURCES = $(srcdir)/stacks_1.c
#stacks_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#stacks_1_LDFLAGS = $(AM_LDFLAGS)

#browse_1_SOURCES = $(srcdir)/browse_1.c
#browse_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#browse_1_LDFLAGS = $(AM_LDFLAGS)
#
#no_instrument_attribute_SOURCES = $(srcdir)/no_instrument_attribute.c
#no_instrument_attribute_CFLAGS = $(AM_CFLAGS) -finstrument-functions
#no_instrument_attribute_LDFLAGS = $(AM_LDFLAGS)
#
#if ENABLE_HWC
#if ON_VECTOR_ENGINE
#sxhwc_1_SOURCES = $(srcdir)/sxhwc_1.c
#sxhwc_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#sxhwc_1_LDFLAGS = $(AM_LDFLAGS)
#endif
#endif

dist_check_SCRIPTS =
dist_check_SCRIPTS += sort_int8.sh \
		      sort_perm_int8.sh \
		      sort_uint8.sh \
		      sort_perm_uint8.sh \
		      sort_float.sh \
		      sort_perm_float.sh \
		      sort_uint64.sh \
		      sort_longlong.sh \
		      sort_double.sh
dist_check_SCRIPTS += chop_chars.sh
dist_check_SCRIPTS += symbols_1.sh \
		      symbols_2.sh \
		      symbols_precise.sh \
		      pauseresume_precise.sh
dist_check_SCRIPTS += environment_1.sh \
		      environment_2.sh
dist_check_SCRIPTS += hashing.sh \
		      stackhashing.sh \
		      collatehashes.sh
if ENABLE_MPI
dist_check_SCRIPTS += collatehashes_parallel.sh
endif
dist_check_SCRIPTS += collatestacks.sh
if ENABLE_MPI
dist_check_SCRIPTS += collatestacks_parallel_1.sh \
		      collatestacks_parallel_2.sh \
		      collatestacks_parallel_3.sh
endif
dist_check_SCRIPTS += stacktree.sh \
		      stacklist.sh \
		      getstackstring.sh
dist_check_SCRIPTS += profilelist.sh \
		      exclusive_times.sh \
		      total_hook_overhead.sh
if ENABLE_MPI
dist_check_SCRIPTS += total_mpi_overhead.sh
endif
if ENABLE_OMP
dist_check_SCRIPTS += total_omp_overhead.sh
endif
dist_check_SCRIPTS += sorting_stacks_none.sh \
		      sorting_stacks_calls.sh \
		      sorting_stacks_overhead.sh \
		      sorting_stacks_stackid.sh \
		      sorting_stacks_texcl.sh \
		      sorting_stacks_tincl.sh
dist_check_SCRIPTS += print_prof_table.sh
if ENABLE_MPI
dist_check_SCRIPTS += print_mpi_table.sh
endif
dist_check_SCRIPTS += threadtree.sh
dist_check_SCRIPTS += filenames.sh
#		     scenario_1.sh \
#		     scenario_2.sh \
#		     scenario_3.sh \
#		     stacks_1.sh \
#		     browse_1.sh \
#		     no_instrument_attribute.sh

if ENABLE_HWC
if ON_VECTOR_ENGINE
dist_check_SCRIPTS += sxhwc_1.sh
endif
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
XFAIL_TESTS = scenario_3.sh

if ENABLE_HWC
if ON_VECTOR_ENGINE
XFAIL_TESTS += sxhwc_1.sh
endif
endif
