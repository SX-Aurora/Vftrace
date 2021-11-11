# install directory for hw-scenario files
# without specific choice it is
# $(prefix)/share/vftrace/

if ENABLE_HWC
vftrace_dir = $(pkgdatadir)
vftrace__DATA =
if ON_VECTOR_ENGINE
vftrace__DATA += ve.json
endif
if HAS_PAPI
vftrace__DATA += tlb_skylake.json \
                 tlb_haswell.json \
                 tlb_broadwell.json \
                 stalls_skylake.json \
                 mflops_skylake.json \
                 mflops_sandybridge.json \
                 mflops_dp_skylake.json \
                 mflops_dp_sandybridge.json \
                 mflops_dp_broadwell.json \
                 mflops_broadwell.json \
                 icache_skylake.json \
                 dcache_skylake.json
endif
endif
