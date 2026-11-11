# hipVS bridge offline test

`tools/hipvs_bridge/seekdb_cuvs_bridge.c` is the only place in the tree that
talks to hipVS/cuVS directly. Because the real library only exists on the AMD
evaluation image, the bridge is hard to exercise while developing on a laptop.
This directory closes that gap:

- `stubs/` contains a minimal, faithful subset of the cuVS C API, the DLPack
  types and the CUDA runtime calls the bridge uses, plus a small
  implementation (`cuvs_stub.c`) that answers with an exact brute-force kNN
  instead of the real approximate CAGRA search. It also records the parameters
  it was handed, which is what lets the test assert that the `OB_CUVS_*`
  overrides really reach cuVS.
- `test_bridge.c` drives the bridge through the whole C ABI and checks the
  behaviour that must not regress.
- `run_bridge_test.sh` builds and runs everything with a plain C compiler.
- `build_adaptor_slice_test.py` slices the `ob_cuvs_*` helpers out of
  `src/oblib/lib/vector/ob_vsag_adaptor.cpp` and compiles them against light
  stubs, so the snapshot serialize/deserialize path can be smoke-tested without
  building all of seekdb.

Run:

```bash
tools/hipvs_bridge/test/run_bridge_test.sh
```

The stub is a development aid only. It is never linked into the shipped
`libseekdb_cuvs_bridge.so`; the production build still uses the hipVS headers
and library from `$HIPVS_ROOT` (see `tools/hipvs_bridge/build.sh`).
