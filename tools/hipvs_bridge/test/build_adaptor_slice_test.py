"""Build an offline TU that compiles and exercises the patched cuVS adaptor code.

It slices the anonymous namespace that holds the ob_cuvs_* helpers out of
src/oblib/lib/vector/ob_vsag_adaptor.cpp, prepends light stubs for the
OceanBase/VSAG symbols those helpers use, and appends a small main() that runs
the snapshot serialize/deserialize path. Compiling this locally catches syntax
and logic errors in the patched region without building all of seekdb.
"""
import os
import re
import sys

ADAPTOR = sys.argv[1]
OUT = sys.argv[2]

with open(ADAPTOR, 'r', encoding='utf-8') as fh:
    lines = fh.readlines()

start = None
end = None
for i, line in enumerate(lines):
    if start is None and line.strip() == 'namespace {' and i > 60:
        start = i
    if line.strip() == '}  // anonymous namespace':
        end = i
if start is None or end is None:
    raise SystemExit('could not locate the anonymous namespace')

slice_text = ''.join(lines[start:end + 1])

# The snapshot helper resolves its scratch file through /proc/self/fd, which only
# exists on Linux. This offline harness runs on Windows as well, so point the
# scratch file at a plain path in the working directory. The shipped code is
# untouched; only the extracted copy is rewritten here.
slice_text = slice_text.replace('"/proc/self/fd/%d", ::fileno(file_)',
                                '"%s", "adaptor_slice_snapshot.bin"')
slice_text = slice_text.replace('FILE *file_ = ::tmpfile();',
                                'FILE *file_ = ::fopen("adaptor_slice_snapshot.bin", "w+b");')

PREAMBLE = r'''
// ---- offline stubs for the OceanBase / VSAG surface the slice needs -------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <istream>
#include <ostream>
#include <sstream>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <mutex>
#include <algorithm>

#ifdef _WIN32
#include <io.h>
#define fileno _fileno
#define fseeko fseek
#define ftello ftell
typedef void *pthread_t;
typedef struct { int unused; } pthread_attr_t;
static int pthread_attr_init(pthread_attr_t *a) { (void)a; return 0; }
static int pthread_attr_setstacksize(pthread_attr_t *a, size_t s) { (void)a; (void)s; return 0; }
static int pthread_attr_destroy(pthread_attr_t *a) { (void)a; return 0; }
static int pthread_create(pthread_t *t, pthread_attr_t *a, void *(*fn)(void *), void *arg) {
  (void)a; *t = (pthread_t)1; fn(arg); return 0;
}
static int pthread_join(pthread_t t, void **r) { (void)t; (void)r; return 0; }
#else
#include <pthread.h>
#endif

struct ObTimeUtility {
  static int64_t current_time() { return 0; }
};

#define K(x) (x)
#define KP(x) (x)
#define LOG_INFO(...) do {} while (0)
#define LOG_WARN(...) do {} while (0)
#define OB_SUCCESS 0
#define OB_ERR_UNEXPECTED 1
#define OB_ERR_VSAG_RETURN_ERROR 2

namespace vsag {
class Allocator {
 public:
  void *Allocate(size_t bytes) { return ::malloc(bytes); }
  void Deallocate(void *p) { ::free(p); }
};
}  // namespace vsag

class FilterInterface {
 public:
  virtual bool test(int64_t) = 0;
};

// ---- bridge symbols implemented by the stub in this TU --------------------
extern "C" {
void *seekdb_cuvs_build(const float *, long, long);
int   seekdb_cuvs_search(void *, const float *, long, long, unsigned int *, float *);
void  seekdb_cuvs_free(void *);
int   seekdb_cuvs_serialize(void *, const char *);
void *seekdb_cuvs_deserialize(const char *, long, long);
}

static int g_serialize_calls = 0;
static int g_build_calls = 0;
static int g_deserialize_calls = 0;
static void *g_stub_index = nullptr;

extern "C" void *seekdb_cuvs_build(const float *, long n, long dim) {
  (void)n; (void)dim;
  g_build_calls++;
  g_stub_index = (void *)0x1;
  return g_stub_index;
}
extern "C" int seekdb_cuvs_search(void *, const float *, long, long, unsigned int *, float *) {
  return 0;
}
extern "C" void seekdb_cuvs_free(void *) {}
extern "C" int seekdb_cuvs_serialize(void *, const char *filename) {
  g_serialize_calls++;
  FILE *f = ::fopen(filename, "wb");
  if (f == nullptr) { return 1; }
  const char *payload = "graph";
  ::fwrite(payload, 1, 5, f);
  ::fclose(f);
  return 0;
}
extern "C" void *seekdb_cuvs_deserialize(const char *, long, long) {
  g_deserialize_calls++;
  return (void *)0x2;
}
'''

POSTAMBLE = r'''
// ---- exercise the patched paths ------------------------------------------
int main() {
  ObCuvsEntry *ent = new ObCuvsEntry();
  ent->dim_ = 4;
  const int64_t n = OB_CUVS_MIN_PTS + 4;   // the snapshot path needs at least OB_CUVS_MIN_PTS rows
  ent->buf_ids_.assign(n, 11);
  ent->buf_vecs_.assign(n * 4, 0.5f);
  ent->built_n_ = n;
  ent->bridge_ = std::shared_ptr<void>((void *)0x1, [](void *) {});
  {
    std::lock_guard<std::mutex> guard(g_ob_cuvs_mu);
    g_ob_cuvs_reg[(void *)0x1] = ent;
  }

  std::ostringstream out(std::ios::binary);
  int ret = OB_SUCCESS;
  const bool handled = ob_cuvs_serialize((void *)0x1, out, ret);
  printf("serialize handled=%d ret=%d stream=%zu bridge_serialize_calls=%d\n",
         (int)handled, (int)ret, out.str().size(), g_serialize_calls);
  if (!handled || ret != OB_SUCCESS) { printf("serialize failed\n"); return 1; }
  if (g_serialize_calls != 1) { printf("bridge serialize not called\n"); return 1; }
  if (out.str().size() == 0) { printf("snapshot stream is empty\n"); return 1; }
  printf("serialize ok, stream bytes=%zu\n", out.str().size());

  // A second serialize must reuse the in-memory index (no rebuild).
  std::ostringstream out2(std::ios::binary);
  ob_cuvs_serialize((void *)0x1, out2, ret);
  if (g_build_calls != 0) { printf("index was rebuilt unexpectedly (builds=%d)\n", g_build_calls); return 1; }
  printf("reuse ok\n");

  // Now feed the snapshot back in and check the deserialize path.
  std::istringstream in(out.str(), std::ios::binary);
  { char magic[8] = {0}; in.read(magic, 8); }  // fdeserialize consumes the magic before calling us
  const int rc = ob_cuvs_deserialize((void *)0x2, 4, in);
  if (rc != OB_SUCCESS) { printf("deserialize failed rc=%d\n", rc); return 1; }
  if (g_deserialize_calls != 1) { printf("bridge deserialize not called\n"); return 1; }
  printf("deserialize ok\n");

  printf("PASS\n");
  return 0;
}
'''

with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write(PREAMBLE)
    fh.write('\n')
    fh.write(slice_text)
    fh.write('\n')
    fh.write(POSTAMBLE)

print('wrote', OUT, 'from lines', start + 1, '-', end + 1)
