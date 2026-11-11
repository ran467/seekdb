/*
 * Offline test for tools/hipvs_bridge/seekdb_cuvs_bridge.c.
 *
 * It links the bridge against the cuVS stub in ./stubs and checks the bridge
 * plumbing end to end:
 *   - build / search / free round trip returns the correct nearest neighbour;
 *   - the OB_CUVS_* environment overrides really reach cuVS;
 *   - serialize / deserialize round trip preserves the index and its shape
 *     validation still rejects a mismatching (n, dim);
 *   - the one-shot seekdb_cuvs_cagra_knn helper works.
 *
 * Run it with tools/hipvs_bridge/test/run_bridge_test.sh (or build it by hand
 * as the script does). It needs no GPU and no hipVS installation.
 */
#include "seekdb_cuvs_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

/* Provided by the cuVS stub so the test can observe what reached cuVS. */
int ob_cuvs_stub_last_niter(void);
int ob_cuvs_stub_last_graph_degree(void);
int ob_cuvs_stub_last_intermediate_degree(void);
int ob_cuvs_stub_last_build_algo(void);
size_t ob_cuvs_stub_last_itopk(void);
size_t ob_cuvs_stub_last_search_width(void);
long ob_cuvs_stub_build_calls(void);
long ob_cuvs_stub_search_calls(void);

static int g_failures = 0;

#define CHECK(cond, msg, ...)                                        \
  do {                                                               \
    if (cond) {                                                      \
      printf("  ok   " msg "\n", ##__VA_ARGS__);                      \
    } else {                                                         \
      printf("  FAIL " msg "\n", ##__VA_ARGS__);                      \
      g_failures++;                                                  \
    }                                                                \
  } while (0)

#define COUNT 512
#define DIM 8

static void fill_vectors(float *vectors)
{
  for (int row = 0; row < COUNT; ++row) {
    for (int column = 0; column < DIM; ++column) {
      vectors[row * DIM + column] = (float)sin((double)(row * 31 + column * 17));
    }
  }
}

int main(void)
{
  float vectors[COUNT * DIM];
  unsigned int ids[16];
  float distances[16];
  fill_vectors(vectors);

  printf("[bridge-test] defaults (no env overrides)\n");
  {
    void *index = seekdb_cuvs_build(vectors, COUNT, DIM);
    CHECK(index != NULL, "build returns a handle");
    if (index == NULL) { return 1; }
    CHECK(ob_cuvs_stub_last_niter() == 100, "default nn_descent_niter == 100 (got %d)",
          ob_cuvs_stub_last_niter());
    CHECK(ob_cuvs_stub_last_graph_degree() == 128, "default graph_degree == 128 (got %d)",
          ob_cuvs_stub_last_graph_degree());
    CHECK(ob_cuvs_stub_last_intermediate_degree() == 256, "default intermediate_graph_degree == 256 (got %d)",
          ob_cuvs_stub_last_intermediate_degree());
    CHECK(ob_cuvs_stub_last_build_algo() == 0, "default build_algo == NN_DESCENT");
    seekdb_cuvs_free(index);
  }

  printf("[bridge-test] environment overrides reach cuVS\n");
  {
    setenv("OB_CUVS_NN_DESCENT_NITER", "7", 1);
    setenv("OB_CUVS_GRAPH_DEGREE", "64", 1);
    setenv("OB_CUVS_INTERMEDIATE_GRAPH_DEGREE", "96", 1);
    void *index = seekdb_cuvs_build(vectors, COUNT, DIM);
    CHECK(index != NULL, "build with overrides returns a handle");
    CHECK(ob_cuvs_stub_last_niter() == 7, "nn_descent_niter override == 7 (got %d)",
          ob_cuvs_stub_last_niter());
    CHECK(ob_cuvs_stub_last_graph_degree() == 64, "graph_degree override == 64 (got %d)",
          ob_cuvs_stub_last_graph_degree());
    CHECK(ob_cuvs_stub_last_intermediate_degree() == 96, "intermediate_graph_degree override == 96 (got %d)",
          ob_cuvs_stub_last_intermediate_degree());

    setenv("OB_CUVS_SEARCH_WIDTH", "16", 1);
    setenv("OB_CUVS_ITOPK_SIZE", "1024", 1);
    int rc = seekdb_cuvs_search(index, vectors + 10 * DIM, 1, 10, ids, distances);
    CHECK(rc == 0, "search succeeds");
    CHECK(ob_cuvs_stub_last_search_width() == 16, "search_width override == 16 (got %zu)",
          ob_cuvs_stub_last_search_width());
    CHECK(ob_cuvs_stub_last_itopk() == 1024, "itopk_size override == 1024 (got %zu)",
          ob_cuvs_stub_last_itopk());
    CHECK(ids[0] == 10u, "top-1 is the query itself (got %u)", ids[0]);
    CHECK(fabsf(distances[0]) < 0.0001f, "top-1 distance is ~0 (got %g)", distances[0]);
    seekdb_cuvs_free(index);
  }

  printf("[bridge-test] serialize / deserialize round trip\n");
  {
    const char *path = "bridge_test_snapshot.bin";
    void *index = seekdb_cuvs_build(vectors, COUNT, DIM);
    CHECK(index != NULL, "build for snapshot succeeds");
    CHECK(seekdb_cuvs_serialize(index, path) == 0, "serialize succeeds");

    void *bad_n = seekdb_cuvs_deserialize(path, COUNT - 1, DIM);
    CHECK(bad_n == NULL, "deserialize rejects a mismatching row count");
    void *bad_dim = seekdb_cuvs_deserialize(path, COUNT, DIM + 1);
    CHECK(bad_dim == NULL, "deserialize rejects a mismatching dimension");

    void *reloaded = seekdb_cuvs_deserialize(path, COUNT, DIM);
    CHECK(reloaded != NULL, "deserialize succeeds for the right shape");
    if (reloaded != NULL) {
      int rc = seekdb_cuvs_search(reloaded, vectors + 42 * DIM, 1, 10, ids, distances);
      CHECK(rc == 0, "search on the reloaded index succeeds");
      CHECK(ids[0] == 42u, "reloaded index returns the same neighbour (got %u)", ids[0]);
      seekdb_cuvs_free(reloaded);
    }
    remove(path);
    seekdb_cuvs_free(index);
  }

  printf("[bridge-test] batch search and one-shot helper\n");
  {
    unsigned int batch_ids[4 * 5];
    float batch_distances[4 * 5];
    void *index = seekdb_cuvs_build(vectors, COUNT, DIM);
    int rc = seekdb_cuvs_search(index, vectors, 4, 5, batch_ids, batch_distances);
    CHECK(rc == 0, "batch search succeeds");
    CHECK(batch_ids[0] == 0u && batch_ids[5] == 1u && batch_ids[10] == 2u && batch_ids[15] == 3u,
          "batch search returns each query's own row");
    seekdb_cuvs_free(index);

    unsigned int one_shot[3];
    rc = seekdb_cuvs_cagra_knn(vectors, COUNT, DIM, vectors + 5 * DIM, 1, 3, one_shot);
    CHECK(rc == 0, "one-shot knn succeeds");
    CHECK(one_shot[0] == 5u, "one-shot knn returns the query itself (got %u)", one_shot[0]);
  }

  printf("[bridge-test] invalid arguments\n");
  {
    CHECK(seekdb_cuvs_build(NULL, COUNT, DIM) == NULL, "build rejects a null base");
    CHECK(seekdb_cuvs_search(NULL, vectors, 1, 10, ids, distances) != 0, "search rejects a null handle");
    CHECK(seekdb_cuvs_serialize(NULL, "x.bin") != 0, "serialize rejects a null handle");
    CHECK(seekdb_cuvs_deserialize(NULL, COUNT, DIM) == NULL, "deserialize rejects a null path");
    seekdb_cuvs_free(NULL);
    CHECK(1, "free tolerates a null handle");
  }

  printf("[bridge-test] build calls: %ld, search calls: %ld\n",
         ob_cuvs_stub_build_calls(), ob_cuvs_stub_search_calls());
  if (g_failures == 0) {
    printf("[bridge-test] PASS\n");
    return 0;
  }
  printf("[bridge-test] FAIL (%d checks failed)\n", g_failures);
  return 1;
}
