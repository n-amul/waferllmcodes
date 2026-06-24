// unlike gpu where sharedmemory/ kernel reads tensor when ever you need,
// On a wafer-scale mesh, every tensor element placement have to be decided by
// the compiler so basically this is sharding pass(only label)
// each core would maximum store 3 tiles. e.g. GEMM A(i=y,k),B(j=x,k), and
// C(cumulative)
// 2D seperation = cut every tensor with row->y-axis, col -> x-axis

// e.g. 32*32 fixed mesh
// goal input: Q = tensor(A) @ WQ. output: 1024 cores runing different program
// individually

/************* front end ********************/
// pass 1 assign 2d shard specs to every tensor - just put labels
// inputs are just big mats A=[L=2048, E=4096]
// WQ=[E=4096,H=4096] note for pass 3(GEMM) we place them as same position
// as mat and core, only diagnal would be correct. for label, where to cut,
// should be oppisite(outer y -> x) so that shift compute could work

// pass 2 calculate how many element of the tensor stored in each cores

// pass 3 Q = A @ WQ <- we don't have label for Q so we label in pass3
// row -> y0axis, col -> x-axis

// case 1 MeshGEMM(C = A @ B)
// C[i][j] = Σ_k A[i][k] · B[k][j]
// A contracts col(k) -> k is A's col -> x-axis
// B contracts row(k) -> k is B's row -> y-axis (oposite)

// compute-shift condition: each core do one thing, self A-tile * B-tile
// so core need to know where(i,j) the result contributes.
// [i,k][k,j]-> the core know i,j while in case like Q @ K^T ->[i,k][j,k] the
// core doesn't know where j is!
// now compute-shift can be perform and fill that square. (iterate all k=2)
// e.g. C[0][0] = A[0][0]·B[0][0] + A[0][1]·B[1][0] + A[0][2]·B[2][0]
// and these A[0][1]·B[1][0] + A[0][2]·B[2][0] elements are in
// core(1,0),(0,1),(2,0),(0,2). C+=A[0][0]·B[0][0] (shift)-> C +=
// A[0][1]·B[1][0] -> C += A[0][2]·B[2][0] complete!

// case 2 dist-GEMM-T need different shift! note. Q and K^T are GEMM output, so
// it is not from pass 1. Q@K^T doesn't meet compute shift condition [i,k]@[j,k]
// setup: core(x,y) = Q[i=y][k=x], k[j=y][k=x]
// instead of transpose k(worst L), step 0-work on diagnal first
// s[0][0], s[1][1], s[2][2]. step 1 - shift k along yaxis by one. s[0][1],
// s[1][2], s[2][0], repeat the shift until there is no remainder. s[0][2],
// s[1][0],s[2][1] k=[j=(y+1)%3]
// propagation and lowering

// pass 4 validate PLMR-M

// e.g. Tensor A
/*
  [1 2 3]   -> [L:2,E:3]
  [4 5 6]
  A.name = "A"
  A.ndims = 2
  A.dims[0] = { name:"L", size:2, shard:Y }
  A.dims[1] = { name:"E", size:3, shard:X }

  so core
  (x=0,y=0) → 1     (x=1,y=0) → 2     (x=2,y=0) → 3
  (x=0,y=1) → 4     (x=1,y=1) → 5     (x=2,y=1) → 6
*/

/************* back end ********************/
// meshgemm_run(compute--shift)
// dist_gemm_t_run(shift)
// meshgemv_run(ktree_allreduce)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// tile ptr
#define TP(base, y, x, N, sz) ((base) + (size_t)((y) * (N) + (x)) * (sz))

// FRONT-END  —  IR + the four passes
typedef enum { AXIS_X, AXIS_Y, AXIS_REP } Axis;
typedef struct {
  int Nx, Ny;
} Mesh;
#define MAXDIMS 4
typedef struct {
  char name[8];
  int size;
  Axis shard;
} Dim;
typedef struct {
  char name[16];
  Dim dims[MAXDIMS];
  int ndims;
} Tensor;
typedef struct {
  const char *kind;
  int reduce_axis;
  const char *note;
} Collective;

static const char *axis_str(Axis a) {
  return a == AXIS_X ? "x" : a == AXIS_Y ? "y" : "rep";
}

// tensor helpers
// find which idx for name L, E, etc..
static int dim_index(const Tensor *t, const char *d) {
  for (int i = 0; i < t->ndims; i++)
    if (!strcmp(t->dims[i].name, d))
      return i;
  return -1;
}
// find that dim's label(which axis to cut?)
static Axis get_shard(const Tensor *t, const char *d) {
  return t->dims[dim_index(t, d)].shard;
}
// find that dim's size
static int get_size(const Tensor *t, const char *d) {
  return t->dims[dim_index(t, d)].size;
}
static int local_dim(Axis a, int sz, Mesh m) {
  return a == AXIS_X ? sz / m.Nx : a == AXIS_Y ? sz / m.Ny : sz;
}

static void tensor_spec(const Tensor *t, char *buf) {
  int n = sprintf(buf, "%s[", t->name);
  for (int i = 0; i < t->ndims; i++)
    n += sprintf(buf + n, "%s%s", t->dims[i].name, axis_str(t->dims[i].shard));
  sprintf(buf + n, "]");
}
// make 2-dim tensor
static Tensor mk2(const char *nm, const char *n1, int s1, const char *n2,
                  int s2) {
  Tensor t;
  strcpy(t.name, nm);
  t.ndims = 2;
  strcpy(t.dims[0].name, n1);
  t.dims[0].size = s1;
  t.dims[0].shard = AXIS_REP;
  strcpy(t.dims[1].name, n2);
  t.dims[1].size = s2;
  t.dims[1].shard = AXIS_REP;
  return t;
}

// pass 1
// label where to cut, activation: outer/token dim ->Y, contraction dim->X
static Tensor assign_activation_shard(Tensor t, const char *outer,
                                      const char *contract) {
  for (int i = 0; i < t.ndims; i++)
    t.dims[i].shard = AXIS_REP;
  t.dims[dim_index(&t, outer)].shard = AXIS_Y;
  t.dims[dim_index(&t, contract)].shard = AXIS_X;
  return t;
}
// outer ->X,contraction ->Y
static Tensor assign_weight_shard(Tensor W, const char *contract,
                                  const char *newdim) {
  for (int i = 0; i < W.ndims; i++)
    W.dims[i].shard = AXIS_REP;
  W.dims[dim_index(&W, contract)].shard = AXIS_Y;
  W.dims[dim_index(&W, newdim)].shard = AXIS_X;
  return W;
}

// pass 3
// normal GEMM lowering: choose the kernel + derive output sharding
static Tensor lower_gemm(Tensor act, Tensor W, const char *out_name,
                         const char *outer, const char *contract,
                         const char *newdim, Collective *coll) {
  /* out[outer,newdim] = act[outer,contract] @ W[contract,newdim].
   * out sharding derived (outer->Y, newdim->X) so sequential GEMMs chain
   * with ZERO re-layout.  dist-GEMM-T if contraction on the SAME axis in
   * both operands (would otherwise force a transpose). */
  if (get_shard(&act, contract) == get_shard(&W, contract)) {
    coll->kind = "dist-GEMM-T";
    coll->reduce_axis = AXIS_X;
    coll->note = "contract on same axis -> avoid transpose";
  } else if (get_size(&act, outer) == 1) {
    coll->kind = "MeshGEMV";
    coll->reduce_axis = AXIS_Y;
    coll->note = "token dim=1 (decode) -> GEMV + K-tree allreduce";
  } else {
    coll->kind = "MeshGEMM";
    coll->reduce_axis = -1;
    coll->note = "A shifts on X, W shifts on Y (compute-shift)";
  }
  // propagation
  Tensor out;
  strcpy(out.name, out_name);
  out.ndims = 2;
  strcpy(out.dims[0].name, outer);
  out.dims[0].size = get_size(&act, outer);
  out.dims[0].shard = AXIS_Y;
  strcpy(out.dims[1].name, newdim);
  out.dims[1].size = get_size(&W, newdim);
  out.dims[1].shard = AXIS_X;
  return out;
}
static Tensor lower_gemm_T(Tensor Q, Tensor K, const char *out_name,
                           const char *outer_q, Collective *coll) {
  /* scores[Lq,Lk] = Q[Lq,h] @ K[Lk,h]^T.  Q,K both have h on X -> transpose
   * trap; emit dist-GEMM-T (shift K on Y, ReduceAdd on X, no real transpose).
   */
  coll->kind = "dist-GEMM-T";
  coll->reduce_axis = AXIS_X;
  coll->note = "Q@K^T: shift K on Y + ReduceAdd on X";
  Tensor out;
  strcpy(out.name, out_name);
  out.ndims = 2;
  strcpy(out.dims[0].name, outer_q);
  out.dims[0].size = get_size(&Q, outer_q);
  out.dims[0].shard = AXIS_Y;
  strcpy(out.dims[1].name, "Lk");
  out.dims[1].size = get_size(&K, outer_q);
  out.dims[1].shard = AXIS_X;
  return out;
}

// pass 4 validate PLMR-M (per-core tile must fit SRAM)
static int validate_M(const Tensor *t, Mesh m, int sram, int dbytes,
                      long *need) {
  long e = 1;
  for (int i = 0; i < t->ndims; i++)
    e *= local_dim(t->dims[i].shard, t->dims[i].size, m);
  *need = e * dbytes;
  return *need <= sram;
}
// BACKEND
// K-tree allreduce
//  e.g.  act(indicies) = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], n=16, crit=0
//  g = 4 ->  [0 1 2 3] -> root0(3 adds),  [4 5 6 7] -> root4...
// -> [0 4 8 12] (3 adds) ->vals0+=vals4,vals8,vals12  crit = 6
static int ktree_allreduce(double *vals, int cnt, int vlen, int K) {
  int g = (int)lround(pow((double)cnt, 1.0 / K));
  if (g < 2)
    g = 2;
  int *act = malloc(cnt * sizeof(int)), n = cnt;
  for (int i = 0; i < cnt; i++)
    act[i] = i;
  int crit = 0;
  for (int ph = 0; ph < K && n > 1; ph++) {
    int *nx = malloc(cnt * sizeof(int)), nn = 0, gs = 0;
    for (int s = 0; s < n; s += g) {
      int root = act[s], mem = (s + g <= n) ? g : (n - s);
      for (int m = 1; m < mem; m++) {
        int src = act[s + m];
        for (int j = 0; j < vlen; j++)
          vals[root * vlen + j] += vals[src * vlen + j];
      }
      if (mem - 1 > gs)
        gs = mem - 1;
      nx[nn++] = root;
    }
    crit += gs;
    memcpy(act, nx, nn * sizeof(int));
    n = nn;
    free(nx);
  }
  while (n > 1) {
    int root = act[0], src = act[n - 1];
    for (int j = 0; j < vlen; j++)
      vals[root * vlen + j] += vals[src * vlen + j];
    n--;
    crit++;
  }
  free(act);
  return crit;
}

// KERNEL 1 MeshGEMM   Cfull = Afull * Bfull
// cut small tiles from A and B
// compute shift N times
// add small results into Cfull
static void meshgemm_run(int N, int TM, int TK, int TN, const double *Afull,
                         const double *Bfull, double *Cfull) {
  int Kd = N * TK, P = N * TN;
  size_t aT = (size_t)TM * TK, bT = (size_t)TK * TN, cT = (size_t)TM * TN;
  double *At = malloc((size_t)N * N * aT * sizeof(double));
  double *Bt = malloc((size_t)N * N * bT * sizeof(double));
  double *Ct = calloc((size_t)N * N * cT, sizeof(double));
  for (int y = 0; y < N; y++)
    for (int x = 0; x < N; x++) { /* Cannon-aligned load */
      int abx = (x + y) % N;
      double *a = TP(At, y, x, N, aT);
      for (int i = 0; i < TM; i++)
        for (int k = 0; k < TK; k++)
          a[i * TK + k] = Afull[(size_t)(y * TM + i) * Kd + (abx * TK + k)];
      int bby = (y + x) % N;
      double *b = TP(Bt, y, x, N, bT);
      for (int k = 0; k < TK; k++)
        for (int j = 0; j < TN; j++)
          b[k * TN + j] = Bfull[(size_t)(bby * TK + k) * P + (x * TN + j)];
    }
  double *ta = malloc(aT * sizeof(double)), *tb = malloc(bT * sizeof(double));
  for (int step = 0; step < N; step++) {
    for (int y = 0; y < N; y++)
      for (int x = 0; x < N; x++) { /* local MAC */
        double *a = TP(At, y, x, N, aT), *b = TP(Bt, y, x, N, bT),
               *c = TP(Ct, y, x, N, cT);
        for (int i = 0; i < TM; i++)
          for (int j = 0; j < TN; j++) {
            double s = c[i * TN + j];
            for (int k = 0; k < TK; k++)
              s += a[i * TK + k] * b[k * TN + j];
            c[i * TN + j] = s;
          }
      }
    for (int y = 0; y < N; y++) { /* shift A left on X */
      memcpy(ta, TP(At, y, 0, N, aT), aT * sizeof(double));
      for (int x = 0; x < N - 1; x++)
        memcpy(TP(At, y, x, N, aT), TP(At, y, x + 1, N, aT),
               aT * sizeof(double));
      memcpy(TP(At, y, N - 1, N, aT), ta, aT * sizeof(double));
    }
    for (int x = 0; x < N; x++) { /* shift B up on Y */
      memcpy(tb, TP(Bt, 0, x, N, bT), bT * sizeof(double));
      for (int y = 0; y < N - 1; y++)
        memcpy(TP(Bt, y, x, N, bT), TP(Bt, y + 1, x, N, bT),
               bT * sizeof(double));
      memcpy(TP(Bt, N - 1, x, N, bT), tb, bT * sizeof(double));
    }
  }
  for (int y = 0; y < N; y++)
    for (int x = 0; x < N; x++) {
      double *c = TP(Ct, y, x, N, cT);
      for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
          Cfull[(size_t)(y * TM + i) * P + (x * TN + j)] = c[i * TN + j];
    }
  free(At);
  free(Bt);
  free(Ct);
  free(ta);
  free(tb);
}

// K ERNEL 2  —  dist-GEMM-T   C = A * B^T
//  dist_gemm_t_run(shift) : no transpose of the big matrix, only Y-shift of B
static void dist_gemm_t_run(int N, int TM, int TK, int TJ, const double *Afull,
                            const double *Bfull, double *Cfull, int Kp) {
  int Kd = N * TK, J = N * TJ;
  size_t aT = (size_t)TM * TK, bT = (size_t)TJ * TK, cT = (size_t)TM * TJ;
  double *At = malloc((size_t)N * N * aT * sizeof(double));
  double *Bt = malloc((size_t)N * N * bT * sizeof(double));
  for (int y = 0; y < N; y++)
    for (int x = 0; x < N; x++) { /* no Cannon align */
      double *a = TP(At, y, x, N, aT);
      for (int i = 0; i < TM; i++)
        for (int k = 0; k < TK; k++)
          a[i * TK + k] = Afull[(size_t)(y * TM + i) * Kd + (x * TK + k)];
      double *b = TP(Bt, y, x, N, bT);
      for (int j = 0; j < TJ; j++)
        for (int k = 0; k < TK; k++)
          b[j * TK + k] = Bfull[(size_t)(y * TJ + j) * Kd + (x * TK + k)];
    }
  double *tb = malloc(bT * sizeof(double));
  double *buf =
      malloc((size_t)N * cT * sizeof(double)); /* N partials for ktree */
  for (int step = 0; step < N; step++) {
    for (int y = 0; y < N; y++) {
      int jb = (y + step) % N;
      for (int x = 0; x < N; x++) { /* local A * B^T per x */
        double *a = TP(At, y, x, N, aT), *b = TP(Bt, y, x, N, bT),
               *p = buf + (size_t)x * cT;
        for (int i = 0; i < TM; i++)
          for (int j = 0; j < TJ; j++) {
            double s = 0;
            for (int k = 0; k < TK; k++)
              s += a[i * TK + k] * b[j * TK + k];
            p[i * TJ + j] = s;
          }
      }
      ktree_allreduce(buf, N, (int)cT, Kp); /* <-- ReduceAdd along X */
      for (int i = 0; i < TM; i++)
        for (int j = 0; j < TJ; j++)
          Cfull[(size_t)(y * TM + i) * J + (jb * TJ + j)] = buf[i * TJ + j];
    }
    for (int x = 0; x < N; x++) { /* shift B up on Y */
      memcpy(tb, TP(Bt, 0, x, N, bT), bT * sizeof(double));
      for (int y = 0; y < N - 1; y++)
        memcpy(TP(Bt, y, x, N, bT), TP(Bt, y + 1, x, N, bT),
               bT * sizeof(double));
      memcpy(TP(Bt, N - 1, x, N, bT), tb, bT * sizeof(double));
    }
  }
  free(At);
  free(Bt);
  free(tb);
  free(buf);
}

// KERNEL 3  —  MeshGEMV   y = x * W   (decode; ktree_allreduce.c)
// Y-axis allreduce is done by ktree_allreduce  <-- 4th file again.
// meshgemv_run(ktree_allreduce) : each core local GEMV, then sum down the Y
// axis
static void meshgemv_run(int N, int mc, int pc, const double *xfull,
                         const double *Wfull, double *yfull, int Kp) {
  int Pc = N * pc;
  double *buf = malloc((size_t)N * pc * sizeof(double));
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < N; y++) { /* per-core local GEMV */
      double *p = buf + (size_t)y * pc;
      for (int jj = 0; jj < pc; jj++) {
        double s = 0;
        for (int kk = 0; kk < mc; kk++) {
          int kr = y * mc + kk, jc = x * pc + jj;
          s += xfull[kr] * Wfull[(size_t)kr * Pc + jc];
        }
        p[jj] = s;
      }
    }
    ktree_allreduce(buf, N, pc, Kp); /* <-- allreduce along Y */
    for (int jj = 0; jj < pc; jj++)
      yfull[x * pc + jj] = buf[jj];
  }
  free(buf);
}

// DISPATCH  —  the bridge: a plan's `kind`  ->  the real kernel        ##
//  DISPATCH  the bridge of front and kernel: kind  ->  the real kernel
typedef struct {
  const char *kind;
  int N, d_outer, d_contract, d_new; /* PER-CORE tile dims from PASS 2 */
  const double *lhs, *rhs;
  double *out;
  int Kphases;
} Plan;

static void dispatch(const Plan *p) {
  if (!strcmp(p->kind, "MeshGEMM"))
    meshgemm_run(p->N, p->d_outer, p->d_contract, p->d_new, p->lhs, p->rhs,
                 p->out);
  else if (!strcmp(p->kind, "dist-GEMM-T"))
    dist_gemm_t_run(p->N, p->d_outer, p->d_contract, p->d_new, p->lhs, p->rhs,
                    p->out, p->Kphases);
  else if (!strcmp(p->kind, "MeshGEMV"))
    meshgemv_run(p->N, p->d_contract, p->d_new, p->lhs, p->rhs, p->out,
                 p->Kphases);
}

// R EFERENCE impls + verify  (plain textbook math = the "answer key")   ##
static void ref_gemm(const double *A, const double *B, double *C, int M, int Kd,
                     int P) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++) {
      double s = 0;
      for (int k = 0; k < Kd; k++)
        s += A[i * Kd + k] * B[k * P + j];
      C[i * P + j] = s;
    }
}
static void ref_gemm_T(const double *A, const double *B, double *C, int M,
                       int Kd, int J) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < J; j++) {
      double s = 0;
      for (int k = 0; k < Kd; k++)
        s += A[i * Kd + k] * B[j * Kd + k];
      C[i * J + j] = s;
    }
}
static void ref_gemv(const double *x, const double *W, double *y, int Mc,
                     int Pc) {
  for (int j = 0; j < Pc; j++) {
    double s = 0;
    for (int k = 0; k < Mc; k++)
      s += x[k] * W[k * Pc + j];
    y[j] = s;
  }
}
static double maxdiff(const double *X, const double *Y, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs(X[i] - Y[i]);
    if (d > m)
      m = d;
  }
  return m;
}

static void report(const char *op, const Collective *c, Mesh mesh, int do_,
                   int dc_, int dn_, const double *got, const double *ref,
                   int n) {
  char nb[8] = {0};
  double err = maxdiff(got, ref, n);
  printf("  %-16s kind=%-12s tiles[outer=%d,contract=%d,new=%d]  ->  dispatch  "
         "->  %s (err %.1e)\n",
         op, c->kind, do_, dc_, dn_, err < 1e-9 ? "OK" : "MISMATCH", err);
  (void)nb;
  (void)mesh;
}

// DRIVER
int main(void) {
  const int Kp = 2, SRAM = 48 * 1024;
  const int N = 4, L = 8, E = 12, Hh = 8; /* attention dims (all %4==0) */
  const int Mc = 12, Pc = 8;              /* decode GEMV dims */
  Mesh mesh = {N, N};

  /* full matrices */
  static double A[8 * 12], WQ[12 * 8], WK[12 * 8], Q[8 * 8], K[8 * 8], S[8 * 8];
  static double xv[12], Wo[12 * 8], ov[8];
  static double rQ[8 * 8], rK[8 * 8], rS[8 * 8], rO[8];
  srand(5);
  for (int i = 0; i < L * E; i++)
    A[i] = (rand() % 2000 - 1000) / 100.0;
  for (int i = 0; i < E * Hh; i++) {
    WQ[i] = (rand() % 2000 - 1000) / 100.0;
    WK[i] = (rand() % 2000 - 1000) / 100.0;
  }
  for (int i = 0; i < Mc; i++)
    xv[i] = (rand() % 2000 - 1000) / 100.0;
  for (int i = 0; i < Mc * Pc; i++)
    Wo[i] = (rand() % 2000 - 1000) / 100.0;

  char sp[64];
  printf("=== FRONT-END: PASS 1 shardings (chosen layout) ===\n");
  Tensor At = assign_activation_shard(mk2("A", "L", L, "E", E), "L", "E");
  Tensor WQt = assign_weight_shard(mk2("WQ", "E", E, "H", Hh), "E", "H");
  Tensor WKt = assign_weight_shard(mk2("WK", "E", E, "H", Hh), "E", "H");
  tensor_spec(&At, sp);
  printf("  activation %s\n", sp);
  tensor_spec(&WQt, sp);
  printf("  weight     %s   (E on opposite axis from activation)\n", sp);

  printf("\n=== PASS 3 lowering + DISPATCH + execute + verify ===\n");

  /* ---- op1: Q = A @ WQ  -> MeshGEMM ---- */
  Collective c1;
  Tensor Qt = lower_gemm(At, WQt, "Q", "L", "E", "H", &c1);
  Plan p1 = {c1.kind, N, L / N, E / N, Hh / N, A, WQ, Q, Kp};
  dispatch(&p1);
  ref_gemm(A, WQ, rQ, L, E, Hh);
  report("Q = A @ WQ", &c1, mesh, L / N, E / N, Hh / N, Q, rQ, L * Hh);

  /* ---- op2: K = A @ WK  -> MeshGEMM ---- */
  Collective c2;
  Tensor Kt = lower_gemm(At, WKt, "K", "L", "E", "H", &c2);
  Plan p2 = {c2.kind, N, L / N, E / N, Hh / N, A, WK, K, Kp};
  dispatch(&p2);
  ref_gemm(A, WK, rK, L, E, Hh);
  report("K = A @ WK", &c2, mesh, L / N, E / N, Hh / N, K, rK, L * Hh);

  /* ---- op3: S = Q @ K^T -> dist-GEMM-T (ktree reduces along X) ---- */
  Collective c3;
  Tensor St = lower_gemm_T(Qt, Kt, "S", "L", &c3);
  Plan p3 = {c3.kind, N, L / N, Hh / N, L / N, Q, K, S, Kp};
  dispatch(&p3); /* TM=Lq/N,TK=H/N,TJ=Lk/N */
  ref_gemm_T(Q, K, rS, L, Hh, L);
  report("S = Q @ K^T", &c3, mesh, L / N, Hh / N, L / N, S, rS, L * L);
  (void)St;

  /* ---- op4: o = x @ Wo  -> MeshGEMV (decode; ktree reduces along Y) ---- */
  Tensor xt = assign_activation_shard(mk2("x", "L", 1, "F", Mc), "L", "F");
  Tensor Wot = assign_weight_shard(mk2("Wo", "F", Mc, "E", Pc), "F", "E");
  Collective c4;
  Tensor ot =
      lower_gemm(xt, Wot, "o", "L", "F", "E", &c4); /* L=1 -> MeshGEMV */
  Plan p4 = {c4.kind, N, 1, Mc / N, Pc / N, xv, Wo, ov, Kp};
  dispatch(&p4);
  ref_gemv(xv, Wo, rO, Mc, Pc);
  report("o = x @ Wo", &c4, mesh, 1, Mc / N, Pc / N, ov, rO, Pc);
  (void)ot;

  /* ---- PASS 4 spot-check (per-core tile vs SRAM) ---- */
  printf("\n=== PASS 4: PLMR-M check ===\n");
  long need;
  const Tensor *chk[] = {&At, &WQt, &Qt};
  const char *nm[] = {"A", "WQ", "Q"};
  for (int i = 0; i < 3; i++) {
    int ok = validate_M(chk[i], mesh, SRAM, 2, &need);
    tensor_spec((Tensor *)chk[i], sp);
    printf("  %-10s %-12s %6ld B  %s\n", nm[i], sp, need,
           ok ? "OK" : "VIOLATION");
  }

  printf("\nAll ops: front-end decided kind -> dispatch routed to kernel -> "
         "verified.\n");
  printf("ktree_allreduce was invoked inside dist-GEMM-T (X-reduce) and "
         "MeshGEMV (Y-reduce).\n");
  return 0;
}