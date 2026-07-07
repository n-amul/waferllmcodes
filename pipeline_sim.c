#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_STAGES 3
#define N_BLOCKS 1000
#define QCAP (N_BLOCKS + 1)

struct block {
  int _ID;
};

// A tiny FIFO (ring buffer) holding blocks waiting in front of a PE.
typedef struct {
  struct block *items[QCAP];
  int head, tail; // occupied = [head, tail); empty iff head==tail
} Queue;

static void q_init(Queue *q) { q->head = q->tail = 0; }
static int q_empty(Queue *q) { return q->head == q->tail; }
static void q_push(Queue *q, struct block *b) {
  q->items[q->tail] = b;
  q->tail = (q->tail + 1) % QCAP;
}
static struct block *q_pop(Queue *q) {
  struct block *b = q->items[q->head];
  q->head = (q->head + 1) % QCAP;
  return b;
}

// PE = one assembly-line station.
struct PE {
  struct block *_block;       // block being processed right now (NULL = idle)
  unsigned int _service_time; // CONFIG: cycles to process one block
  char _busy;
  int _finish_at; // cycle at which the current block finishes
  Queue que;      // blocks handed over from the previous stage
};

struct job {
  int stage, bid, start, end;
};

int *algo1(unsigned int *atomic, int size, int m) {
  int C = 0;
  for (int i = 0; i < size; i++)
    C += atomic[i];
  int goal = C / m;

  int *res = calloc(m, sizeof(int));
  int idx = 0;
  for (int i = 0; i < m; i++) {
    int sum = 0;
    while (res[i] < goal && idx < size) {
      res[i] += atomic[idx++];
    }
  }
  return res;
}
int run_pipeline(int num_stages, const unsigned int *service, int verbose) {
  struct PE *pipeline = malloc(sizeof(*pipeline) * num_stages);
  for (int i = 0; i < num_stages; i++) {
    pipeline[i]._block = NULL;
    pipeline[i]._service_time = service[i];
    pipeline[i]._busy = 0;
    pipeline[i]._finish_at = -1;
    q_init(&pipeline[i].que);
  }
  struct block *blocks = malloc(N_BLOCKS * sizeof *blocks);
  for (int i = 0; i < N_BLOCKS; i++) {
    blocks[i]._ID = i;
    q_push(&pipeline[0].que, &blocks[i]);
  }
  // clock:
  int now = 0, completed = 0, last_finish = 0;
  while (completed < N_BLOCKS) {
    for (int i = 0; i < num_stages; i++) // completions first
      if (pipeline[i]._busy && pipeline[i]._finish_at == now) {
        struct block *b = pipeline[i]._block;
        pipeline[i]._busy = 0;
        pipeline[i]._block = NULL;
        if (i == num_stages - 1) {
          last_finish = now;
          completed++;
        } else
          q_push(&pipeline[i + 1].que, b);
      }
    for (int i = 0; i < num_stages; i++) // then everyone grabs
      if (!pipeline[i]._busy && !q_empty(&pipeline[i].que)) {
        struct block *b = q_pop(&pipeline[i].que);
        pipeline[i]._block = b;
        pipeline[i]._busy = 1;
        pipeline[i]._finish_at = now + (int)pipeline[i]._service_time;
      }
    now++;
  }
  free(blocks);
  free(pipeline);

  if (verbose) {
    unsigned int slowest = 0;
    for (int i = 0; i < num_stages; i++)
      if (service[i] > slowest)
        slowest = service[i];
    for (int i = 0; i < num_stages; i++) {
      long busy = (long)N_BLOCKS * service[i];
      printf("    stage %d: %6u cyc   util %5.1f%%\n", i, service[i],
             100.0 * busy / last_finish);
    }
    printf("    -> finish=%d  throughput=%.6e blocks/cyc  (1/slowest=%.6e)\n",
           last_finish, (double)N_BLOCKS / last_finish, 1.0 / slowest);
  }
  return last_finish;
}
// Lenght, # of cols, total atomics, cost of relay hop, cost of sending
// stage(touch fabric mem so c1<c2)
double throughput(int L, int TC, double C, double C1, double C2) {
  double total_time = C / TC + L * C1 + L * L * C2;
  return 1.0 / total_time;
}

int main(void) {
  // config the line
  // naive -> atomic
  unsigned int atomic[6 + 17];
  unsigned int head[6] = {5078, 1033, 975, 1044, 1037, 1386};
  int atomic_size = 6 + 17;
  for (int i = 0; i < 6; i++)
    atomic[i] = head[i];
  for (int i = 6; i < 6 + 17; i++)
    atomic[i] = 1977;

  unsigned int naive[NUM_STAGES] = {6000, 1000, 37000};
  int fn = run_pipeline(3, naive, 1);

  // same 3 PEs, but repack from atomic pieces with balance
  int m = 3;
  int *bal = algo1(atomic, atomic_size, m);
  printf("\n=== (b) balanced: same %d PEs, greedy split ===\n", m);
  printf("    service = {%d, %d, %d}\n", bal[0], bal[1], bal[2]);
  int fb = run_pipeline(m, (unsigned int *)bal, 1);
  free(bal);

  // proving L=1 is best if we can fit in localmem and fill mesh sufficently
  double C = 0;
  for (int i = 0; i < atomic_size; i++)
    C += atomic[i];
  int TC = 512;
  double C1 = 30; // relay hop
  double C2_set[3] = {0.0, 50.0, 200.0};
  for (int i = 0; i < 3; i++) {
    double C2 = C2_set[i];
    printf("\n(TC=%d C=%.0f C1=%.0f C2=%.0f) ===\n", TC, C, C1, C2);
    printf("     L |  C/TC + L*C1 + L^2*C2 =  total | throughput\n");
    int best_L = 1;
    double best_t = -1;
    for (int L = 1; L <= 8; L++) {
      double t = throughput(L, TC, C, C1, C2);
      double base = C / TC, relay = L * C1, xfer = (double)L * L * C2;
      printf("    %2d | %6.1f +%5.1f + %7.1f = %7.1f | %.4e\n", L, base, relay,
             xfer, base + relay + xfer, t);
      if (t > best_t) {
        best_t = t;
        best_L = L;
      }
    }
    printf(" >>> best L = %d\n", best_L);
  }
  return 0;
}