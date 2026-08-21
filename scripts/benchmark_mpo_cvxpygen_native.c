#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpg_workspace.h"
#include "cpg_solve.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static long rss_kib(void) {
    FILE *file = fopen("/proc/self/status", "r");
    if (file == NULL) {
        return -1;
    }
    char line[256];
    long value = -1;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "VmRSS: %ld kB", &value) == 1) {
            break;
        }
    }
    fclose(file);
    return value;
}

int main(int argc, char **argv) {
    int runs = argc > 1 ? atoi(argv[1]) : 30;
    const char *mode = argc > 2 ? argv[2] : "same";
    const int free_solver = argc > 3 ? atoi(argv[3]) : 1;
    if (runs < 1) {
        runs = 1;
    }

    double *samples = calloc((size_t)runs, sizeof(double));
    if (samples == NULL) {
        return 2;
    }

    cpg_set_solver_verbose(0);
    cpg_set_solver_max_iter(200);
    cpg_set_solver_tol_gap_abs(1e-8);
    cpg_set_solver_tol_gap_rel(1e-8);
    cpg_set_solver_tol_feas(1e-8);

    for (int k = 0; k < 2; ++k) {
        cpg_solve();
        if (free_solver && solver != NULL) {
            clarabel_DefaultSolver_free(solver);
            solver = NULL;
        }
    }

    const long rss_before = rss_kib();
    for (int k = 0; k < runs; ++k) {
        if (strcmp(mode, "q") == 0) {
            const double value = -2.6546904942559377e-4 *
                                 (1.0 + 1e-6 * ((k & 1) ? 1.0 : -1.0));
            cpg_update_expected_returns(0, value);
        } else if (strcmp(mode, "A") == 0) {
            const double value = 7.3403158778105029e-4 *
                                 (1.0 + 1e-6 * ((k & 1) ? 1.0 : -1.0));
            cpg_update_risk_factor_0(0, value);
        }

        const double start = now_s();
        cpg_solve();
        if (free_solver && solver != NULL) {
            clarabel_DefaultSolver_free(solver);
            solver = NULL;
        }
        samples[k] = now_s() - start;
    }
    const long rss_after = rss_kib();

    qsort(samples, (size_t)runs, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int k = 0; k < runs; ++k) {
        sum += samples[k];
    }
    int p90_index = (int)ceil(0.9 * runs) - 1;
    if (p90_index < 0) {
        p90_index = 0;
    } else if (p90_index >= runs) {
        p90_index = runs - 1;
    }
    const double median = runs % 2
        ? samples[runs / 2]
        : 0.5 * (samples[runs / 2 - 1] + samples[runs / 2]);

    printf(
        "mode=%s free=%d runs=%d mean_ms=%.6f median_ms=%.6f "
        "min_ms=%.6f p90_ms=%.6f max_ms=%.6f obj=%.12g iter=%lu "
        "status=%lu rss_before_kib=%ld rss_after_kib=%ld rss_delta_kib=%ld\n",
        mode,
        free_solver,
        runs,
        1e3 * sum / runs,
        1e3 * median,
        1e3 * samples[0],
        1e3 * samples[p90_index],
        1e3 * samples[runs - 1],
        (double)CPG_Result.info->obj_val,
        (unsigned long)CPG_Result.info->iter,
        (unsigned long)CPG_Result.info->status,
        rss_before,
        rss_after,
        rss_after - rss_before
    );

    free(samples);
    return 0;
}
