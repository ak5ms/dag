#ifndef CPP_STREAM_CLARABEL_H
#define CPP_STREAM_CLARABEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CppStreamClarabelSettings {
    uint32_t max_iter;
    bool verbose;
    double tol_gap_abs;
    double tol_gap_rel;
    double tol_feas;
    uint32_t max_threads;
} CppStreamClarabelSettings;

typedef struct CppStreamClarabelInfo {
    int32_t status;
    uint32_t iterations;
    double objective;
    double solve_time;
    double primal_residual;
    double dual_residual;
} CppStreamClarabelInfo;

enum CppStreamClarabelConeKind {
    CPP_STREAM_CLARABEL_ZERO = 0,
    CPP_STREAM_CLARABEL_NONNEGATIVE = 1,
    CPP_STREAM_CLARABEL_SECOND_ORDER = 2,
};

CppStreamClarabelSettings cpp_stream_clarabel_default_settings(void);

void* cpp_stream_clarabel_new(
    size_t n,
    size_t m,
    const size_t* p_colptr,
    const size_t* p_rowval,
    const double* p_values,
    size_t p_nnz,
    const double* q,
    const size_t* a_colptr,
    const size_t* a_rowval,
    const double* a_values,
    size_t a_nnz,
    const double* b,
    const uint32_t* cone_kinds,
    const size_t* cone_dims,
    size_t n_cones,
    CppStreamClarabelSettings settings);

int32_t cpp_stream_clarabel_update(
    void* solver,
    const double* q,
    const double* a_values,
    const double* b);
int32_t cpp_stream_clarabel_solve(void* solver);
CppStreamClarabelInfo cpp_stream_clarabel_info(void* solver);
const double* cpp_stream_clarabel_x(void* solver);
const double* cpp_stream_clarabel_z(void* solver);
const double* cpp_stream_clarabel_s(void* solver);
void cpp_stream_clarabel_free(void* solver);
const char* cpp_stream_clarabel_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
