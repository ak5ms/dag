#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace nnqp_eigen {

using MatrixRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;
using MapMatC = Eigen::Map<const MatrixRM>;
using MapVecC = Eigen::Map<const Vector>;
using MapVec = Eigen::Map<Vector>;

static constexpr double kTol = 1e-10;

inline bool solve_ldlt(const Eigen::Ref<const MatrixRM>& A, const Eigen::Ref<const Vector>& b, Vector& x) {
  Eigen::LDLT<MatrixRM> ldlt(A);
  if (ldlt.info() != Eigen::Success) return false;
  x = ldlt.solve(b);
  return ldlt.info() == Eigen::Success && x.allFinite();
}

inline void solve_restricted(const MapMatC& A, const MapVecC& c, const std::vector<uint8_t>& free, Vector& beta) {
  const int p = static_cast<int>(c.size());
  beta.setZero(p);

  int k = 0;
  for (int i = 0; i < p; ++i) {
    if (free[i]) ++k;
  }
  if (k == 0) return;

  std::vector<int> idx;
  idx.reserve(k);
  for (int i = 0; i < p; ++i) {
    if (free[i]) idx.push_back(i);
  }

  MatrixRM Af(k, k);
  Vector cf(k);
  for (int ii = 0; ii < k; ++ii) {
    const int i = idx[ii];
    cf[ii] = c[i];
    for (int jj = 0; jj < k; ++jj) Af(ii, jj) = A(i, idx[jj]);
  }

  Vector sol(k);
  bool ok = solve_ldlt(Af, cf, sol);
  if (!ok) {
    Af.diagonal().array() += 1e-10;
    ok = solve_ldlt(Af, cf, sol);
  }
  if (!ok) return;
  for (int ii = 0; ii < k; ++ii) beta[idx[ii]] = sol[ii];
}

inline void active_set_impl(const double* A_ptr, const double* c_ptr, double* out_ptr, int p, int max_iter) {
  MapMatC A(A_ptr, p, p);
  MapVecC c(c_ptr, p);
  MapVec out(out_ptr, p);

  std::vector<uint8_t> free(p, 0);
  Vector beta_unc(p);
  bool ok = solve_ldlt(A, c, beta_unc);
  if (!ok) {
    MatrixRM Ar = A;
    Ar.diagonal().array() += 1e-10;
    ok = solve_ldlt(Ar, c, beta_unc);
  }
  for (int i = 0; i < p; ++i) free[i] = ok && beta_unc[i] > 0.0;

  Vector beta(p);
  Vector beta_trial(p);
  Vector grad(p);
  beta.setZero();

  for (int it = 0; it < max_iter; ++it) {
    solve_restricted(A, c, free, beta_trial);

    bool has_neg = false;
    int drop = -1;
    double most_neg = 0.0;
    for (int i = 0; i < p; ++i) {
      if (free[i] && beta_trial[i] < -kTol) {
        if (!has_neg || beta_trial[i] < most_neg) {
          has_neg = true;
          most_neg = beta_trial[i];
          drop = i;
        }
      }
    }
    if (has_neg) {
      free[drop] = 0;
      continue;
    }

    beta = beta_trial.cwiseMax(0.0);
    grad.noalias() = A * beta;
    grad.noalias() -= c;

    bool kkt_ok = true;
    int add = -1;
    double min_grad = 0.0;
    bool first = true;
    for (int i = 0; i < p; ++i) {
      if (!free[i]) {
        if (first || grad[i] < min_grad) {
          first = false;
          min_grad = grad[i];
          add = i;
        }
        if (grad[i] < -kTol) kkt_ok = false;
      }
    }
    if (kkt_ok) {
      out = beta;
      return;
    }
    if (add >= 0) free[add] = 1;
  }

  out = beta.cwiseMax(0.0);
}

}  // namespace nnqp_eigen
