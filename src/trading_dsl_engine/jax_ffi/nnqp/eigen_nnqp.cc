#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

#include "trading_dsl_engine/jax_ffi/nnqp/nnqp_eigen_impl.h"

namespace nnqp_eigen {

inline int64_t dim0(const ffi::BufferR2<ffi::DataType::F64>& x) { return x.dimensions()[0]; }
inline int64_t dim1(const ffi::BufferR2<ffi::DataType::F64>& x) { return x.dimensions()[1]; }
inline int64_t dim0(const ffi::BufferR1<ffi::DataType::F64>& x) { return x.dimensions()[0]; }

py::array_t<double> solve_direct_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> A,
    py::array_t<double, py::array::c_style | py::array::forcecast> c) {
  auto Ab = A.request();
  auto cb = c.request();
  if (Ab.ndim != 2 || cb.ndim != 1) throw std::runtime_error("rank mismatch");
  int p = static_cast<int>(cb.shape[0]);
  if (Ab.shape[0] != p || Ab.shape[1] != p) throw std::runtime_error("shape mismatch");

  py::array_t<double> out({p});
  auto ob = out.request();
  active_set_impl(static_cast<const double*>(Ab.ptr), static_cast<const double*>(cb.ptr), static_cast<double*>(ob.ptr), p, std::max(64, 4 * p));
  return out;
}

ffi::Error FwdImpl(
    ffi::BufferR2<ffi::DataType::F64> A,
    ffi::BufferR1<ffi::DataType::F64> c,
    ffi::ResultBufferR1<ffi::DataType::F64> out) {
  int64_t p0 = dim0(A);
  int64_t p1 = dim1(A);
  int64_t pc = dim0(c);
  int64_t po = out->dimensions()[0];
  if (p0 != p1 || p0 != pc || p0 != po) return ffi::Error::InvalidArgument("nnqp_eigen_fwd shape mismatch");
  int p = static_cast<int>(p0);
  active_set_impl(A.typed_data(), c.typed_data(), out->typed_data(), p, std::max(64, 4 * p));
  return ffi::Error::Success();
}

ffi::Error BwdImpl(
    ffi::BufferR2<ffi::DataType::F64> A,
    ffi::BufferR1<ffi::DataType::F64> c,
    ffi::BufferR1<ffi::DataType::F64> beta,
    ffi::BufferR1<ffi::DataType::F64> g,
    ffi::ResultBufferR2<ffi::DataType::F64> dA,
    ffi::ResultBufferR1<ffi::DataType::F64> dc) {
  int p = static_cast<int>(dim0(c));
  if (dim0(A) != p || dim1(A) != p || dim0(beta) != p || dim0(g) != p || dA->dimensions()[0] != p ||
      dA->dimensions()[1] != p || dc->dimensions()[0] != p) {
    return ffi::Error::InvalidArgument("nnqp_eigen_bwd shape mismatch");
  }

  MapMatC Amap(A.typed_data(), p, p);
  MapVecC bmap(beta.typed_data(), p);
  MapVecC gmap(g.typed_data(), p);
  double* dAp = dA->typed_data();
  double* dcp = dc->typed_data();
  std::fill(dAp, dAp + p * p, 0.0);
  std::fill(dcp, dcp + p, 0.0);

  std::vector<int> idx;
  idx.reserve(p);
  for (int i = 0; i < p; ++i) {
    if (bmap[i] > 1e-9) idx.push_back(i);
  }

  int k = static_cast<int>(idx.size());
  if (k == 0) return ffi::Error::Success();

  MatrixRM Af(k, k);
  Vector gf(k);
  for (int ii = 0; ii < k; ++ii) {
    int i = idx[ii];
    gf[ii] = gmap[i];
    for (int jj = 0; jj < k; ++jj) Af(ii, jj) = Amap(i, idx[jj]);
  }

  Vector v(k);
  bool ok = solve_ldlt(Af, gf, v);
  if (!ok) return ffi::Error::InvalidArgument("nnqp_eigen_bwd restricted solve failed");

  for (int ii = 0; ii < k; ++ii) dcp[idx[ii]] = v[ii];

  for (int ii = 0; ii < k; ++ii) {
    int i = idx[ii];
    for (int jj = 0; jj < k; ++jj) {
      int j = idx[jj];
      dAp[i * p + j] = -0.5 * (v[ii] * bmap[j] + bmap[i] * v[jj]);
    }
  }

  return ffi::Error::Success();
}

}  // namespace nnqp_eigen

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    nnqp_eigen_fwd,
    nnqp_eigen::FwdImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::DataType::F64>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Ret<ffi::BufferR1<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    nnqp_eigen_bwd,
    nnqp_eigen::BwdImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::DataType::F64>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Ret<ffi::BufferR2<ffi::DataType::F64>>()
        .Ret<ffi::BufferR1<ffi::DataType::F64>>());

PYBIND11_MODULE(_eigen_nnqp, m) {
  m.def("solve_direct", &nnqp_eigen::solve_direct_py);
  m.def("registrations", []() {
    py::dict d;
    d["nnqp_eigen_fwd"] = py::capsule(reinterpret_cast<void*>(nnqp_eigen_fwd), "xla._CUSTOM_CALL_TARGET");
    d["nnqp_eigen_bwd"] = py::capsule(reinterpret_cast<void*>(nnqp_eigen_bwd), "xla._CUSTOM_CALL_TARGET");
    return d;
  });
}
