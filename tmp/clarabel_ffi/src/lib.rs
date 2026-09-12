//! Minimal stable C ABI around Clarabel 0.11.1 for cpp_stream.
//!
//! The ABI owns one persistent `DefaultSolver<f64>` and exposes fixed-sparsity
//! q/A/b updates. The Python formula compiler never enters this library at run
//! time; generated C++ nodes call it directly.

use clarabel::{algebra::CscMatrix, solver::*};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_error(message: impl ToString) {
    let message = message.to_string().replace('\0', " ");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(message)
            .unwrap_or_else(|_| CString::new("clarabel error").unwrap());
    });
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CppStreamClarabelSettings {
    pub max_iter: u32,
    pub verbose: bool,
    pub tol_gap_abs: f64,
    pub tol_gap_rel: f64,
    pub tol_feas: f64,
    pub max_threads: u32,
}

impl Default for CppStreamClarabelSettings {
    fn default() -> Self {
        Self {
            max_iter: 200,
            verbose: false,
            tol_gap_abs: 1e-8,
            tol_gap_rel: 1e-8,
            tol_feas: 1e-8,
            max_threads: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CppStreamClarabelInfo {
    pub status: i32,
    pub iterations: u32,
    pub objective: f64,
    pub solve_time: f64,
    pub primal_residual: f64,
    pub dual_residual: f64,
}

struct SolverHandle {
    solver: DefaultSolver<f64>,
    n: usize,
    m: usize,
    a_nnz: usize,
}

/// Sized borrowed wrappers avoid allocating temporary Vecs for Clarabel's
/// generic data-update API, whose slice implementations are otherwise unsized.
struct BorrowedMatrixUpdate<'a>(&'a [f64]);
struct BorrowedVectorUpdate<'a>(&'a [f64]);

impl MatrixProblemDataUpdate<f64> for BorrowedMatrixUpdate<'_> {
    fn update_matrix(
        &self,
        matrix: &mut CscMatrix<f64>,
        left_scale: &[f64],
        right_scale: &[f64],
        constant_scale: Option<f64>,
    ) -> Result<(), SparseFormatError> {
        self.0
            .update_matrix(matrix, left_scale, right_scale, constant_scale)
    }
}

impl VectorProblemDataUpdate<f64> for BorrowedVectorUpdate<'_> {
    fn update_vector(
        &self,
        vector: &mut [f64],
        scale: &[f64],
        constant_scale: Option<f64>,
    ) -> Result<(), SparseFormatError> {
        self.0.update_vector(vector, scale, constant_scale)
    }
}

fn cone(kind: u32, dim: usize) -> Result<SupportedConeT<f64>, String> {
    match kind {
        0 => Ok(ZeroConeT(dim)),
        1 => Ok(NonnegativeConeT(dim)),
        2 => Ok(SecondOrderConeT(dim)),
        _ => Err(format!("unsupported Clarabel cone kind {kind}")),
    }
}

unsafe fn required_slice<'a, T>(
    ptr: *const T,
    len: usize,
    name: &str,
) -> Result<&'a [T], String> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(format!("{name} pointer is null for nonzero length {len}"));
    }
    Ok(slice::from_raw_parts(ptr, len))
}

fn with_handle_mut<R>(
    raw: *mut c_void,
    f: impl FnOnce(&mut SolverHandle) -> Result<R, String>,
) -> Result<R, String> {
    if raw.is_null() {
        return Err("Clarabel solver pointer is null".to_string());
    }
    let handle = unsafe { &mut *(raw as *mut SolverHandle) };
    f(handle)
}

#[no_mangle]
pub extern "C" fn cpp_stream_clarabel_default_settings() -> CppStreamClarabelSettings {
    CppStreamClarabelSettings::default()
}

#[no_mangle]
pub unsafe extern "C" fn cpp_stream_clarabel_new(
    n: usize,
    m: usize,
    p_colptr: *const usize,
    p_rowval: *const usize,
    p_values: *const f64,
    p_nnz: usize,
    q: *const f64,
    a_colptr: *const usize,
    a_rowval: *const usize,
    a_values: *const f64,
    a_nnz: usize,
    b: *const f64,
    cone_kinds: *const u32,
    cone_dims: *const usize,
    n_cones: usize,
    settings: CppStreamClarabelSettings,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> {
        if n == 0 || m == 0 {
            return Err("Clarabel dimensions must be positive".to_string());
        }
        let p_colptr = required_slice(p_colptr, n + 1, "P colptr")?.to_vec();
        let p_rowval = required_slice(p_rowval, p_nnz, "P rowval")?.to_vec();
        let p_values = required_slice(p_values, p_nnz, "P values")?.to_vec();
        let q = required_slice(q, n, "q")?.to_vec();
        let a_colptr = required_slice(a_colptr, n + 1, "A colptr")?.to_vec();
        let a_rowval = required_slice(a_rowval, a_nnz, "A rowval")?.to_vec();
        let a_values = required_slice(a_values, a_nnz, "A values")?.to_vec();
        let b = required_slice(b, m, "b")?.to_vec();
        let cone_kinds = required_slice(cone_kinds, n_cones, "cone kinds")?;
        let cone_dims = required_slice(cone_dims, n_cones, "cone dims")?;
        let cones: Vec<_> = cone_kinds
            .iter()
            .zip(cone_dims.iter())
            .map(|(&kind, &dim)| cone(kind, dim))
            .collect::<Result<_, _>>()?;
        let total_cone_dim: usize = cone_dims.iter().sum();
        if total_cone_dim != m {
            return Err(format!(
                "cone dimensions sum to {total_cone_dim}; expected m={m}"
            ));
        }

        let p = CscMatrix::new(n, n, p_colptr, p_rowval, p_values);
        let a = CscMatrix::new(m, n, a_colptr, a_rowval, a_values);
        let mut native = DefaultSettings::<f64>::default();
        native.max_iter = settings.max_iter.max(1);
        native.verbose = settings.verbose;
        native.tol_gap_abs = settings.tol_gap_abs;
        native.tol_gap_rel = settings.tol_gap_rel;
        native.tol_feas = settings.tol_feas;
        native.max_threads = settings.max_threads.max(1);
        // These settings preserve the original fixed sparsity pattern so q/A/b
        // can be updated without rebuilding symbolic solver state.
        native.presolve_enable = false;
        native.input_sparse_dropzeros = false;

        let solver = DefaultSolver::new(&p, &q, &a, &b, &cones, native)
            .map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SolverHandle {
            solver,
            n,
            m,
            a_nnz,
        })) as *mut c_void)
    }));
    match result {
        Ok(Ok(pointer)) => pointer,
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic while constructing Clarabel solver");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cpp_stream_clarabel_update(
    raw: *mut c_void,
    q: *const f64,
    a_values: *const f64,
    b: *const f64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        with_handle_mut(raw, |handle| {
            let q = required_slice(q, handle.n, "q")?;
            let a_values = required_slice(a_values, handle.a_nnz, "A values")?;
            let b = required_slice(b, handle.m, "b")?;
            handle
                .solver
                .update_data(
                    &[] as &[f64; 0],
                    &BorrowedVectorUpdate(q),
                    &BorrowedMatrixUpdate(a_values),
                    &BorrowedVectorUpdate(b),
                )
                .map_err(|error| error.to_string())?;
            Ok(())
        })
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error);
            1
        }
        Err(_) => {
            set_error("panic while updating Clarabel data");
            2
        }
    }
}

#[no_mangle]
pub extern "C" fn cpp_stream_clarabel_solve(raw: *mut c_void) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        with_handle_mut(raw, |handle| {
            handle.solver.solve();
            Ok(handle.solver.solution.status as i32)
        })
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            set_error(error);
            -1
        }
        Err(_) => {
            set_error("panic while solving Clarabel problem");
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn cpp_stream_clarabel_info(raw: *mut c_void) -> CppStreamClarabelInfo {
    let result = catch_unwind(AssertUnwindSafe(|| {
        with_handle_mut(raw, |handle| {
            let solution = &handle.solver.solution;
            Ok(CppStreamClarabelInfo {
                status: solution.status as i32,
                iterations: solution.iterations,
                objective: solution.obj_val,
                solve_time: solution.solve_time,
                primal_residual: solution.r_prim,
                dual_residual: solution.r_dual,
            })
        })
    }));
    match result {
        Ok(Ok(info)) => info,
        Ok(Err(error)) => {
            set_error(error);
            CppStreamClarabelInfo {
                status: -1,
                ..Default::default()
            }
        }
        Err(_) => {
            set_error("panic while reading Clarabel info");
            CppStreamClarabelInfo {
                status: -2,
                ..Default::default()
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn cpp_stream_clarabel_x(raw: *mut c_void) -> *const f64 {
    match with_handle_mut(raw, |handle| Ok(handle.solver.solution.x.as_ptr())) {
        Ok(pointer) => pointer,
        Err(error) => {
            set_error(error);
            ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn cpp_stream_clarabel_z(raw: *mut c_void) -> *const f64 {
    match with_handle_mut(raw, |handle| Ok(handle.solver.solution.z.as_ptr())) {
        Ok(pointer) => pointer,
        Err(error) => {
            set_error(error);
            ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn cpp_stream_clarabel_s(raw: *mut c_void) -> *const f64 {
    match with_handle_mut(raw, |handle| Ok(handle.solver.solution.s.as_ptr())) {
        Ok(pointer) => pointer,
        Err(error) => {
            set_error(error);
            ptr::null()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cpp_stream_clarabel_free(raw: *mut c_void) {
    if !raw.is_null() {
        drop(Box::from_raw(raw as *mut SolverHandle));
    }
}

#[no_mangle]
pub extern "C" fn cpp_stream_clarabel_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}
