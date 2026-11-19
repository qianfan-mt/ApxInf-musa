//! Optional loader for the experimentally validated MATE GDN native launcher.
//!
//! The launcher is deliberately opt-in through `APXINF_MUSA_NATIVE_GDN_LIB`.
//! It owns the `dlopen` handle for the lifetime of queued launches and always
//! uses the context's current stream, including graph-capture streams.

use std::ffi::{c_void, CStr, CString};
use std::ptr::NonNull;

use apxinf_core::{DType, Device, Tensor};

use crate::{MusaBuffer, MusaContext};

const SYMBOL: &CStr = c"mate_native_launch";

#[cfg(unix)]
struct Library(NonNull<c_void>);

#[cfg(unix)]
impl Library {
    fn open(path: &str) -> Result<Self, String> {
        let path = CString::new(path).map_err(|_| "native GDN library path contains NUL".to_string())?;
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        NonNull::new(handle)
            .map(Library)
            .ok_or_else(|| unsafe { last_dl_error() })
    }

    fn symbol(&self, name: &CStr) -> Result<*mut c_void, String> {
        unsafe { dl_symbol(self.0.as_ptr(), name) }
    }
}

#[cfg(unix)]
impl Drop for Library {
    fn drop(&mut self) {
        unsafe { libc::dlclose(self.0.as_ptr()); }
    }
}

#[cfg(unix)]
unsafe fn last_dl_error() -> String {
    let error = libc::dlerror();
    if error.is_null() {
        "dlopen failed without a loader diagnostic".to_string()
    } else {
        CStr::from_ptr(error).to_string_lossy().into_owned()
    }
}

#[cfg(unix)]
unsafe fn dl_symbol(handle: *mut c_void, name: &CStr) -> Result<*mut c_void, String> {
    libc::dlerror();
    let symbol = libc::dlsym(handle, name.as_ptr());
    if symbol.is_null() {
        Err(last_dl_error())
    } else {
        Ok(symbol)
    }
}

#[cfg(not(unix))]
struct Library;

#[cfg(not(unix))]
impl Library {
    fn open(_path: &str) -> Result<Self, String> {
        Err("native GDN loader is only supported on Unix".to_string())
    }
    fn symbol(&self, _name: &CStr) -> Result<*mut c_void, String> {
        unreachable!()
    }
}

type LaunchFn = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut c_void,
    i32,
    *mut c_void,
) -> i32;

/// An opt-in native Qwen3.8 GDN launcher.
///
/// The `Library` is held until this object is dropped. Callers must keep this
/// object alive while queued work or captured graphs can invoke the symbol.
pub struct NativeGdnLauncher {
    _library: Library,
    launch: LaunchFn,
    device_id: usize,
}

impl NativeGdnLauncher {
    /// Load `mate_native_launch` from the environment, or return `None` when
    /// the native path is not explicitly enabled.
    pub fn from_env(device_id: usize) -> Result<Option<Self>, String> {
        let Some(path) = std::env::var_os("APXINF_MUSA_NATIVE_GDN_LIB") else {
            return Ok(None);
        };
        let path = path
            .to_str()
            .ok_or_else(|| "native GDN library path is not valid UTF-8".to_string())?;
        let library = Library::open(path)?;
        let symbol = library.symbol(SYMBOL)?;
        // SAFETY: the adapter contract fixes the exported C ABI and symbol is
        // checked by dlsym before this cast. The library remains owned above.
        let launch = unsafe { std::mem::transmute::<*mut c_void, LaunchFn>(symbol) };
        Ok(Some(Self { _library: library, launch, device_id }))
    }

    pub fn device_id(&self) -> usize { self.device_id }

    /// Launch the fixed Qwen3.8 geometry asynchronously on `ctx.stream()`.
    pub fn launch(
        &self,
        ctx: &MusaContext,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a_log: &Tensor,
        a: &Tensor,
        dt_bias: &Tensor,
        b: &Tensor,
        state: &mut Tensor,
        output: &mut Tensor,
    ) -> Result<(), String> {
        if ctx.device_id() != self.device_id {
            return Err(format!("native GDN device mismatch: loader={}, context={}", self.device_id, ctx.device_id()));
        }
        require(q, DType::BF16, &[16, 128], "q")?;
        require(k, DType::BF16, &[16, 128], "k")?;
        require(v, DType::BF16, &[48, 128], "v")?;
        require(a_log, DType::F32, &[48], "A_log")?;
        require(a, DType::BF16, &[48], "a")?;
        require(dt_bias, DType::F32, &[48], "dt_bias")?;
        require(b, DType::BF16, &[48], "b")?;
        require(state, DType::F32, &[48, 128, 128], "state")?;
        require(output, DType::BF16, &[48, 128], "output")?;
        for (name, tensor) in [("q", q), ("k", k), ("v", v), ("A_log", a_log), ("a", a), ("dt_bias", dt_bias), ("b", b), ("state", state), ("output", output)] {
            if tensor.device() != Device::Musa(self.device_id) {
                return Err(format!("native GDN {name} is on the wrong device"));
            }
        }
        let q = MusaBuffer::from_tensor(q).map_err(|e| format!("q buffer: {e}"))?;
        let k = MusaBuffer::from_tensor(k).map_err(|e| format!("k buffer: {e}"))?;
        let v = MusaBuffer::from_tensor(v).map_err(|e| format!("v buffer: {e}"))?;
        let a_log = MusaBuffer::from_tensor(a_log).map_err(|e| format!("A_log buffer: {e}"))?;
        let a = MusaBuffer::from_tensor(a).map_err(|e| format!("a buffer: {e}"))?;
        let dt_bias = MusaBuffer::from_tensor(dt_bias).map_err(|e| format!("dt_bias buffer: {e}"))?;
        let b = MusaBuffer::from_tensor(b).map_err(|e| format!("b buffer: {e}"))?;
        let state = MusaBuffer::from_tensor(state).map_err(|e| format!("state buffer: {e}"))?;
        let output = MusaBuffer::from_tensor(output).map_err(|e| format!("output buffer: {e}"))?;
        // SAFETY: every buffer is validated above, the library is retained by
        // self, and the stream belongs to the same context/device.
        let status = unsafe {
            (self.launch)(q.ptr(), k.ptr(), v.ptr(), a_log.ptr(), a.ptr(), dt_bias.ptr(), b.ptr(), state.ptr(), output.ptr(), 1, ctx.stream().handle())
        };
        if status == 0 { Ok(()) } else { Err(format!("native GDN launcher returned status {status}")) }
    }
}

fn require(tensor: &Tensor, dtype: DType, shape: &[usize], name: &str) -> Result<(), String> {
    if tensor.dtype() != dtype || tensor.shape().dims() != shape {
        return Err(format!("native GDN {name}: expected {dtype:?} {shape:?}, got {:?} {:?}", tensor.dtype(), tensor.shape().dims()));
    }
    Ok(())
}
