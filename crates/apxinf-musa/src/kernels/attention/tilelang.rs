//! Exact-shape BF16 attention; all scratch belongs to the current execution arena.
use crate::context::MusaContext;
use crate::kernels::contracts::tensor_ptr;
use crate::kernels::gemm::tilelang::{self, InitFn, TilelangStatus};
use crate::kernels::TacticDispatch;
use apxinf_core::{Error, Result, Tensor};
use std::ffi::c_void;
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

type Three = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> i32;
type Four =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void) -> i32;
type Five = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
) -> i32;

pub(crate) struct TilelangPrefix {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub status: TilelangStatus,
    pub scratch: usize,
    pub pack: Four,
    pub pack_init: InitFn,
    pub pack_ready: AtomicBool,
    pub fast: Five,
    pub fast_init: InitFn,
    pub fast_ready: AtomicBool,
    pub fallback: Five,
    pub fallback_init: InitFn,
    pub fallback_ready: AtomicBool,
}
pub(crate) struct TilelangAction {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub status: TilelangStatus,
    pub scratch: usize,
    pub qk: Three,
    pub qk_init: InitFn,
    pub qk_ready: AtomicBool,
    pub softmax: Three,
    pub softmax_init: InitFn,
    pub softmax_ready: AtomicBool,
    pub pv: Four,
    pub pv_init: InitFn,
    pub pv_ready: AtomicBool,
}
pub(crate) struct TilelangVision {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub status: TilelangStatus,
    pub scratch: usize,
    pub run: Four,
    pub run_init: InitFn,
    pub run_ready: AtomicBool,
}

fn enabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    !tilelang::policy_off()
        && !*OFF.get_or_init(|| {
            std::env::var("APXINF_TILELANG_ATTENTION").is_ok_and(|v| v.eq_ignore_ascii_case("off"))
        })
}
fn check(status: i32, stage: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "TileLang attention {stage} failed: {status}"
        )))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_mqa(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    queries: usize,
    capacity: usize,
    keys: usize,
    tactic: TacticDispatch,
    workspace_limit: usize,
) -> Result<bool> {
    // The caller validates contiguous BF16 Q/O [queries,8,256] and K/V
    // [capacity,256]. Never replace partially filled caches or other geometries.
    // `RequireExact` promises the measured packed decode route. TileLang is a
    // separate provider and must not silently replace that tactic when the
    // caller has disabled fallback.
    if !enabled() || capacity != keys || tactic.requires_exact() {
        return Ok(false);
    }
    let stream = ctx.stream().handle();
    if let Some(op) = tilelang::TILELANG_PREFIX.iter().find(|op| {
        op.status == TilelangStatus::Enabled && (op.m, op.n, op.k) == (queries, keys, 256)
    }) {
        if op.scratch > workspace_limit {
            return Ok(false);
        }
        tilelang::initialize(op.pack_init, &op.pack_ready)?;
        tilelang::initialize(op.fast_init, &op.fast_ready)?;
        tilelang::initialize(op.fallback_init, &op.fallback_ready)?;
        let scratch = crate::workspace::output_buffer(ctx, op.scratch)?;
        let packed_bytes = keys.div_ceil(32) * 8192 * 2;
        // SAFETY: offsets partition exactly the declared arena allocation. Its
        // backing arena is retained by the prepared session for graph replay.
        unsafe {
            let kp = scratch.ptr();
            let vp = kp.byte_add(packed_bytes);
            let flags = vp.byte_add(packed_bytes);
            check(
                (op.pack)(tensor_ptr(k)?, tensor_ptr(v)?, kp, vp, stream),
                "pack",
            )?;
            check(
                (op.fast)(tensor_ptr(q)?, kp, vp, tensor_ptr(out)?, flags, stream),
                "fast",
            )?;
            // Always launch: device flags select the stable online recomputation.
            check(
                (op.fallback)(tensor_ptr(q)?, kp, vp, tensor_ptr(out)?, flags, stream),
                "fallback",
            )?;
        }
        return Ok(true);
    }
    if let Some(op) = tilelang::TILELANG_ACTION.iter().find(|op| {
        op.status == TilelangStatus::Enabled && (op.m, op.n, op.k) == (queries, keys, 256)
    }) {
        if op.scratch > workspace_limit {
            return Ok(false);
        }
        tilelang::initialize(op.qk_init, &op.qk_ready)?;
        tilelang::initialize(op.softmax_init, &op.softmax_ready)?;
        tilelang::initialize(op.pv_init, &op.pv_ready)?;
        let scratch = crate::workspace::output_buffer(ctx, op.scratch)?;
        // SAFETY: S FP32 [80,keys], P BF16 [80,keys], D FP32 [80]
        // occupy disjoint, aligned slices and all stages use the caller's stream.
        unsafe {
            let scores = scratch.ptr();
            let probs = scores.byte_add(80 * keys * 4);
            let den = probs.byte_add(80 * keys * 2);
            check(
                (op.qk)(tensor_ptr(q)?, tensor_ptr(k)?, scores, stream),
                "qk",
            )?;
            check((op.softmax)(scores, probs, den, stream), "softmax")?;
            check(
                (op.pv)(probs, tensor_ptr(v)?, den, tensor_ptr(out)?, stream),
                "pv",
            )?;
        }
        return Ok(true);
    }
    Ok(false)
}

pub(super) fn try_vision(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    batches: usize,
) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    let Some(op) = tilelang::TILELANG_VISION.iter().find(|op| {
        op.status == TilelangStatus::Enabled && (op.m, op.n, op.k) == (batches, 256, 72)
    }) else {
        return Ok(false);
    };
    debug_assert_eq!(op.scratch, 0);
    tilelang::initialize(op.run_init, &op.run_ready)?;
    // SAFETY: caller validates BF16 [batches*256,16,72] and device/outputs.
    unsafe {
        check(
            (op.run)(
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tensor_ptr(out)?,
                ctx.stream().handle(),
            ),
            "vision",
        )?;
    }
    Ok(true)
}
