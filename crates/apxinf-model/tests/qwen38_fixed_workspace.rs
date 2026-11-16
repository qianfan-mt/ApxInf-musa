#![cfg(feature = "musa")]

use apxinf_core::Tensor;
use apxinf_model::musa_qwen38::model::fixed::LayerWorkspace;
use apxinf_model::musa_qwen38::model::{
    Qwen38GdnBf16Block, Qwen38GdnBf16State, Qwen38GdnBf16Weights,
};
use apxinf_model::musa_qwen38::model::mlp::Qwen38MlpBf16;
use apxinf_musa::{transfers, MusaContext};
use half::bf16;

fn upload(ctx: &MusaContext, dims: Vec<usize>, values: Vec<f32>) -> Tensor {
    transfers::to_device(
        &Tensor::from_bf16(
            dims,
            &values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
        )
        .unwrap(),
        ctx.device_id(),
    )
    .unwrap()
}

fn constant(ctx: &MusaContext, dims: Vec<usize>, value: f32) -> Tensor {
    upload(ctx, dims.clone(), vec![value; dims.iter().product()])
}

fn projection(ctx: &MusaContext, out: usize, value: f32) -> Tensor {
    let mut values = vec![0.0; 5120 * out];
    for column in 0..out {
        values[column] = value;
    }
    upload(ctx, vec![5120, out], values)
}

fn test_weights(ctx: &MusaContext) -> (Qwen38GdnBf16Weights, Qwen38MlpBf16) {
    let mut out = vec![0.0; 6144 * 5120];
    for column in 0..5120 {
        out[column * 5120 + column] = 0.25;
    }
    let gdn = Qwen38GdnBf16Weights {
        input_norm: constant(ctx, vec![5120], 0.5),
        qkv: projection(ctx, 10240, 0.5),
        z: projection(ctx, 6144, 0.25),
        a: projection(ctx, 48, 0.25),
        b: projection(ctx, 48, -0.25),
        conv: constant(ctx, vec![10240, 4], 0.25),
        a_log: constant(ctx, vec![48], 0.0),
        dt_bias: constant(ctx, vec![48], 0.125),
        gdn_norm: constant(ctx, vec![128], 0.75),
        out: upload(ctx, vec![6144, 5120], out),
    };
    let mut gate = vec![0.0; 5120 * 17408];
    let mut up = vec![0.0; 5120 * 17408];
    for column in 0..17408 {
        gate[column] = [-0.5, 0.0, 0.5][column % 3];
        up[column] = 0.25;
    }
    let mut down = vec![0.0; 17408 * 5120];
    for column in 0..5120 {
        down[column * 5120 + column] = 0.5;
    }
    let mlp = Qwen38MlpBf16 {
        norm: constant(ctx, vec![5120], 0.5),
        gate: upload(ctx, vec![5120, 17408], gate),
        up: upload(ctx, vec![5120, 17408], up),
        down: upload(ctx, vec![17408, 5120], down),
    };
    (gdn, mlp)
}

#[test]
fn fixed_workspace_matches_eager_output_and_state() {
    let ctx = MusaContext::new(0).unwrap();
    let (gdn_weights, mlp) = test_weights(&ctx);
    // Keep two identical GDN weight copies because the eager and fixed paths
    // execute independently and own their layer objects.
    let fixed_gdn = Qwen38GdnBf16Weights {
        input_norm: gdn_weights.input_norm.clone(),
        qkv: gdn_weights.qkv.clone(),
        z: gdn_weights.z.clone(),
        a: gdn_weights.a.clone(),
        b: gdn_weights.b.clone(),
        conv: gdn_weights.conv.clone(),
        a_log: gdn_weights.a_log.clone(),
        dt_bias: gdn_weights.dt_bias.clone(),
        gdn_norm: gdn_weights.gdn_norm.clone(),
        out: gdn_weights.out.clone(),
    };
    let eager = Qwen38GdnBf16Block::new(gdn_weights);
    let fixed = Qwen38GdnBf16Block::new(fixed_gdn);
    let mut eager_state = Qwen38GdnBf16State::zeroed(&ctx).unwrap();
    let mut fixed_state = Qwen38GdnBf16State::zeroed(&ctx).unwrap();
    let mut workspace = LayerWorkspace::new(&ctx).unwrap();
    let input = constant(&ctx, vec![1, 5120], 1.0);
    for token in 0..2 {
        let eager_output = mlp.forward(&ctx, &eager.forward_decode(&ctx, &input, &mut eager_state).unwrap()).unwrap();
        let fixed_output = workspace.forward_decode(&ctx, &input, &mut fixed_state, &fixed, &mlp).unwrap();
        ctx.synchronize().unwrap();
        let eager_host = transfers::to_cpu(&eager_output).unwrap();
        let fixed_host = transfers::to_cpu(&fixed_output).unwrap();
        for (index, (a, b)) in eager_host.as_bf16().unwrap().iter().zip(fixed_host.as_bf16().unwrap()).enumerate() {
            assert!((a.to_f32() - b.to_f32()).abs() < 0.01, "token {token}, column {index}: {} vs {}", a.to_f32(), b.to_f32());
        }
        let eager_recurrent = transfers::to_cpu(&eager_state.recurrent).unwrap();
        let fixed_recurrent = transfers::to_cpu(&fixed_state.recurrent).unwrap();
        for (a, b) in eager_recurrent.as_f32().unwrap().iter().zip(fixed_recurrent.as_f32().unwrap()) {
            assert!((a - b).abs() < 2e-5, "recurrent state {a} vs {b}");
        }
        assert_eq!(
            transfers::to_cpu(&eager_state.conv_window).unwrap().as_f32().unwrap(),
            transfers::to_cpu(&fixed_state.conv_window).unwrap().as_f32().unwrap(),
        );
    }
}

#[test]
fn fixed_workspace_reuses_storage_between_calls() {
    let ctx = MusaContext::new(0).unwrap();
    let (gdn_weights, mlp) = test_weights(&ctx);
    let gdn = Qwen38GdnBf16Block::new(gdn_weights);
    let mut state = Qwen38GdnBf16State::zeroed(&ctx).unwrap();
    let mut workspace = LayerWorkspace::new(&ctx).unwrap();
    let input = constant(&ctx, vec![1, 5120], 1.0);
    let first = workspace.forward_decode(&ctx, &input, &mut state, &gdn, &mlp).unwrap();
    workspace.forward_decode(&ctx, &input, &mut state, &gdn, &mlp).unwrap();
    assert_eq!(first.shape().dims(), &[1, 5120]);
}
