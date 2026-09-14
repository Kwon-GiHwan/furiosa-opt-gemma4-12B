
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated, Slice};

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    type QueryRows = m![Qs / 16];

    let x: DmTensorView<'_, bf16, Chip, Cluster, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, QueryRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, QueryRows, m![Qs % 16, H]> = weight.to_dm(&mut ctx.tdma);

    let contraction: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 16, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 16]>()
        .contract_lane::<m![Qs % 16], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = scaled.to_dm(&mut ctx.tdma);

    unsafe { output.reshape() }
}

type KvRows = m![Ps / 8];

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, KvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> {
    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, KvRows, m![Ps % 8, H]> = weight.to_dm(&mut ctx.tdma);

    let contraction: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 8, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 8, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 8]>()
        .contract_lane::<m![Ps % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 2], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();

    scaled.to_dm(&mut ctx.tdma)
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, Cluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, KvRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (unsafe { k.reshape() }, unsafe { v.reshape() })
}

pub(crate) fn project_output(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputCluster, Replicated, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {

    const CHUNK: usize = 2048;

    // Regroup 30-row contraction outputs before the original BF16 reduction.
    let p0 = output_partial(ctx, x, weight, 0);
    let p1 = output_partial(ctx, x, weight, CHUNK);

    let result = add_partials(ctx, &p0, &p1);
    let result = apply_output_channel_scale(ctx, &result, weight_scale);
    // SDK 0.6 does not lower cluster swaps. Stage only the 7.5 KiB final output
    // through HBM to gather both clusters without materializing decoded weights.
    let result_hbm: HbmTensor<bf16, Chip, m![H]> = result.to_hbm(&mut ctx.tdma);
    result_hbm.to_dm(&mut ctx.tdma)
}

// Distribute 3840 output rows over 128 live slices, with 30 rows per slice.
type OutputCluster = m![H / 1920];
// 64 live slices per cluster, with 30 contiguous output rows per slice.
type HiddenRows = m![H / 30 % 64, 1 # 4];
// Regroup four 30-row partials into each dense 120-row buffer.
type MergeRows = m![H / 120 % 16, 1 # 16];

// Split the K dimension across four adjacent slices per output-row group.
// The FP32 inter-slice sum restores the existing dense 30-row output layout.
pub(crate) fn project_output_k_sharded_distributed(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputCluster, m![H / 30 % 64, Qs / 1024], m![Qs % 1024]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]> {
    use crate::axes::Dummy8;

    type KRows = m![H / 30 % 64, Qs / 1024];

    // Use two native FP8 lanes for x ~= s * (hi + lo / 16). Both lanes
    // consume the same weight stream, so weights are neither decoded nor
    // read twice. The scale is computed from the input, not from fixtures.
    // Clear the FP32 sign bit before reduction; avoid squaring and sqrt.
    let scale_value: DmTensor<f32, Chip, OutputCluster, KRows, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Qs / 8 % 128], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 128], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, const { f32::from_bits(0x7fff_ffff) })
        .vector_narrow_split::<m![Qs / 4 % 256], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.0e-15)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let mut scales: DmTensor<f32, Chip, OutputCluster, KRows, m![Dummy8 % 2, 1 # 8]> =
        DmTensor::new();
    ctx.main
        .begin(scale_value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(scales.view_mut().tile::<m![Dummy8 % 2], 1, m![1 #{!} 2, 1 # 8]>(0));

    let scale_vrf: VrfTensor<f32, Chip, OutputCluster, KRows, m![1 # 8]> = ctx
        .sub
        .begin(scales.view().tile::<m![Dummy8 % 2], 1, m![1 # 2, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(scale_value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(scales.view_mut().tile::<m![Dummy8 % 2], 1, m![1 #{!} 2, 1 # 8]>(1));

    let normalized: DmTensor<f32, Chip, OutputCluster, KRows, m![Qs % 1024]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Qs / 8 % 128], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 128], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 256], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 128], m![Qs % 8]>()
        .vector_final()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    let mut parts: DmTensor<f8e4m3, Chip, OutputCluster, KRows, m![Dummy8 % 2, Qs % 1024]> =
        DmTensor::new();
    ctx.main
        .begin(normalized.view())
        .fetch::<m![Qs / 8 % 128], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 128], m![Qs % 8]>()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy8 % 2], 1, m![1 #{!} 2, Qs % 1024]>(0));

    let high_vrf: VrfTensor<f32, Chip, OutputCluster, KRows, m![Qs % 1024]> = ctx
        .sub
        .begin(parts.view().tile::<m![Dummy8 % 2], 1, m![1 # 2, Qs % 1024]>(0))
        .fetch::<m![Qs / 8 % 128], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 128], m![Qs % 8]>()
        .to_vrf();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![Qs / 8 % 128], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 128], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 256], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &high_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 16.0)
        .vector_widen_concat::<m![Qs / 8 % 128], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy8 % 2], 1, m![1 #{!} 2, Qs % 1024]>(1));

    let x_trf: TrfTensor<f8e4m3, Chip, OutputCluster, KRows, m![Dummy8 % 2], m![Qs % 1024]> = ctx
        .sub
        .begin(parts.view())
        .fetch::<m![Dummy8 % 2], m![Qs % 1024]>()
        .collect::<m![Dummy8 % 2, Qs / 32 % 32], m![Qs % 32]>()
        .to_trf();
    let scales_vrf: VrfTensor<f32, Chip, OutputCluster, KRows, m![Dummy8 % 2, 1 # 8]> = ctx
        .sub
        .begin(scales.view())
        .fetch::<m![Dummy8 % 2], m![1 # 8]>()
        .collect::<m![Dummy8 % 2], m![1 # 8]>()
        .to_vrf();

    let weight_f8: DmTensor<f8e4m3, Chip, OutputCluster, KRows, m![H % 30, Qs % 1024]> =
        weight.to_dm(&mut ctx.tdma);

    let partial: DmTensor<f32, Chip, OutputCluster, HiddenRows, m![H % 30]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![H % 30, Qs / 64 % 16], m![Qs % 64]>()
        .collect::<m![H % 30, Qs / 32 % 32], m![Qs % 32]>()
        .contract_outer::<m![H % 30, Qs / 64 % 16], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 30]>()
        .contract_lane::<m![H % 30, Dummy8 % 2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales_vrf)
        .vector_intra_slice_reduce::<Dummy8, m![H % 30], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 30]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .transpose::<m![H / 2 % 15], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();

    let partial: DmTensor<f32, Chip, OutputCluster, MergeRows, m![H % 120]> =
        partial.to_dm(&mut ctx.tdma);
    let result: DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    apply_output_channel_scale(ctx, &result, weight_scale)
}

fn output_partial(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputCluster, Replicated, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    offset: usize,
) -> DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]> {
    let x: DmTensorView<'_, bf16, Chip, OutputCluster, HiddenRows, m![Qs]> = unsafe { x.view().reshape() };
    let x = x.tile::<m![Qs], 2048, m![Qs = 2048 # 4096]>(offset);
    let x_trf: TrfTensor<bf16, Chip, OutputCluster, HiddenRows, m![1], m![Qs = 2048]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![Qs = 2048]>()
        .collect::<m![Qs = 2048 / 16], m![Qs = 2048 % 16]>()
        .to_trf();
    let weight_f8: DmTensor<f8e4m3, Chip, OutputCluster, HiddenRows, m![H % 30, Qs = 2048]> = weight
        .view()
        .tile::<m![Qs], 2048, m![H, Qs = 2048 # 4096]>(offset)
        .to_dm(&mut ctx.tdma);
    // Decode directly into the contraction stream without storing BF16 weights in DM.
    let partial: DmTensor<f32, Chip, OutputCluster, HiddenRows, m![H % 30]> = ctx.main
        .begin(weight_f8.view())
        .fetch::<m![H % 30, Qs = 2048 / 32], m![Qs = 2048 % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 30, Qs = 2048 / 16], m![Qs = 2048 % 16]>()
        .contract_outer::<m![H % 30, Qs = 2048 / 32], m![Qs = 2048 % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 30]>()
        .contract_lane::<m![H % 30], m![1 # 8]>(LaneMode::Interleaved)
        // Two FP32 results form an aligned 8-byte store packet.
        .transpose::<m![H / 2 % 15], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();

    // Thirty BF16 results are only 60 bytes. Move aligned FP32 partials first,
    // then retain the original BF16 rounding before any partial sums.
    let partial: DmTensor<f32, Chip, OutputCluster, MergeRows, m![H % 120]> = partial.to_dm(&mut ctx.tdma);
    ctx.main
        .begin(partial.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

fn add_partials(
    ctx: &mut Context,
    left: &DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]>,
    right: &DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]>,
) -> DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]> {
    let left: VrfTensor<f32, Chip, OutputCluster, MergeRows, m![H % 120]> = ctx
        .sub
        .begin(left.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    ctx.main
        .begin(right.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &left)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

fn apply_output_channel_scale(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]> {
    let weight_scale: DmTensor<bf16, Chip, OutputCluster, MergeRows, m![H % 120]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, OutputCluster, MergeRows, m![H % 120]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn prepare_qkv_input(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![H]> {
    use crate::axes::{Dummy8, Dummy256};
    // Eight independent 32-slice groups assemble the same hidden vector.
    let shards: DmTensor<bf16, Chip, Cluster, m![Dummy8, H / 480, 1 # 4], m![H % 480]> =
        x.to_dm(&mut ctx.tdma);
    let shared: DmTensor<bf16, Chip, Cluster, m![Dummy8, Dummy256 % 32], m![H]> = ctx.main
        .begin(shards.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<m![Dummy8, Dummy256 % 32], m![H / 480]>(
            SwitchConfig::CustomBroadcast { ring_size: 32 },
        )
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    // All 256 slices contain the identical vector; only the slice labels change.
    unsafe { shared.reshape::<Chip, Cluster, Replicated, m![H]>() }
}

// QKV02: split output rows, not the reduction dimension, across both clusters.
// Both copies of the input are initialized by DMA before their axes are relabeled.
pub(crate) type QkvDualQueryCluster = m![Qs / 2048];
type QkvDualKvCluster = m![Ps / 1024];
type QkvDualKvRows = m![Ps / 8 % 128, 1 # 2];

pub(crate) fn prepare_qkv_dual_input(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, QkvDualQueryCluster, Replicated, m![H]> {
    use crate::axes::{Dummy8, Dummy256};
    // An explicit HBM bridge initializes both clusters; padding is never a replica.
    let shared_hbm: HbmTensor<bf16, Chip, m![H]> = x.to_hbm(&mut ctx.tdma);
    let shards: DmTensor<bf16, Chip, QkvDualQueryCluster, m![Dummy8, H / 480, 1 # 4], m![H % 480]> =
        shared_hbm.to_dm(&mut ctx.tdma);
    let shared: DmTensor<bf16, Chip, QkvDualQueryCluster, m![Dummy8, Dummy256 % 32], m![H]> = ctx.main
        .begin(shards.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<m![Dummy8, Dummy256 % 32], m![H / 480]>(
            SwitchConfig::CustomBroadcast { ring_size: 32 },
        )
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    // The slice and cluster values are identical initialized copies of the input.
    unsafe { shared.reshape::<Chip, QkvDualQueryCluster, Replicated, m![H]>() }
}

pub(crate) fn project_qkv_dual_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, QkvDualQueryCluster, Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    type QueryRows = m![Qs / 8 % 256];

    let x: DmTensorView<'_, bf16, Chip, QkvDualQueryCluster, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, QkvDualQueryCluster, QueryRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, QueryRows, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);

    let contraction: DmTensor<bf16, Chip, QkvDualQueryCluster, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 8, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, QkvDualQueryCluster, QueryRows, m![Qs % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QkvDualQueryCluster, QueryRows, m![Qs % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 1], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, QkvDualQueryCluster, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 1], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 2], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 1], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    // Gather locally before crossing HBM, so writes have contiguous 4096-byte payloads.
    let gathered: DmTensor<bf16, Chip, QkvDualQueryCluster, Slice, m![Qs % 2048]> =
        scaled.to_dm(&mut ctx.tdma);
    let gathered: HbmTensor<bf16, Chip, m![Qs]> = gathered.to_hbm(&mut ctx.tdma);
    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = gathered.to_dm(&mut ctx.tdma);

    unsafe { output.reshape() }
}

fn project_qkv_dual_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, QkvDualKvCluster, QkvDualKvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> {
    let weight_f8: DmTensor<f8e4m3, Chip, QkvDualKvCluster, QkvDualKvRows, m![Ps % 8, H]> = weight.to_dm(&mut ctx.tdma);

    let contraction: DmTensor<bf16, Chip, QkvDualKvCluster, QkvDualKvRows, m![Ps % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 8, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 8, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 8]>()
        .contract_lane::<m![Ps % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, QkvDualKvCluster, QkvDualKvRows, m![Ps % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QkvDualKvCluster, QkvDualKvRows, m![Ps % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, QkvDualKvCluster, QkvDualKvRows, m![Ps % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 2], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();

    // Gather the valid rows; the padded slices never participate in arithmetic.
    let gathered: DmTensor<bf16, Chip, QkvDualKvCluster, Slice, m![Ps % 1024]> =
        scaled.to_dm(&mut ctx.tdma);
    let gathered: HbmTensor<bf16, Chip, m![Ps]> = gathered.to_hbm(&mut ctx.tdma);
    gathered.to_dm(&mut ctx.tdma)
}

pub(crate) fn project_qkv_dual_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, QkvDualQueryCluster, Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, QkvDualKvCluster, QkvDualKvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, QkvDualKvCluster, QkvDualKvRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_qkv_dual_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_qkv_dual_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (unsafe { k.reshape() }, unsafe { v.reshape() })
}
