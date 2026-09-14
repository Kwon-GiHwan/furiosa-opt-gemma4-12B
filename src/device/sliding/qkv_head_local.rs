use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, E, Gs, H, Ns, Ps};
use crate::axes::{Qs, Dummy2, Dummy8, Dummy256};
use crate::device::layout::Replicated;
use crate::{Chip, EPS};
use super::projection::QkvDualQueryCluster;

// Keep complete heads local through normalization, RoPE and the final cache write.
pub(crate) type HeadCluster = m![Ns / 4];
pub(crate) type QueryHeads = m![Ns % 4, Gs, 1 # 32];
pub(crate) type KvHeads = m![Ns % 4, 1 # 64];
pub(crate) type QueryHeadTensor = DmTensor<bf16, Chip, HeadCluster, QueryHeads, m![Ds]>;
pub(crate) type KvHeadTensor = DmTensor<bf16, Chip, HeadCluster, KvHeads, m![Ds]>;
type KvCluster = m![Ps / 1024];
type KvRows = m![Ps / 4 % 256];


type NativeSlices = m![Dummy256];
pub(crate) type NativeTrf = TrfTensor<f8e4m3, Chip, QkvDualQueryCluster, NativeSlices, m![Dummy2], m![H]>;
pub(crate) type NativeScale = VrfTensor<f32, Chip, QkvDualQueryCluster, NativeSlices, m![1 # 8]>;

pub(crate) fn project_query(
    ctx: &mut Context,
    x_trf: &NativeTrf,
    x_scale: &NativeScale,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> QueryHeadTensor {
    let weight: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, m![Qs / 8 % 256], m![Qs % 8, H]> =
        weight.to_dm(&mut ctx.tdma);
    // Relabel fully initialized row partitions, without changing their wire order.
    let weight: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, NativeSlices, m![Qs % 8, H]> =
        unsafe { weight.reshape() };
    let contraction: DmTensor<bf16, Chip, QkvDualQueryCluster, NativeSlices, m![Qs % 8]> = ctx.main
        .begin(weight.view())
        .fetch::<m![Qs % 8, H / 32], m![H % 32]>()
        .collect::<m![Qs % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8, Dummy2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale)
        .vector_intra_slice_reduce::<Dummy2, m![Qs % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    let scaled = contraction;
    // Qs is flattened [Ns, Gs, Ds]. This only relabels the same wire order.
    let rows: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { scaled.reshape() };
    // Each 32-slice group owns one query head. Gather through the switch,
    // leaving the DMA context available for K/V weight transfers.
    let heads: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Gs, Dummy256 % 32], m![Ds]> = ctx.main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 8]>()
        .switch::<m![Ns % 4, Gs, Dummy256 % 32], m![Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    // The switch initialized every replica. Keep the first replica of each head.
    let heads: QueryHeadTensor = unsafe { heads.reshape() };
    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, HeadCluster, QueryHeads, m![Ds]> =
        weight_scale.to_dm(&mut ctx.tdma);
    scale_head(ctx, &heads, &weight_scale)
}


fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &NativeTrf,
    x_scale: &NativeScale,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> KvHeadTensor {
    let weight: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]> =
        weight.to_dm(&mut ctx.tdma);
    // Relabel fully initialized row partitions, without changing their wire order.
    let weight: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, NativeSlices, m![Ps % 4, H]> =
        unsafe { weight.reshape() };
    let contraction: DmTensor<bf16, Chip, QkvDualQueryCluster, NativeSlices, m![Ps % 4]> = ctx.main
        .begin(weight.view())
        .fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4, Dummy2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale)
        .vector_intra_slice_reduce::<Dummy2, m![Ps % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    let scaled = contraction;
    // Ps is flattened [Ns, Ds]; all 256 row slices contain valid data.
    let rows: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Ds / 4], m![Ds % 4]> =
        unsafe { scaled.reshape() };
    // Each 64-slice group owns one key/value head.
    let heads: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Dummy256 % 64], m![Ds]> = ctx.main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 4]>()
        .switch::<m![Ns % 4, Dummy256 % 64], m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    // Only hide already initialized replicas; padding never creates data.
    let heads: KvHeadTensor = unsafe { heads.reshape() };
    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, HeadCluster, KvHeads, m![Ds]> =
        weight_scale.to_dm(&mut ctx.tdma);
    scale_head(ctx, &heads, &weight_scale)
}



pub(crate) fn project_key_value(
    ctx: &mut Context,
    x_trf: &NativeTrf,
    x_scale: &NativeScale,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (KvHeadTensor, KvHeadTensor) {
    let k = project_one_kv_matrix(ctx, x_trf, x_scale, k_weight, k_weight_scale);
    let v = project_one_kv_matrix(ctx, x_trf, x_scale, v_weight, v_weight_scale);
    (k, v)
}

fn inverse_root_mean_square<S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
) -> VrfTensor<f32, Chip, HeadCluster, S, m![1 # 8]> {
    let mean_square: DmTensor<f32, Chip, HeadCluster, S, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Ds::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let inv_rms: DmTensor<f32, Chip, HeadCluster, S, m![1 # 8]> = ctx.main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        // Reuse one reciprocal per reduction group; FpDiv follows the Sqrt FP stage.
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    ctx.sub
        .begin(inv_rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

pub(crate) fn normalize_weighted<S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    let inv_rms_vrf = inverse_root_mean_square(ctx, x);
    let weight: DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> = weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx.sub
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_value(
    ctx: &mut Context,
    x: &KvHeadTensor,
) -> KvHeadTensor {
    let inv_rms_vrf = inverse_root_mean_square(ctx, x);
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn rope_one<S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    cos: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    sin: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    let cos_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx.sub
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let sin_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx.sub
        .begin(sin.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // Preserve the official half-swap and use the supplied sine table unchanged.
    let first_half = x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half = x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);
    let mut rotated: DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> = DmTensor::new();
    ctx.main
        .begin(first_half)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotated.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));
    ctx.main
        .begin(second_half)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotated.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let sin_product: DmTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx.main
        .begin(rotated.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let sin_product_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx.sub
        .begin(sin_product.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    // The cosine product feeds the final addition without a DM round trip.
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &sin_product_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn apply_rope(
    ctx: &mut Context,
    q: &QueryHeadTensor,
    k: &KvHeadTensor,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (QueryHeadTensor, KvHeadTensor) {
    use crate::device::layout::{Cluster, Slice};

    // A scalar indirect gather reads one table row. Keep that row free of head
    // axes, then use ordinary DMA to explicitly populate every head replica.
    let cos_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> =
        cos.dma_gather_scaled(rope_offset);
    let sin_row: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> =
        sin.dma_gather_scaled(rope_offset);
    let cos_row: HbmTensor<bf16, Chip, m![Ds]> = cos_row.to_hbm(&mut ctx.tdma);
    let sin_row: HbmTensor<bf16, Chip, m![Ds]> = sin_row.to_hbm(&mut ctx.tdma);
    let q_cos: QueryHeadTensor = cos_row.to_dm(&mut ctx.tdma);
    let q_sin: QueryHeadTensor = sin_row.to_dm(&mut ctx.tdma);
    let q = rope_one(ctx, q, &q_cos, &q_sin);
    // Both Gs copies were initialized by DMA; retain Gs=0 for each key head.
    let cos: KvHeadTensor = unsafe { q_cos.reshape() };
    let sin: KvHeadTensor = unsafe { q_sin.reshape() };
    let k = rope_one(ctx, k, &cos, &sin);
    (q, k)
}

fn scale_head<S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    weight: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    let weight_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx.sub
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_native_input(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (NativeTrf, NativeScale) {
    type Shards = m![Dummy256 / 32, H / 120];
    type Copies = m![Dummy256 / 32, Dummy256 % 32];
    // Every cluster and 32-slice group receives all input shards.
    let x: DmTensor<bf16, Chip, QkvDualQueryCluster, Shards, m![H % 120]> =
        x.to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, QkvDualQueryCluster, Shards, m![H % 120]> =
        weight.to_dm(&mut ctx.tdma);
    let partial: DmTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let mean: DmTensor<f32, Chip, QkvDualQueryCluster, Copies, m![1 # 8]> = ctx.main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Copies, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let inv_rms: DmTensor<f32, Chip, QkvDualQueryCluster, Copies, m![1 # 8]> = ctx.main
        .begin(mean.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        // Reuse one reciprocal per reduction group; FpDiv follows the Sqrt FP stage.
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Retain only the shard lanes from the initialized all-reduce replicas.
    let inv_rms: DmTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> =
        unsafe { inv_rms.reshape() };
    let inv_rms_vrf: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> = ctx.sub
        .begin(inv_rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let weight_vrf: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx.sub
        .begin(weight.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let normalized: DmTensor<bf16, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx.main
        .begin(x.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    // Calibrate from the current normalized input. The high component stays
    // within FP8's finite range; a second FP8 component retains its residual.
    let local_max: DmTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> = ctx.main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let global_max: DmTensor<f32, Chip, QkvDualQueryCluster, Copies, m![1 # 8]> = ctx.main
        .begin(local_max.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Copies, m![1]>(InterSliceReduceOpF32::Max)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let scale: DmTensor<f32, Chip, QkvDualQueryCluster, Copies, m![1 # 8]> = ctx.main
        .begin(global_max.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.00390625)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let scale_for_shards: DmTensorView<'_, f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> =
        unsafe { scale.view().reshape() };
    let quant_scale: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> = ctx.sub
        .begin(scale_for_shards)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let scaled: DmTensor<f32, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx.main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &quant_scale)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    // Integer multiples of 16 in [-256, 256] are exactly representable in
    // e4m3. Round with an FP32 unit-ULP bias, eliminating the FP8 decode table.
    let high_integer: DmTensor<f32, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx.main
        .begin(scaled.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625)
        .vector_fp_binary(FpBinaryOp::AddF, 12582912.0)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    let high: DmTensor<f32, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx.main
        .begin(high_integer.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, 12582912.0)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 16.0)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    let mut parts: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, Shards, m![Dummy2, H % 120]> =
        DmTensor::new();
    ctx.main
        .begin(high.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 120]>(0));
    let high_vrf: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx.sub
        .begin(high.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    ctx.main
        .begin(scaled.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &high_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 120]>(1));

    // Broadcast aligned 8-byte pieces. Keep the transposed DM order explicit.
    let shared: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, Copies, m![Dummy2, H / 8 % 15, H / 120, H % 8]> = ctx.main
        .begin(parts.view())
        .fetch::<m![Dummy2, H / 8 % 15], m![H % 8]>()
        .switch::<Copies, m![Dummy2, H / 8 % 15, H / 120]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Dummy2, H / 8 % 15, H / 120], m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let shared: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, NativeSlices, m![Dummy2, H / 8 % 15, H / 120, H % 8]> =
        unsafe { shared.reshape() };
    // Restore H order with an actual DM copy before the contiguous TRF load.
    let dense: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, NativeSlices, m![Dummy2, H]> = ctx.main
        .begin(shared.view())
        .fetch::<m![Dummy2, H / 8], m![H % 8]>()
        .collect::<m![Dummy2, H / 8], m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let trf: NativeTrf = ctx.sub
        .begin(dense.view())
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();
    let scale: DmTensor<f32, Chip, QkvDualQueryCluster, NativeSlices, m![1 # 8]> =
        unsafe { scale.reshape() };
    let scale_vrf: NativeScale = ctx.sub
        .begin(scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    (trf, scale_vrf)
}
