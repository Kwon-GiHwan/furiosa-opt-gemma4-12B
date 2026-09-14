use furiosa_opt_std::prelude::*;
use crate::Chip;
use crate::axes::{Dummy2, Dummy8, Dummy256, H, L};
use crate::device::layout::{Cluster, Slice};

const INVSQRT2: f32 = 0.70710678118f32;
// Wider K chunks make the FP4 and block-scale DMA less fragmented.
type UpCluster = m![L / 7680];
type UpGateKRows = m![L / 30 % 256];
pub type UpGateRows = m![L / 60 % 128, 1 # 2];
pub type UpGateRowsPaired = m![L / 120 % 64, 1 # 4];
type DownCluster = m![H / 1920];
pub type DownRows = m![H / 60 % 32, 1 # 8];
pub type DownRowsByColumns = m![H / 15 % 128, L / 7680];
type DownPartialRows = m![H / 15 % 128, 1 # 2];

fn prepare_up_input(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> (
    HbmTensor<f8e4m3, Chip, m![H / 960, Dummy2, H % 960]>,
    HbmTensor<f32, Chip, m![H / 960, Dummy2]>,
) {
    // Quantize disjoint 960-element groups once, not once per output row.
    let x: DmTensor<bf16, Chip, Cluster, m![H / 960, 1 # 64], m![H % 960]> =
        x.to_dm(&mut ctx.tdma);
    // Clear the FP32 sign bit before reduction; avoid squaring and sqrt.
    let scale_value: DmTensor<f32, Chip, Cluster, m![H / 960, 1 # 64], m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![H / 8 % 120], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, const { f32::from_bits(0x7fff_ffff) })
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.0e-15f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let mut scales: DmTensor<f32, Chip, Cluster, m![H / 960, 1 # 64], m![Dummy2, 1 # 8]> = DmTensor::new();
    ctx.main.begin(scale_value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(scales.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, 1 # 8]>(0));
    ctx.main.begin(scale_value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(scales.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, 1 # 8]>(1));

    let high_scale: VrfTensor<f32, Chip, Cluster, m![H / 960, 1 # 64], m![1 # 8]> = ctx.sub
        .begin(scales.view().tile::<m![Dummy2], 1, m![1 # 2, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let normalized: DmTensor<f32, Chip, Cluster, m![H / 960, 1 # 64], m![H % 960]> = ctx.main
        .begin(x.view())
        .fetch::<m![H / 8 % 120], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_div(&high_scale)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    let mut parts: DmTensor<f8e4m3, Chip, Cluster, m![H / 960, 1 # 64], m![Dummy2, H % 960]> = DmTensor::new();
    ctx.main.begin(normalized.view())
        .fetch::<m![H / 8 % 120], m![H % 8]>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, H % 960]>(0));

    let high: VrfTensor<f32, Chip, Cluster, m![H / 960, 1 # 64], m![H % 960]> = ctx.sub
        .begin(parts.view().tile::<m![Dummy2], 1, m![1 # 2, H % 960]>(0))
        .fetch::<m![H / 8 % 120], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .to_vrf();
    ctx.main.begin(normalized.view())
        .fetch::<m![H / 8 % 120], m![H % 8]>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &high)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 16.0f32)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, H % 960]>(1));

    // A high/low group is 1920 bytes in HBM.
    let parts: HbmTensor<f8e4m3, Chip, m![H / 960, Dummy2, H % 960]> =
        parts.to_hbm(&mut ctx.tdma);
    // Gather the tiny scales before HBM, avoiding one short write per group.
    let scales: DmTensor<f32, Chip, Cluster, m![H / 960, 1 # 64], m![Dummy2]> = ctx.main
        .begin(scales.view())
        .fetch::<m![Dummy2], m![1 # 8]>()
        .collect::<m![Dummy2], m![1 # 8]>()
        .transpose::<m![1], m![Dummy2 # 8]>()
        .commit_trim::<m![Dummy2]>()
        .commit();
    let scales: DmTensor<f32, Chip, Cluster, Slice, m![H / 960, Dummy2]> =
        scales.to_dm(&mut ctx.tdma);
    let scales: HbmTensor<f32, Chip, m![H / 960, Dummy2]> =
        scales.to_hbm(&mut ctx.tdma);
    (parts, scales)
}

fn prepare_down_input(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, UpCluster, UpGateRowsPaired, m![L % 120]>,
) -> (
    HbmTensor<f8e4m3, Chip, m![L / 960, Dummy2, L % 960]>,
    HbmTensor<f32, Chip, m![L / 960, Dummy2]>,
) {
    // Quantize disjoint 960-element groups once, not once per output row.
    let x: DmTensor<bf16, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![L % 960]> =
        x.to_dm(&mut ctx.tdma);
    // Clear the FP32 sign bit before reduction; avoid squaring and sqrt.
    let scale_value: DmTensor<f32, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![L / 8 % 120], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 120], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, const { f32::from_bits(0x7fff_ffff) })
        .vector_narrow_split::<m![L / 4 % 240], m![L % 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.0e-15f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let mut scales: DmTensor<f32, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![Dummy2, 1 # 8]> = DmTensor::new();
    ctx.main.begin(scale_value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(scales.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, 1 # 8]>(0));
    ctx.main.begin(scale_value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(scales.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, 1 # 8]>(1));

    let high_scale: VrfTensor<f32, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![1 # 8]> = ctx.sub
        .begin(scales.view().tile::<m![Dummy2], 1, m![1 # 2, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let normalized: DmTensor<f32, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![L % 960]> = ctx.main
        .begin(x.view())
        .fetch::<m![L / 8 % 120], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 120], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 240], m![L % 4]>()
        .vector_fp_div(&high_scale)
        .vector_widen_concat::<m![L / 8 % 120], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let mut parts: DmTensor<f8e4m3, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![Dummy2, L % 960]> = DmTensor::new();
    ctx.main.begin(normalized.view())
        .fetch::<m![L / 8 % 120], m![L % 8]>()
        .collect::<m![L / 8 % 120], m![L % 8]>()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, L % 960]>(0));

    let high: VrfTensor<f32, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![L % 960]> = ctx.sub
        .begin(parts.view().tile::<m![Dummy2], 1, m![1 # 2, L % 960]>(0))
        .fetch::<m![L / 8 % 120], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 120], m![L % 8]>()
        .to_vrf();
    ctx.main.begin(normalized.view())
        .fetch::<m![L / 8 % 120], m![L % 8]>()
        .collect::<m![L / 8 % 120], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 240], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &high)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 16.0f32)
        .vector_widen_concat::<m![L / 8 % 120], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, L % 960]>(1));

    // A high/low group is 1920 bytes in HBM.
    let parts: HbmTensor<f8e4m3, Chip, m![L / 960, Dummy2, L % 960]> =
        parts.to_hbm(&mut ctx.tdma);
    // Gather the tiny scales before HBM, avoiding one short write per group.
    let scales: DmTensor<f32, Chip, UpCluster, m![L / 960 % 8, 1 # 32], m![Dummy2]> = ctx.main
        .begin(scales.view())
        .fetch::<m![Dummy2], m![1 # 8]>()
        .collect::<m![Dummy2], m![1 # 8]>()
        .transpose::<m![1], m![Dummy2 # 8]>()
        .commit_trim::<m![Dummy2]>()
        .commit();
    let scales: DmTensor<f32, Chip, UpCluster, Slice, m![L / 960 % 8, Dummy2]> =
        scales.to_dm(&mut ctx.tdma);
    let scales: HbmTensor<f32, Chip, m![L / 960, Dummy2]> =
        scales.to_hbm(&mut ctx.tdma);
    (parts, scales)
}



fn load_up_input(
    ctx: &mut Context,
    parts: &HbmTensor<f8e4m3, Chip, m![H / 960, Dummy2, H % 960]>,
) -> TrfTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![Dummy2], m![H % 3840]> {
    // Replicate into small independent sub-rings, then share 480-value
    // fragments. Every received K value is valid; contraction has no K pad.
    let shards: DmTensor<f8e4m3, Chip, UpCluster, m![Dummy8, H / 480, 1 # 4], m![Dummy2, H % 480]> =
        parts.to_dm(&mut ctx.tdma);
    let shared: DmTensor<f8e4m3, Chip, UpCluster, m![Dummy8, Dummy256 % 32], m![Dummy2, H % 3840]> = ctx.main
        .begin(shards.view())
        .fetch::<m![Dummy2], m![H % 480]>()
        .switch::<m![Dummy8, Dummy256 % 32], m![Dummy2, H / 480 % 8]>(
            SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![Dummy2, H / 32 % 120], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    // Safety: only the fully populated, 256-entry Slice dimension is
    // relabeled in the same physical order. Chip, Cluster, Element and
    // K-shard order are unchanged; all row replicas hold identical inputs.
    let shared = unsafe {
        shared.reshape::<Chip, UpCluster, UpGateKRows, m![Dummy2, H % 3840]>()
    };
    ctx.sub.begin(shared.view())
        .fetch::<m![Dummy2], m![H % 3840]>()
        .collect::<m![Dummy2, H / 32 % 120], m![H % 32]>()
        .to_trf()
}

fn project_up_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![Dummy2], m![H % 3840]>,
    x_scales: &DmTensor<f32, Chip, UpCluster, UpGateKRows, m![H / 960 % 4, Dummy2]>,
    weight: &DmTensor<f4e2m1, Chip, UpCluster, UpGateKRows, m![L % 30, H % 3840]>,
    scale: &DmTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![L % 30, H / 16 % 240]>,
) -> DmTensor<bf16, Chip, UpCluster, UpGateRows, m![L % 60]> {
    let input_scale_vrf: VrfTensor<f32, Chip, UpCluster, UpGateKRows, m![H / 960 % 4, Dummy2]> = ctx.sub
        .begin(x_scales.view())
        .fetch::<m![1], m![H / 960 % 4, Dummy2]>()
        .collect::<m![1], m![H / 960 % 4, Dummy2]>()
        .to_vrf();
    let mut partials: DmTensor<f32, Chip, UpCluster, UpGateKRows, m![L % 30, Dummy2]> = DmTensor::new();
    {
        // Decode only the row block consumed by the following contractions.
        let native_tile: DmTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![L % 30 = 30, H % 3840]> = ctx.main
            .begin(weight.view().tile::<m![L % 30], 30, m![L % 30 = 30 # 30, H % 3840]>(0))
            .fetch::<m![L % 30 = 30, H / 64 % 60], m![H % 64]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 30 = 30, H / 32 % 120], m![H % 32]>()
            .commit_trim::<m![H % 32]>()
            .commit();
{
let block_scale: VrfTensor<f32, Chip, UpCluster, UpGateKRows, m![L % 30 = 6, H / 16 % 240]> = ctx.sub
            .begin(scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16 % 240]>(0))
            .fetch::<m![L % 30 = 6], m![H / 16 % 240]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = 6, H / 128 % 30], m![H / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![L % 30 = 30], 6, m![L % 30 = 6 # 30, H % 3840]>(0))
            .fetch::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .collect::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .contract_outer::<m![L % 30 = 6, H / 64 % 60], m![H % 64], _, _, _>(x_trf)
            .contract_packet::<m![H / 16 % 4]>()
            .contract_time::<m![L % 30 = 6, H / 64 % 60]>()
            .contract_lane::<m![L % 30 = 6, H / 64 % 60, Dummy2], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 30 = 6, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![L % 30 = 6], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30, Dummy2]>(0));
    
}
{
let block_scale: VrfTensor<f32, Chip, UpCluster, UpGateKRows, m![L % 30 = 6, H / 16 % 240]> = ctx.sub
            .begin(scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16 % 240]>(6))
            .fetch::<m![L % 30 = 6], m![H / 16 % 240]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = 6, H / 128 % 30], m![H / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![L % 30 = 30], 6, m![L % 30 = 6 # 30, H % 3840]>(6))
            .fetch::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .collect::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .contract_outer::<m![L % 30 = 6, H / 64 % 60], m![H % 64], _, _, _>(x_trf)
            .contract_packet::<m![H / 16 % 4]>()
            .contract_time::<m![L % 30 = 6, H / 64 % 60]>()
            .contract_lane::<m![L % 30 = 6, H / 64 % 60, Dummy2], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 30 = 6, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![L % 30 = 6], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30, Dummy2]>(6));
    
}
{
let block_scale: VrfTensor<f32, Chip, UpCluster, UpGateKRows, m![L % 30 = 6, H / 16 % 240]> = ctx.sub
            .begin(scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16 % 240]>(12))
            .fetch::<m![L % 30 = 6], m![H / 16 % 240]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = 6, H / 128 % 30], m![H / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![L % 30 = 30], 6, m![L % 30 = 6 # 30, H % 3840]>(12))
            .fetch::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .collect::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .contract_outer::<m![L % 30 = 6, H / 64 % 60], m![H % 64], _, _, _>(x_trf)
            .contract_packet::<m![H / 16 % 4]>()
            .contract_time::<m![L % 30 = 6, H / 64 % 60]>()
            .contract_lane::<m![L % 30 = 6, H / 64 % 60, Dummy2], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 30 = 6, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![L % 30 = 6], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30, Dummy2]>(12));
    
}
{
let block_scale: VrfTensor<f32, Chip, UpCluster, UpGateKRows, m![L % 30 = 6, H / 16 % 240]> = ctx.sub
            .begin(scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16 % 240]>(18))
            .fetch::<m![L % 30 = 6], m![H / 16 % 240]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = 6, H / 128 % 30], m![H / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![L % 30 = 30], 6, m![L % 30 = 6 # 30, H % 3840]>(18))
            .fetch::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .collect::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .contract_outer::<m![L % 30 = 6, H / 64 % 60], m![H % 64], _, _, _>(x_trf)
            .contract_packet::<m![H / 16 % 4]>()
            .contract_time::<m![L % 30 = 6, H / 64 % 60]>()
            .contract_lane::<m![L % 30 = 6, H / 64 % 60, Dummy2], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 30 = 6, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![L % 30 = 6], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30, Dummy2]>(18));
    
}
{
let block_scale: VrfTensor<f32, Chip, UpCluster, UpGateKRows, m![L % 30 = 6, H / 16 % 240]> = ctx.sub
            .begin(scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16 % 240]>(24))
            .fetch::<m![L % 30 = 6], m![H / 16 % 240]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = 6, H / 128 % 30], m![H / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![L % 30 = 30], 6, m![L % 30 = 6 # 30, H % 3840]>(24))
            .fetch::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .collect::<m![L % 30 = 6, H / 32 % 120], m![H % 32]>()
            .contract_outer::<m![L % 30 = 6, H / 64 % 60], m![H % 64], _, _, _>(x_trf)
            .contract_packet::<m![H / 16 % 4]>()
            .contract_time::<m![L % 30 = 6, H / 64 % 60]>()
            .contract_lane::<m![L % 30 = 6, H / 64 % 60, Dummy2], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 30 = 6, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![L % 30 = 6], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30, Dummy2]>(24));
    
}
}


    let partials: DmTensor<f32, Chip, UpCluster, UpGateRows, m![L % 60, Dummy2]> =
        partials.to_dm(&mut ctx.tdma);
    ctx.main.begin(partials.view())
        .fetch::<m![L % 60, Dummy2], m![1 # 8]>()
        .collect::<m![L % 60, Dummy2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Dummy2, m![L % 60], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 / 4], m![L % 60 % 4 # 16]>()
        .commit_trim::<m![L % 60 % 4]>()
        .commit()
}



fn load_down_input(
    ctx: &mut Context,
    parts: &HbmTensor<f8e4m3, Chip, m![L / 960, Dummy2, L % 960]>,
) -> TrfTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![Dummy2], m![L % 7680]> {
    // Replicate into small independent sub-rings, then share 480-value
    // fragments. Every received K value is valid; contraction has no K pad.
    let shards: DmTensor<f8e4m3, Chip, DownCluster, m![Dummy8 % 4, L / 7680, L / 480 % 16, 1 # 2], m![Dummy2, L % 480]> =
        parts.to_dm(&mut ctx.tdma);
    let shared: DmTensor<f8e4m3, Chip, DownCluster, m![Dummy8 % 4, Dummy256 % 32, L / 7680], m![Dummy2, L % 7680]> = ctx.main
        .begin(shards.view())
        .fetch::<m![Dummy2], m![L % 480]>()
        .switch::<m![Dummy8 % 4, Dummy256 % 32, L / 7680], m![Dummy2, L / 480 % 16]>(
            SwitchConfig::CustomBroadcast { ring_size: 64 })
        .collect::<m![Dummy2, L / 32 % 240], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();
    // Safety: only the fully populated, 256-entry Slice dimension is
    // relabeled in the same physical order. Chip, Cluster, Element and
    // K-shard order are unchanged; all row replicas hold identical inputs.
    let shared = unsafe {
        shared.reshape::<Chip, DownCluster, DownRowsByColumns, m![Dummy2, L % 7680]>()
    };
    ctx.sub.begin(shared.view())
        .fetch::<m![Dummy2], m![L % 7680]>()
        .collect::<m![Dummy2, L / 32 % 240], m![L % 32]>()
        .to_trf()
}

fn project_down_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![Dummy2], m![L % 7680]>,
    x_scales: &DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![L / 960 % 8, Dummy2]>,
    weight: &DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 15, L % 7680]>,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 15, L / 16 % 480]>,
) -> DmTensor<bf16, Chip, DownCluster, DownRows, m![H % 60]> {
    let input_scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![L / 960 % 8, Dummy2]> = ctx.sub
        .begin(x_scales.view())
        .fetch::<m![L / 3840 % 2], m![L / 960 % 4, Dummy2]>()
        .collect::<m![L / 3840 % 2], m![L / 960 % 4, Dummy2]>()
        .to_vrf();
    let mut partials: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 15, Dummy2]> = DmTensor::new();
    {
        // Decode only the row block consumed by the following contractions.
        let native_tile: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 15 = 15, L % 7680]> = ctx.main
            .begin(weight.view().tile::<m![H % 15], 15, m![H % 15 = 15 # 15, L % 7680]>(0))
            .fetch::<m![H % 15 = 15, L / 64 % 120], m![L % 64]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H % 15 = 15, L / 32 % 240], m![L % 32]>()
            .commit_trim::<m![L % 32]>()
            .commit();
{
let block_scale: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 15 = 3, L / 16 % 480]> = ctx.sub
            .begin(scale.view().tile::<m![H % 15], 3, m![H % 15 = 3 # 15, L / 16 % 480]>(0))
            .fetch::<m![H % 15 = 3], m![L / 16 % 480]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 15 = 3, L / 128 % 60], m![L / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![H % 15 = 15], 3, m![H % 15 = 3 # 15, L % 7680]>(0))
            .fetch::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .collect::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .contract_outer::<m![H % 15 = 3, L / 64 % 120], m![L % 64], _, _, _>(x_trf)
            .contract_packet::<m![L / 16 % 4]>()
            .contract_time::<m![H % 15 = 3, L / 64 % 120]>()
            .contract_lane::<m![H % 15 = 3, L / 64 % 120, Dummy2], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![L / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 15 = 3, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![H % 15 = 3], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![H % 15], 3, m![H % 15 = 3 #{!} 15, Dummy2]>(0));
    
}
{
let block_scale: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 15 = 3, L / 16 % 480]> = ctx.sub
            .begin(scale.view().tile::<m![H % 15], 3, m![H % 15 = 3 # 15, L / 16 % 480]>(3))
            .fetch::<m![H % 15 = 3], m![L / 16 % 480]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 15 = 3, L / 128 % 60], m![L / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![H % 15 = 15], 3, m![H % 15 = 3 # 15, L % 7680]>(3))
            .fetch::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .collect::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .contract_outer::<m![H % 15 = 3, L / 64 % 120], m![L % 64], _, _, _>(x_trf)
            .contract_packet::<m![L / 16 % 4]>()
            .contract_time::<m![H % 15 = 3, L / 64 % 120]>()
            .contract_lane::<m![H % 15 = 3, L / 64 % 120, Dummy2], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![L / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 15 = 3, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![H % 15 = 3], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![H % 15], 3, m![H % 15 = 3 #{!} 15, Dummy2]>(3));
    
}
{
let block_scale: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 15 = 3, L / 16 % 480]> = ctx.sub
            .begin(scale.view().tile::<m![H % 15], 3, m![H % 15 = 3 # 15, L / 16 % 480]>(6))
            .fetch::<m![H % 15 = 3], m![L / 16 % 480]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 15 = 3, L / 128 % 60], m![L / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![H % 15 = 15], 3, m![H % 15 = 3 # 15, L % 7680]>(6))
            .fetch::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .collect::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .contract_outer::<m![H % 15 = 3, L / 64 % 120], m![L % 64], _, _, _>(x_trf)
            .contract_packet::<m![L / 16 % 4]>()
            .contract_time::<m![H % 15 = 3, L / 64 % 120]>()
            .contract_lane::<m![H % 15 = 3, L / 64 % 120, Dummy2], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![L / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 15 = 3, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![H % 15 = 3], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![H % 15], 3, m![H % 15 = 3 #{!} 15, Dummy2]>(6));
    
}
{
let block_scale: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 15 = 3, L / 16 % 480]> = ctx.sub
            .begin(scale.view().tile::<m![H % 15], 3, m![H % 15 = 3 # 15, L / 16 % 480]>(9))
            .fetch::<m![H % 15 = 3], m![L / 16 % 480]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 15 = 3, L / 128 % 60], m![L / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![H % 15 = 15], 3, m![H % 15 = 3 # 15, L % 7680]>(9))
            .fetch::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .collect::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .contract_outer::<m![H % 15 = 3, L / 64 % 120], m![L % 64], _, _, _>(x_trf)
            .contract_packet::<m![L / 16 % 4]>()
            .contract_time::<m![H % 15 = 3, L / 64 % 120]>()
            .contract_lane::<m![H % 15 = 3, L / 64 % 120, Dummy2], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![L / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 15 = 3, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![H % 15 = 3], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![H % 15], 3, m![H % 15 = 3 #{!} 15, Dummy2]>(9));
    
}
{
let block_scale: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 15 = 3, L / 16 % 480]> = ctx.sub
            .begin(scale.view().tile::<m![H % 15], 3, m![H % 15 = 3 # 15, L / 16 % 480]>(12))
            .fetch::<m![H % 15 = 3], m![L / 16 % 480]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 15 = 3, L / 128 % 60], m![L / 16 % 8]>()
            .to_vrf();
        ctx.main
            .begin(native_tile.view().tile::<m![H % 15 = 15], 3, m![H % 15 = 3 # 15, L % 7680]>(12))
            .fetch::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .collect::<m![H % 15 = 3, L / 32 % 240], m![L % 32]>()
            .contract_outer::<m![H % 15 = 3, L / 64 % 120], m![L % 64], _, _, _>(x_trf)
            .contract_packet::<m![L / 16 % 4]>()
            .contract_time::<m![H % 15 = 3, L / 64 % 120]>()
            .contract_lane::<m![H % 15 = 3, L / 64 % 120, Dummy2], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![L / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 15 = 3, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![H % 15 = 3], m![Dummy2 # 8]>()
            .commit_trim::<m![Dummy2]>()
            .commit_view(partials.view_mut().tile::<m![H % 15], 3, m![H % 15 = 3 #{!} 15, Dummy2]>(12));
    
}
}


    // Keep both compensation parts until after the K reduction and row gather.
    // Fifteen rows therefore still commit in aligned two-FP32-value packets.
    let partials: DmTensor<f32, Chip, DownCluster, DownPartialRows, m![H % 15, Dummy2]> = ctx.main
        .begin(partials.view())
        .fetch::<m![H % 15], m![Dummy2 # 8]>()
        .collect::<m![H % 15], m![Dummy2 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<DownPartialRows, m![H % 15]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![Dummy2]>()
        .commit();

    let partials: DmTensor<f32, Chip, DownCluster, DownRows, m![H % 60, Dummy2]> =
        partials.to_dm(&mut ctx.tdma);
    ctx.main.begin(partials.view())
        .fetch::<m![H % 60, Dummy2], m![1 # 8]>()
        .collect::<m![H % 60, Dummy2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Dummy2, m![H % 60], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 / 4], m![H % 60 % 4 # 16]>()
        .commit_trim::<m![H % 60 % 4]>()
        .commit()
}

fn project_up_and_gate(
    ctx: &mut Context,
    parts: &HbmTensor<f8e4m3, Chip, m![H / 960, Dummy2, H % 960]>,
    scales: &HbmTensor<f32, Chip, m![H / 960, Dummy2]>,
    up_weight: &DmTensor<f4e2m1, Chip, UpCluster, UpGateKRows, m![L % 30, H % 3840]>,
    gate_weight: &DmTensor<f4e2m1, Chip, UpCluster, UpGateKRows, m![L % 30, H % 3840]>,
    up_scale: &DmTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![L % 30, H / 16 % 240]>,
    gate_scale: &DmTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![L % 30, H / 16 % 240]>,
) -> (
    DmTensor<bf16, Chip, UpCluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, UpCluster, UpGateRows, m![L % 60]>,
) {
    let x_trf = load_up_input(ctx, parts);
    let scales: DmTensor<f32, Chip, UpCluster, UpGateKRows, m![H / 960 % 4, Dummy2]> =
        scales.to_dm(&mut ctx.tdma);
    let up = project_up_matrix(ctx, &x_trf, &scales, up_weight, up_scale);
    let gate = project_up_matrix(ctx, &x_trf, &scales, gate_weight, gate_scale);
    (up, gate)
}

fn project_down(
    ctx: &mut Context,
    parts: &HbmTensor<f8e4m3, Chip, m![L / 960, Dummy2, L % 960]>,
    scales: &HbmTensor<f32, Chip, m![L / 960, Dummy2]>,
    weight: &DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 15, L % 7680]>,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 15, L / 16 % 480]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x_trf = load_down_input(ctx, parts);
    let scales: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![L / 960 % 8, Dummy2]> =
        scales.to_dm(&mut ctx.tdma);
    let down = project_down_matrix(ctx, &x_trf, &scales, weight, scale);
    let down: HbmTensor<bf16, Chip, m![H]> = down.to_hbm(&mut ctx.tdma);
    down.to_dm(&mut ctx.tdma)
}


pub(crate) fn feedforward(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let up_packed_dm: DmTensor<f4e2m1, Chip, UpCluster, UpGateKRows, m![L % 30, H % 3840]> =
        up_weight_packed.to_dm(&mut ctx.tdma);
    let gate_packed_dm: DmTensor<f4e2m1, Chip, UpCluster, UpGateKRows, m![L % 30, H % 3840]> =
        gate_weight_packed.to_dm(&mut ctx.tdma);
    let down_packed_dm: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 15, L % 7680]> =
        down_weight_packed.to_dm(&mut ctx.tdma);

    let up_block_scales: DmTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![L % 30, H / 16 % 240]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_block_scales: DmTensor<f8e4m3, Chip, UpCluster, UpGateKRows, m![L % 30, H / 16 % 240]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);
    let down_block_scales: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 15, L / 16 % 480]> =
        down_weight_scale.to_dm(&mut ctx.tdma);

    let (x_parts, x_scales) = prepare_up_input(ctx, x);
    let (up, gate) = project_up_and_gate(
        ctx, &x_parts, &x_scales,
        &up_packed_dm, &gate_packed_dm, &up_block_scales, &gate_block_scales,
    );
    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    let (down_parts, down_scales) = prepare_down_input(ctx, x);
    let down = project_down(ctx, &down_parts, &down_scales, &down_packed_dm, &down_block_scales);



    let down_global_scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(down.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn geglu(
    ctx: &mut Context,
    up: DmTensor<bf16, Chip, UpCluster, UpGateRows, m![L % 60]>,
    gate: DmTensor<bf16, Chip, UpCluster, UpGateRows, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpCluster, UpGateRowsPaired, m![L % 120]> {
    let up: DmTensor<bf16, Chip, UpCluster, UpGateRowsPaired, m![L % 120]> = up.to_dm(&mut ctx.tdma);
    let up_global_scale: DmTensor<f32, Chip, UpCluster, UpGateRowsPaired, m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale_vrf: VrfTensor<f32, Chip, UpCluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let up: DmTensor<bf16, Chip, UpCluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gate_global_scale: DmTensor<f32, Chip, UpCluster, UpGateRowsPaired, m![1 # 8]> =
        gate_global_scale.to_dm(&mut ctx.tdma);
    let gate_global_scale_vrf: VrfTensor<f32, Chip, UpCluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(gate_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let gate: DmTensor<bf16, Chip, UpCluster, UpGateRowsPaired, m![L % 120]> = gate.to_dm(&mut ctx.tdma);
    let gate: DmTensor<bf16, Chip, UpCluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, UpCluster, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpCluster, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}
