
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H};

use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

// Keep each projected half in its original cluster. Only the two FP32
// mean-square contributions cross HBM; both BF16 rounding boundaries remain.
pub(crate) fn normalize_residual_distributed(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![H / 1920], m![H / 120 % 16, 1 # 16], m![H % 120]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    type OutputCluster = m![H / 1920];
    type LocalSlices = m![1 # 64, H / 480 % 4];

    let x: DmTensor<bf16, Chip, OutputCluster, LocalSlices, m![H % 480]> =
        x.to_dm(&mut ctx.tdma);
    let mean_square: DmTensor<f32, Chip, OutputCluster, LocalSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let cluster_mean: DmTensor<f32, Chip, OutputCluster, m![1 # 256], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // Relabel the two physical cluster positions without moving data. The
    // independent scalar axis lets every H partition read both contributions.
    let cluster_mean: DmTensor<f32, Chip, m![Dummy8 / 4], m![1 # 256], m![1 # 8]> =
        unsafe { cluster_mean.reshape() };
    let means_hbm: HbmTensor<f32, Chip, m![Dummy8 / 4, 1 # 8]> =
        cluster_mean.to_hbm(&mut ctx.tdma);
    let means: DmTensor<f32, Chip, OutputCluster, LocalSlices, m![Dummy8 / 4, 1 # 8]> =
        means_hbm.to_dm(&mut ctx.tdma);

    let global_mean: DmTensor<f32, Chip, OutputCluster, LocalSlices, m![1 # 8]> = ctx
        .main
        .begin(means.view())
        .fetch::<m![Dummy8 / 4], m![1 # 8]>()
        .collect::<m![Dummy8 / 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Dummy8, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let inv_rms: DmTensor<f32, Chip, OutputCluster, LocalSlices, m![1 # 8]> = ctx
        .main
        .begin(global_mean.view())
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

    let weight: DmTensor<bf16, Chip, OutputCluster, LocalSlices, m![H % 480]> =
        rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, OutputCluster, LocalSlices, m![H % 480]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let inv_rms_vrf: VrfTensor<f32, Chip, OutputCluster, LocalSlices, m![1 # 8]> = ctx
        .sub
        .begin(inv_rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, OutputCluster, LocalSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let residual: DmTensor<bf16, Chip, OutputCluster, LocalSlices, m![H % 480]> =
        residual_hbm.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, OutputCluster, LocalSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, OutputCluster, LocalSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    output
        .view()
        .to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

pub(crate) fn normalize<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let inv_rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
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
    let inv_rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { inv_rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let inv_rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(inv_rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
