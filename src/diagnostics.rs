//! Diagnostic-only kernels; excluded from normal builds.
use furiosa_opt_std::prelude::*;
use crate::Chip;
use crate::device::layout::{Cluster, Slice};

/// A 32-byte HBM -> DM -> HBM copy executed as an NPU task.
#[device(chip = 1)]
pub fn warmup_copy(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![16]>,
    output: &mut HbmTensor<bf16, Chip, m![16]>,
) {
    let value: DmTensor<bf16, Chip, Cluster, Slice, m![16]> = input.to_dm(&mut ctx.tdma);
    value.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
