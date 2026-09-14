use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use furiosa_opt_std::prelude::*;

use furiosa_opt_gemma4::axes::*;
use furiosa_opt_gemma4::{Chip, ops};

mod prng {
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
    const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

    pub fn name_hash(name: &str) -> u64 {
        let mut hash = FNV_OFFSET;
        for byte in name.as_bytes() {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
        }
        hash
    }

    pub fn word(seed: u64, index: usize) -> u64 {
        let mut x = (seed ^ index as u64).wrapping_add(GOLDEN);
        x = (x ^ (x >> 30)).wrapping_mul(MIX_A);
        x = (x ^ (x >> 27)).wrapping_mul(MIX_B);
        x ^ (x >> 31)
    }

    pub fn u01(w: u64) -> f32 {
        ((w >> 40) as u32) as f32 / 16_777_216.0
    }

    pub fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounding = 0x7FFF + ((bits >> 16) & 1);
        (bits.wrapping_add(rounding) >> 16) as u16
    }

    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn bf16_uniform(name: &str, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        bf16_uniform_range(name, 0, count, lo, hi)
    }

    pub fn bf16_uniform_range(name: &str, offset: usize, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count * 2);
        for index in offset..offset + count {
            push_u16(&mut out, f32_to_bf16_bits(lo + (hi - lo) * u01(word(seed, index))));
        }
        out
    }

    pub fn bf16_signs(name: &str, count: usize, scale: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let magnitude = f32_to_bf16_bits(scale.abs());
        let mut out = Vec::with_capacity(count * 2);
        for index in 0..count {
            let negative = word(seed, index) >> 63 == 1;
            push_u16(&mut out, if negative { magnitude | 0x8000 } else { magnitude });
        }
        out
    }

    pub fn f8_banded(name: &str, count: usize, exp_min: u8, exp_max: u8, signed: bool) -> Vec<u8> {
        assert!(
            exp_min <= exp_max && exp_max <= 14,
            "{name}: exponent band out of range"
        );
        let seed = name_hash(name);
        let span = u64::from(exp_max - exp_min + 1);
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let w = word(seed, index);
            let exponent = exp_min + ((w >> 32) % span) as u8;
            let mantissa = ((w >> 56) & 0x7) as u8;
            let sign = if signed { ((w >> 63) as u8) << 7 } else { 0 };
            out.push(sign | (exponent << 3) | mantissa);
        }
        out
    }

    pub fn f4_nibbles(name: &str, count: usize) -> Vec<u8> {
        assert!(count % 2 == 0, "{name}: f4 element count must be even");
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count / 2);
        for pair in 0..count / 2 {
            let low = ((word(seed, pair * 2) >> 60) & 0xF) as u8;
            let high = ((word(seed, pair * 2 + 1) >> 60) & 0xF) as u8;
            out.push(low | (high << 4));
        }
        out
    }

    pub fn checksum(bytes: &[u8], word_offset: usize) -> u32 {
        let mut total: u32 = 0;
        for (block, chunk) in bytes.chunks(4).enumerate() {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            let index = (word_offset + block) as u32;
            total = total.wrapping_add(u32::from_le_bytes(word).wrapping_mul(index.wrapping_mul(2).wrapping_add(1)));
        }
        total
    }
}

const WEIGHT_EXP: (u8, u8) = (7, 14);
const LOCAL_SCALE_EXP: (u8, u8) = (8, 10);
const ROW_SCALE: (f32, f32) = (0.0, 1e-3);
const UNIT: (f32, f32) = (0.0, 1.0);
const RMS_WEIGHT: (f32, f32) = (0.75, 1.25);

const POS: usize = 137;
const LAYER_SCALAR: f32 = 0.375;

const RAW_GLOBAL_SCALES: [f32; 3] = [9600.0, 9600.0, 12928.0];

struct Fixture {
    expected: HashMap<String, Vec<f32>>,
    checksums: HashMap<String, u32>,
}

const FIXTURE_PATHS: [&str; 2] = ["ref/fixtures.safetensors", "fixtures.safetensors"];

fn fixture_path() -> String {
    if let Ok(path) = std::env::var("GEMMA4_FIXTURE") {
        return path;
    }
    FIXTURE_PATHS
        .iter()
        .find(|path| std::path::Path::new(path).exists())
        .unwrap_or(&FIXTURE_PATHS[0])
        .to_string()
}

impl Fixture {
    fn load(path: &str) -> Self {
        let file = File::open(path)
            .unwrap_or_else(|e| panic!("{path}: {e} -- run `python3 scripts/generate_references.py` first"));
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap_or_else(|e| panic!("{path}: {e}"));
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .unwrap_or_else(|e| panic!("{path}: not a safetensors file: {e}"));

        let mut expected = HashMap::new();
        let mut checksums = HashMap::new();
        for (name, view) in tensors.tensors() {
            match view.dtype() {
                safetensors::Dtype::F32 => {
                    let values = view
                        .data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    expected.insert(name, values);
                }
                safetensors::Dtype::I64 => {
                    let raw = i64::from_le_bytes(view.data()[..8].try_into().unwrap());
                    checksums.insert(name, raw as u32);
                }
                other => panic!("{name}: unexpected fixture dtype {other:?}"),
            }
        }
        Self { expected, checksums }
    }

    fn expect(&self, test: &str, label: &str) -> &[f32] {
        let key = format!("{test}.{label}");
        self.expected
            .get(&key)
            .unwrap_or_else(|| panic!("fixture has no `{key}` -- regenerate with scripts/generate_references.py"))
    }

    fn assert_every_expectation_is_tested(&self) {
        let orphans: Vec<&str> = self
            .expected
            .keys()
            .map(String::as_str)
            .filter(|key| {
                let test = key.split('.').next().unwrap_or(key);
                !TESTS.iter().any(|candidate| candidate.name == test)
            })
            .collect();
        assert!(
            orphans.is_empty(),
            "the fixture has expectations no test reads, so they are silently unchecked: {orphans:?}\n\
             add the matching `Test` row, `run_test` arm and shim, or drop the generator"
        );
    }

    fn checksum(&self, test: &str, input: &str) -> u32 {
        let key = format!("{test}.check.{input}");
        *self.checksums.get(&key).unwrap_or_else(|| {
            panic!("fixture has no checksum `{key}` -- regenerate with scripts/generate_references.py")
        })
    }
}

struct Synth<'a> {
    test: &'static str,
    fixture: &'a Fixture,
}

impl<'a> Synth<'a> {
    fn new(test: &'static str, fixture: &'a Fixture) -> Self {
        Self { test, fixture }
    }

    fn seed(&self, name: &str) -> String {
        format!("{}.{}", self.test, name)
    }

    fn verify(&self, name: &str, storage: &[u8]) {
        let expected = self.fixture.checksum(self.test, name);
        let actual = prng::checksum(storage, 0);
        assert_eq!(
            actual, expected,
            "{}.{name}: synthesized input does not match scripts/generate_references.py \
             (checksum {actual:#010x} vs {expected:#010x}) -- fixture_prng.py and \
             this file's prng module have diverged",
            self.test
        );
    }

    async fn upload<D: MaterializableScalar, E: M>(
        &self,
        ctx: &mut Context,
        name: &str,
        storage: Vec<u8>,
    ) -> HbmTensor<D, Chip, E> {
        self.verify(name, &storage);
        HostTensor::<D, E>::from_buf(storage).to_hbm(&mut ctx.pdma).await
    }

    async fn bf16<E: M>(&self, ctx: &mut Context, name: &str, span: (f32, f32)) -> HbmTensor<bf16, Chip, E> {
        let storage = prng::bf16_uniform(&self.seed(name), E::SIZE, span.0, span.1);
        self.upload(ctx, name, storage).await
    }

    async fn signs<E: M>(&self, ctx: &mut Context, name: &str, scale: f32) -> HbmTensor<bf16, Chip, E> {
        let storage = prng::bf16_signs(&self.seed(name), E::SIZE, scale);
        self.upload(ctx, name, storage).await
    }

    async fn f8<E: M>(
        &self,
        ctx: &mut Context,
        name: &str,
        band: (u8, u8),
        signed: bool,
    ) -> HbmTensor<f8e4m3, Chip, E> {
        let storage = prng::f8_banded(&self.seed(name), E::SIZE, band.0, band.1, signed);
        self.upload(ctx, name, storage).await
    }

    async fn f4<E: M>(&self, ctx: &mut Context, name: &str) -> HbmTensor<f4e2m1, Chip, E> {
        let storage = prng::f4_nibbles(&self.seed(name), E::SIZE);
        self.upload(ctx, name, storage).await
    }

    async fn constant_f32<E: M>(&self, ctx: &mut Context, name: &str, values: &[f32]) -> HbmTensor<f32, Chip, E> {
        let storage: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(ctx, name, storage).await
    }

    async fn constant_bf16<E: M>(&self, ctx: &mut Context, name: &str, values: &[f32]) -> HbmTensor<bf16, Chip, E> {
        let storage: Vec<u8> = values
            .iter()
            .flat_map(|v| prng::f32_to_bf16_bits(*v).to_le_bytes())
            .collect();
        self.upload(ctx, name, storage).await
    }

    async fn constant_i32<E: M>(&self, ctx: &mut Context, name: &str, value: i32) -> HbmTensor<i32, Chip, E> {
        self.upload(ctx, name, value.to_le_bytes().to_vec()).await
    }
}

async fn exact_rmsnorm_input(ctx: &mut Context, s: &Synth<'_>) -> HbmTensor<bf16, Chip, m![H]> {
    let signs = prng::bf16_signs(&s.seed("x_signs"), H::SIZE, 1.0);
    s.verify("x_signs", &signs);
    let weights = prng::bf16_uniform(&s.seed("input_rms_weight"), H::SIZE, RMS_WEIGHT.0, RMS_WEIGHT.1);

    let storage: Vec<u8> = signs
        .chunks_exact(2)
        .zip(weights.chunks_exact(2))
        .flat_map(|(sign, weight)| {
            let sign = prng::bf16_bits_to_f32(u16::from_le_bytes(sign.try_into().unwrap()));
            let weight = prng::bf16_bits_to_f32(u16::from_le_bytes(weight.try_into().unwrap()));
            prng::f32_to_bf16_bits(sign / weight).to_le_bytes()
        })
        .collect();
    s.verify("x_exact", &storage);
    HostTensor::<bf16, m![H]>::from_buf(storage).to_hbm(&mut ctx.pdma).await
}

async fn zeros<D: ScalarBytes + MaterializableScalar, E: M>(ctx: &mut Context) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(vec![0u8; E::SIZE * D::BITS / 8])
        .to_hbm(&mut ctx.pdma)
        .await
}

async fn read_bf16<E: M>(ctx: &mut Context, tensor: &HbmTensor<bf16, Chip, E>) -> Vec<f32> {
    let host: HostTensor<bf16, E> = tensor.to_host(&mut ctx.pdma).await;
    host.into_vec().into_iter().map(bf16::to_f32).collect()
}

fn rope_tables(head_dim: usize, theta: f64, partial_rotary_factor: f64, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let angles = (partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize;
    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f64; half];
    for (i, slot) in inv_freq.iter_mut().enumerate().take(angles) {
        *slot = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
    }

    let mut cos = vec![0.0f32; head_dim];
    let mut sin = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        let angle = pos as f64 * inv_freq[i % half];
        cos[i] = angle.cos() as f32;
        sin[i] = angle.sin() as f32;
    }
    (cos, sin)
}

fn negate_low_half(sin: &[f32]) -> Vec<f32> {
    let half = sin.len() / 2;
    sin.iter()
        .enumerate()
        .map(|(i, v)| if i < half { -v } else { *v })
        .collect()
}

async fn rope_table<D: AxisName>(
    ctx: &mut Context,
    s: &Synth<'_>,
    name: &str,
    values: &[f32],
    pos: usize,
) -> HbmTensor<bf16, Chip, m![E, D]> {
    let row: Vec<u8> = values
        .iter()
        .flat_map(|v| prng::f32_to_bf16_bits(*v).to_le_bytes())
        .collect();
    s.verify(name, &row);

    let mut storage = vec![0u8; E::SIZE * D::SIZE * 2];
    let byte_offset = pos * D::SIZE * 2;
    storage[byte_offset..byte_offset + row.len()].copy_from_slice(&row);
    HostTensor::<bf16, m![E, D]>::from_buf(storage)
        .to_hbm(&mut ctx.pdma)
        .await
}

struct Test {
    name: &'static str,
    atol: f32,
    rtol: f32,
}

const RTOL: f32 = 1e-2;

const TESTS: &[Test] = &[
    Test {
        name: "sliding_project_qkv",
        atol: 0.04,
        rtol: RTOL,
    },
    Test {
        name: "sliding_attention_output",
        atol: 0.05,
        rtol: RTOL,
    },
    Test {
        name: "decoder_feedforward",
        atol: 0.01,
        rtol: RTOL,
    },
];

enum Prepared {
    Qkv(Qkv),
    Attention(Attention),
    Feedforward(Feedforward),
}

impl Prepared {
    async fn launch(&mut self, ctx: &mut Context) -> u128 {
        match self {
            Self::Qkv(input) => input.launch(ctx).await,
            Self::Attention(input) => input.launch(ctx).await,
            Self::Feedforward(input) => input.launch(ctx).await,
        }
    }
    async fn read_outputs(&self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        match self {
            Self::Qkv(input) => input.read_outputs(ctx).await,
            Self::Attention(input) => input.read_outputs(ctx).await,
            Self::Feedforward(input) => input.read_outputs(ctx).await,
        }
    }
    async fn read_input(&self, ctx: &mut Context) {
        // Read-only PDMA traffic: no kernel execution and no mutation of inputs.
        match self {
            Self::Qkv(input) => {
                let _ = read_bf16(ctx, &input.x).await;
            }
            Self::Attention(input) => {
                let _ = read_bf16(ctx, &input.x).await;
            }
            Self::Feedforward(input) => {
                let _ = read_bf16(ctx, &input.residual).await;
            }
        }
    }
    async fn new(ctx: &mut Context, fixture: &Fixture, name: &str) -> Self {
        match name {
            "sliding_project_qkv" => Self::Qkv(Qkv::prepare(ctx, fixture).await),
            "sliding_attention_output" => Self::Attention(Attention::prepare(ctx, fixture).await),
            "decoder_feedforward" => Self::Feedforward(Feedforward::prepare(ctx, fixture).await),
            _ => panic!("unknown kernel"),
        }
    }
    async fn execute(&mut self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        match self {
            Self::Qkv(input) => input.execute(ctx).await,
            Self::Attention(input) => input.execute(ctx).await,
            Self::Feedforward(input) => input.execute(ctx).await,
        }
    }
}

struct Qkv {
    input_rms_weight: HbmTensor<bf16, Chip, m![H]>,
    x: HbmTensor<bf16, Chip, m![H]>,
    q_weight: HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: HbmTensor<bf16, Chip, m![Ps]>,
    q_rms_weight: HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: HbmTensor<bf16, Chip, m![Ds]>,
    cos: HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: HbmTensor<bf16, Chip, m![E, Ds]>,
    rope_offset: HbmTensor<i32, Chip, m![1]>,
    kv_offset: HbmTensor<i32, Chip, m![1]>,
    k_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    q_out: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    slot: usize,
}

impl Qkv {
    async fn prepare(ctx: &mut Context, fixture: &Fixture) -> Self {
        let s = Synth::new("sliding_project_qkv", fixture);

        let input_rms_weight: HbmTensor<bf16, Chip, m![H]> =
            s.bf16(ctx, "input_rms_weight", RMS_WEIGHT).await;
        let x: HbmTensor<bf16, Chip, m![H]> = exact_rmsnorm_input(ctx, &s).await;

        let q_weight: HbmTensor<f8e4m3, Chip, m![Qs, H]> =
            s.f8(ctx, "q_weight", WEIGHT_EXP, true).await;
        let k_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> =
            s.f8(ctx, "k_weight", WEIGHT_EXP, true).await;
        let v_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> =
            s.f8(ctx, "v_weight", WEIGHT_EXP, true).await;
        let q_weight_scale: HbmTensor<bf16, Chip, m![Qs]> =
            s.bf16(ctx, "q_weight_scale", ROW_SCALE).await;
        let k_weight_scale: HbmTensor<bf16, Chip, m![Ps]> =
            s.bf16(ctx, "k_weight_scale", ROW_SCALE).await;
        let v_weight_scale: HbmTensor<bf16, Chip, m![Ps]> =
            s.bf16(ctx, "v_weight_scale", ROW_SCALE).await;
        let q_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(ctx, "q_rms_weight", UNIT).await;
        let k_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(ctx, "k_rms_weight", UNIT).await;

        let (cos_values, sin_values) = rope_tables(Ds::SIZE, 10_000.0, 1.0, POS);
        let cos: HbmTensor<bf16, Chip, m![E, Ds]> =
            rope_table::<Ds>(ctx, &s, "cos", &cos_values, POS).await;
        let sin: HbmTensor<bf16, Chip, m![E, Ds]> =
            rope_table::<Ds>(ctx, &s, "sin", &negate_low_half(&sin_values), POS).await;
        let rope_offset: HbmTensor<i32, Chip, m![1]> = s
            .constant_i32(ctx, "rope_offset", (POS * Ds::SIZE * 2) as i32)
            .await;

        let slot = POS % Ts::SIZE;
        let offset = (slot * Ns::SIZE * Ds::SIZE * 2) as i32;
        let kv_offset: HbmTensor<i32, Chip, m![1]> = s.constant_i32(ctx, "kv_offset", offset).await;

        let k_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(ctx).await;
        let v_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(ctx).await;
        let q_out: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = zeros(ctx).await;

        Self {
            input_rms_weight,
            x,
            q_weight,
            k_weight,
            v_weight,
            q_weight_scale,
            k_weight_scale,
            v_weight_scale,
            q_rms_weight,
            k_rms_weight,
            cos,
            sin,
            rope_offset,
            kv_offset,
            k_cache,
            v_cache,
            q_out,
            slot,
        }
    }

    async fn execute(&mut self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        println!(
            "DIAG_LAUNCH pid={} host_us={}",
            std::process::id(),
            host_us()
        );
        let elapsed = self.launch(ctx).await;
        println!("DIAG_HOST_LAUNCH elapsed_ns={elapsed}");
        self.read_outputs(ctx).await
    }
    async fn launch(&mut self, ctx: &mut Context) -> u128 {
        let launch_started = Instant::now();
        launch(
            ops::sliding_project_qkv,
            (
                ctx,
                &self.x,
                &self.q_weight,
                &self.k_weight,
                &self.v_weight,
                &self.q_weight_scale,
                &self.k_weight_scale,
                &self.v_weight_scale,
                &self.input_rms_weight,
                &self.q_rms_weight,
                &self.k_rms_weight,
                &self.kv_offset,
                &self.rope_offset,
                &self.cos,
                &self.sin,
                &mut self.k_cache,
                &mut self.v_cache,
                &mut self.q_out,
            ),
        )
        .await;
        launch_started.elapsed().as_nanos()
    }
    async fn read_outputs(&self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        let width = Ns::SIZE * Ds::SIZE;
        let k = read_bf16(ctx, &self.k_cache).await[self.slot * width..(self.slot + 1) * width]
            .to_vec();
        let v = read_bf16(ctx, &self.v_cache).await[self.slot * width..(self.slot + 1) * width]
            .to_vec();
        vec![
            ("expected.q", read_bf16(ctx, &self.q_out).await),
            ("expected.k", k),
            ("expected.v", v),
        ]
    }
}

struct Attention {
    x: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: HbmTensor<bf16, Chip, m![H]>,
    o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: HbmTensor<bf16, Chip, m![H]>,
    residual: HbmTensor<bf16, Chip, m![H]>,
}

impl Attention {
    async fn prepare(ctx: &mut Context, fixture: &Fixture) -> Self {
        let s = Synth::new("sliding_attention_output", fixture);
        let x: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = s.signs(ctx, "x", 1.0).await;
        let post_attn_rms_weight: HbmTensor<bf16, Chip, m![H]> =
            s.bf16(ctx, "post_attn_rms_weight", UNIT).await;
        let o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]> =
            s.f8(ctx, "o_weight", WEIGHT_EXP, true).await;
        let o_weight_scale: HbmTensor<bf16, Chip, m![H]> =
            s.bf16(ctx, "o_weight_scale", ROW_SCALE).await;
        let residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "residual", UNIT).await;

        Self {
            x,
            post_attn_rms_weight,
            o_weight,
            o_weight_scale,
            residual,
        }
    }

    async fn execute(&mut self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        println!(
            "DIAG_LAUNCH pid={} host_us={}",
            std::process::id(),
            host_us()
        );
        let elapsed = self.launch(ctx).await;
        println!("DIAG_HOST_LAUNCH elapsed_ns={elapsed}");
        self.read_outputs(ctx).await
    }
    async fn launch(&mut self, ctx: &mut Context) -> u128 {
        let launch_started = Instant::now();
        launch(
            ops::sliding_attention_output,
            (
                ctx,
                &self.x,
                &self.post_attn_rms_weight,
                &self.o_weight,
                &self.o_weight_scale,
                &mut self.residual,
            ),
        )
        .await;
        launch_started.elapsed().as_nanos()
    }
    async fn read_outputs(&self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        vec![("expected", read_bf16(ctx, &self.residual).await)]
    }
}

struct Feedforward {
    residual: HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: HbmTensor<bf16, Chip, m![H]>,
    post_ff_rms_weight: HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: HbmTensor<f32, Chip, m![1]>,
    down_global_scale: HbmTensor<f32, Chip, m![1]>,
    layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]>,
}

impl Feedforward {
    async fn prepare(ctx: &mut Context, fixture: &Fixture) -> Self {
        let s = Synth::new("decoder_feedforward", fixture);

        let residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "residual", UNIT).await;
        let pre_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> =
            s.bf16(ctx, "pre_ff_rms_weight", UNIT).await;
        let post_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> =
            s.bf16(ctx, "post_ff_rms_weight", UNIT).await;

        let up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> =
            s.f4(ctx, "up_weight_packed").await;
        let gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> =
            s.f4(ctx, "gate_weight_packed").await;
        let down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]> =
            s.f4(ctx, "down_weight_packed").await;
        let up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
            s.f8(ctx, "up_weight_scale", LOCAL_SCALE_EXP, false).await;
        let gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
            s.f8(ctx, "gate_weight_scale", LOCAL_SCALE_EXP, false).await;
        let down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]> =
            s.f8(ctx, "down_weight_scale", LOCAL_SCALE_EXP, false).await;

        let up_global_scale: HbmTensor<f32, Chip, m![1]> = s
            .constant_f32(ctx, "up_global_scale", &[1.0 / RAW_GLOBAL_SCALES[0]])
            .await;
        let gate_global_scale: HbmTensor<f32, Chip, m![1]> = s
            .constant_f32(ctx, "gate_global_scale", &[1.0 / RAW_GLOBAL_SCALES[1]])
            .await;
        let down_global_scale: HbmTensor<f32, Chip, m![1]> = s
            .constant_f32(ctx, "down_global_scale", &[1.0 / RAW_GLOBAL_SCALES[2]])
            .await;

        let layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]> = s
            .constant_bf16(ctx, "layer_scalar", &[LAYER_SCALAR; 8])
            .await;

        Self {
            residual,
            pre_ff_rms_weight,
            post_ff_rms_weight,
            up_weight_packed,
            gate_weight_packed,
            down_weight_packed,
            up_weight_scale,
            gate_weight_scale,
            down_weight_scale,
            up_global_scale,
            gate_global_scale,
            down_global_scale,
            layer_scalar,
        }
    }

    async fn execute(&mut self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        println!(
            "DIAG_LAUNCH pid={} host_us={}",
            std::process::id(),
            host_us()
        );
        let elapsed = self.launch(ctx).await;
        println!("DIAG_HOST_LAUNCH elapsed_ns={elapsed}");
        self.read_outputs(ctx).await
    }
    async fn launch(&mut self, ctx: &mut Context) -> u128 {
        let launch_started = Instant::now();
        launch(
            ops::decoder_feedforward,
            (
                ctx,
                &mut self.residual,
                &self.pre_ff_rms_weight,
                &self.up_weight_packed,
                &self.gate_weight_packed,
                &self.down_weight_packed,
                &self.up_weight_scale,
                &self.gate_weight_scale,
                &self.down_weight_scale,
                &self.up_global_scale,
                &self.gate_global_scale,
                &self.down_global_scale,
                &self.post_ff_rms_weight,
                &self.layer_scalar,
            ),
        )
        .await;
        launch_started.elapsed().as_nanos()
    }
    async fn read_outputs(&self, ctx: &mut Context) -> Vec<(&'static str, Vec<f32>)> {
        vec![("expected", read_bf16(ctx, &self.residual).await)]
    }
}

fn compare(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> bool {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: shape mismatch ({} expected vs {} from device)",
        expected.len(),
        actual.len()
    );

    let mut max_diff = 0.0f32;
    let mut max_index = 0usize;
    let mut sum_diff = 0.0f64;
    let mut within = 0usize;

    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if !got.is_finite() {
            println!("[{label:34}] FAIL -- non-finite device output at {index}");
            return false;
        }
        let diff = (want - got).abs();
        if diff <= atol + rtol * want.abs() {
            within += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
        sum_diff += f64::from(diff);
    }

    let count = expected.len();
    let ok = within == count;
    let relative = if expected[max_index].abs() > 1e-12 {
        max_diff / expected[max_index].abs() * 100.0
    } else {
        0.0
    };
    println!(
        "[{label:34}] max|Δ|={max_diff:9.5} ({relative:7.2}% of expected)  mean|Δ|={:9.6}  \
         within tol={:6.2}%  -> {}",
        sum_diff / count as f64,
        within as f32 / count as f32 * 100.0,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

// --- on-device cycle collection (only armed when TUC_PROFILE_LEVEL is set) ---

const TRACING_TARGET_NPU: &str = "span::npu";

#[derive(Clone, Copy)]
struct Span {
    begin: u64,
    end: u64,
}

/// A minimal `tracing::Subscriber`: all we need is to see each `span::npu` span's fields
/// as it's created, not the full `Layer`/`Registry` machinery `tracing-subscriber` offers.
#[derive(Clone, Default)]
struct Collector {
    spans: Arc<Mutex<Vec<Span>>>,
    next_id: Arc<AtomicU64>,
}

impl Collector {
    async fn take_task(&self) -> Span {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                {
                    let mut spans = self.spans.lock().unwrap();
                    assert!(spans.len() <= 1, "multiple Task spans for one launch");
                    if let Some(span) = spans.pop() {
                        return span;
                    }
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("Task trace missing after 5 seconds")
    }
}

#[derive(Default)]
struct FieldExtractor {
    name: String,
    begin: Option<u64>,
    end: Option<u64>,
}

impl tracing::field::Visit for FieldExtractor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "begin_cycle" => self.begin = Some(value),
            "end_cycle" => self.end = Some(value),
            _ => {}
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "name" {
            self.name = value.to_string();
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "name" {
            self.name = format!("{value:?}");
        }
    }
}

impl tracing::Subscriber for Collector {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == TRACING_TARGET_NPU
    }

    fn new_span(&self, attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        if attrs.metadata().target() == TRACING_TARGET_NPU {
            let mut extractor = FieldExtractor::default();
            attrs.record(&mut extractor);
            if let (Some(begin), Some(end)) = (extractor.begin, extractor.end) {
                assert_eq!(extractor.name, "Task", "unexpected profile span");
                self.spans.lock().unwrap().push(Span { begin, end });
            }
        }
        // 0 is reserved by `span::Id`; spans aren't tracked individually here, so the
        // id only needs to be unique and non-zero.
        tracing::span::Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, _event: &tracing::Event<'_>) {}
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

fn profiling_enabled() -> bool {
    let level = std::env::var("TUC_PROFILE_LEVEL")
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(level.as_str(), "info" | "debug" | "trace")
}

fn host_us() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros()
}

async fn measure(
    ctx: &mut Context,
    fixture: &Fixture,
    test: &Test,
    collector: &Collector,
    prepared: &mut Prepared,
    label: &str,
    burst: usize,
) {
    assert!(
        collector.spans.lock().unwrap().is_empty(),
        "unconsumed Task trace"
    );
    println!("==> {}", test.name);
    println!(
        "DIAG_CASE kernel={} {} burst={} pid={} host_us={}",
        test.name,
        label,
        burst,
        std::process::id(),
        host_us()
    );
    let started = Instant::now();
    let outputs = prepared.execute(ctx).await;
    let returned_us = started.elapsed().as_micros();
    let cycles = if std::env::var("DIAG_MODE").unwrap() == "unprofiled" {
        println!(
            "DIAG_RESULT kernel={} {} burst={} pid={} begin=none end=none cycles=none shim_us={} trace_wait_us=0",
            test.name,
            label,
            burst,
            std::process::id(),
            returned_us
        );
        None
    } else {
        let span = collector.take_task().await;
        let trace_wait_us = started.elapsed().as_micros() - returned_us;
        assert!(span.end >= span.begin, "invalid Task timestamps");
        println!(
            "DIAG_RESULT kernel={} {} burst={} pid={} begin={} end={} cycles={} shim_us={} trace_wait_us={}",
            test.name,
            label,
            burst,
            std::process::id(),
            span.begin,
            span.end,
            span.end - span.begin,
            returned_us,
            trace_wait_us
        );
        Some(span.end - span.begin)
    };
    assert!(!outputs.is_empty(), "shim produced no outputs");
    for (label, actual) in &outputs {
        assert!(
            compare(
                label,
                fixture.expect(test.name, label),
                actual,
                test.atol,
                test.rtol
            ),
            "accuracy failure in {}",
            test.name
        );
    }
    if let Some(cycles) = cycles {
        println!("    cycles={cycles}");
    }
}

#[tokio::main]
async fn main() {
    assert!(profiling_enabled(), "TUC_PROFILE_LEVEL=info is required");
    let kernel = std::env::var("DIAG_KERNEL").expect("DIAG_KERNEL is required");
    let mode = std::env::var("DIAG_MODE").expect("DIAG_MODE required");
    assert!(matches!(
        mode.as_str(),
        "cross" | "transfer" | "unprofiled" | "cost"
    ));
    let fixture = Fixture::load(&fixture_path());
    fixture.assert_every_expectation_is_tested();
    let test = TESTS
        .iter()
        .find(|test| test.name == kernel)
        .expect("unknown kernel");
    if mode == "cost" {
        assert!(!tracing::enabled!(target: "span::npu", tracing::Level::INFO));
        benchmark_cost(&fixture, test).await;
        return;
    }
    let collector = Collector::default();
    if mode != "unprofiled" {
        tracing::subscriber::set_global_default(collector.clone()).expect("set tracing subscriber");
    }
    // SDK ffi::run chooses furiosa_kernel_run when this target is disabled.
    assert_eq!(
        tracing::enabled!(target: "span::npu", tracing::Level::INFO),
        mode != "unprofiled"
    );
    let mut ctx = Context::acquire();
    // Exercise all kernel entry points before the experiment, so first-use
    // loading of a different kernel cannot masquerade as a wake-up effect.
    for bootstrap_test in TESTS {
        let mut input = Prepared::new(&mut ctx, &fixture, bootstrap_test.name).await;
        measure(
            &mut ctx,
            &fixture,
            bootstrap_test,
            &collector,
            &mut input,
            &format!("mode={mode} condition=bootstrap repetition=0 phase=bootstrap"),
            0,
        )
        .await;
    }
    let cross_conditions = [
        "none",
        "sliding_project_qkv",
        "sliding_attention_output",
        "decoder_feedforward",
    ];
    let conditions: &[&str] = if mode == "unprofiled" {
        &["none", "sliding_attention_output"]
    } else if mode == "transfer" {
        &["none", "transfer", "sliding_attention_output"]
    } else {
        &cross_conditions
    };
    for repetition in 0..3 {
        for offset in 0..conditions.len() {
            let condition = conditions[(offset + repetition) % conditions.len()];
            // All allocations/transfers finish BEFORE the seed launch and idle gap.
            // Every launch owns fresh input/output buffers, including residual and KV.
            let mut seed = Prepared::new(&mut ctx, &fixture, test.name).await;
            let mut targets = Vec::new();
            for _ in 0..3 {
                targets.push(Prepared::new(&mut ctx, &fixture, test.name).await);
            }
            let mut primer = if matches!(condition, "none" | "transfer") {
                None
            } else {
                Some(Prepared::new(&mut ctx, &fixture, condition).await)
            };
            let label = format!("mode={mode} condition={condition} repetition={repetition}");
            measure(
                &mut ctx,
                &fixture,
                test,
                &collector,
                &mut seed,
                &format!("{label} phase=seed"),
                0,
            )
            .await;
            let slept = Instant::now();
            tokio::time::sleep(Duration::from_millis(2000)).await;
            println!(
                "DIAG_SLEEP {label} delay_ms=2000 actual_us={}",
                slept.elapsed().as_micros()
            );
            if condition == "transfer" {
                let started = Instant::now();
                seed.read_input(&mut ctx).await;
                println!(
                    "DIAG_TRANSFER {label} elapsed_ns={}",
                    started.elapsed().as_nanos()
                );
            }
            if let Some(ref mut input) = primer {
                let primer_test = TESTS.iter().find(|test| test.name == condition).unwrap();
                measure(
                    &mut ctx,
                    &fixture,
                    primer_test,
                    &collector,
                    input,
                    &format!("{label} phase=primer"),
                    0,
                )
                .await;
            }
            for (burst, input) in targets.iter_mut().enumerate() {
                measure(
                    &mut ctx,
                    &fixture,
                    test,
                    &collector,
                    input,
                    &format!("{label} phase=measure"),
                    burst,
                )
                .await;
            }
        }
    }
    println!(
        "DIAG_COMPLETE kernel={kernel} mode={mode} measured={} seed={} primer={} bootstrap=3",
        conditions.len() * 9,
        conditions.len() * 3,
        conditions
            .iter()
            .filter(|condition| !matches!(**condition, "none" | "transfer"))
            .count()
            * 3
    );
}

struct TinyWarmup {
    input: HbmTensor<bf16, Chip, m![16]>,
    output: HbmTensor<bf16, Chip, m![16]>,
}

impl TinyWarmup {
    async fn prepare(ctx: &mut Context) -> Self {
        let bytes: Vec<u8> = (0..16).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
        Self {
            input: HostTensor::<bf16, m![16]>::from_buf(bytes)
                .to_hbm(&mut ctx.pdma)
                .await,
            output: HostTensor::<bf16, m![16]>::from_buf(vec![0; 32])
                .to_hbm(&mut ctx.pdma)
                .await,
        }
    }
    async fn launch(&mut self, ctx: &mut Context) {
        launch(
            furiosa_opt_gemma4::diagnostics::warmup_copy,
            (ctx, &self.input, &mut self.output),
        )
        .await;
    }
    async fn check(&self, ctx: &mut Context) {
        assert_eq!(read_bf16(ctx, &self.output).await, vec![1.0; 16]);
    }
}

fn check_outputs(fixture: &Fixture, test: &Test, outputs: &[(&'static str, Vec<f32>)]) {
    assert!(!outputs.is_empty());
    println!("==> {}", test.name);
    for (label, actual) in outputs {
        assert!(compare(
            label,
            fixture.expect(test.name, label),
            actual,
            test.atol,
            test.rtol
        ));
    }
}

async fn benchmark_cost(fixture: &Fixture, test: &Test) {
    let mut ctx = Context::acquire();
    let attention = TESTS
        .iter()
        .find(|t| t.name == "sliding_attention_output")
        .unwrap();
    // Preload both warmup entry points and the target before any idle trial.
    for bootstrap in [test, attention] {
        let mut input = Prepared::new(&mut ctx, fixture, bootstrap.name).await;
        input.launch(&mut ctx).await;
        check_outputs(fixture, bootstrap, &input.read_outputs(&mut ctx).await);
    }
    let mut tiny_bootstrap = TinyWarmup::prepare(&mut ctx).await;
    tiny_bootstrap.launch(&mut ctx).await;
    tiny_bootstrap.check(&mut ctx).await;

    let conditions = ["none", "tiny_copy", "attention"];
    for repetition in 0..3 {
        for offset in 0..conditions.len() {
            let condition = conditions[(offset + repetition) % conditions.len()];
            let mut seed = Prepared::new(&mut ctx, fixture, test.name).await;
            let mut targets = Vec::new();
            for _ in 0..3 {
                targets.push(Prepared::new(&mut ctx, fixture, test.name).await);
            }
            let mut tiny = TinyWarmup::prepare(&mut ctx).await;
            let mut primer = Prepared::new(&mut ctx, fixture, attention.name).await;
            // Identical setup for every condition. Its cost is excluded from the
            // request latency: this models a service with already-loaded weights.
            seed.launch(&mut ctx).await;
            check_outputs(fixture, test, &seed.read_outputs(&mut ctx).await);
            tokio::time::sleep(Duration::from_secs(2)).await;

            // No logging, correctness checks or profiler waits in this region.
            // Warmup result reads are deferred; the target's output is read before
            // stopping each completion timer. This includes runtime and output I/O.
            let started = Instant::now();
            match condition {
                "tiny_copy" => tiny.launch(&mut ctx).await,
                "attention" => {
                    primer.launch(&mut ctx).await;
                }
                "none" => {}
                _ => unreachable!(),
            }
            let warmup_ns = started.elapsed().as_nanos();
            let mut outputs = Vec::new();
            let mut completion_ns = Vec::new();
            let mut launch_ns = Vec::new();
            for input in &mut targets {
                launch_ns.push(input.launch(&mut ctx).await);
                outputs.push(input.read_outputs(&mut ctx).await);
                completion_ns.push(started.elapsed().as_nanos());
            }
            let three_total_ns = completion_ns[2];
            println!(
                "COST_RESULT kernel={} condition={} repetition={} warmup_submit_ns={} first_total_ns={} first_target_ns={} three_total_ns={} second_ns={} third_ns={} first_launch_ns={} pid={}",
                test.name,
                condition,
                repetition,
                warmup_ns,
                completion_ns[0],
                completion_ns[0] - warmup_ns,
                three_total_ns,
                completion_ns[1] - completion_ns[0],
                completion_ns[2] - completion_ns[1],
                launch_ns[0],
                std::process::id()
            );
            for output in outputs {
                check_outputs(fixture, test, &output);
            }
            match condition {
                "tiny_copy" => tiny.check(&mut ctx).await,
                "attention" => {
                    check_outputs(fixture, attention, &primer.read_outputs(&mut ctx).await)
                }
                _ => {}
            }
        }
    }
    println!(
        "DIAG_COMPLETE kernel={} mode=cost trials=9 measured=27",
        test.name
    );
}
