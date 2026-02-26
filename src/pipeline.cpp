#include "pipeline.h"
#include <arm_neon.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <chrono>

// ── Noise schedule constants ────────────────────────────────────────────────
// Pre-computed from diffusers EulerDiscreteScheduler with 1 step (SDXS)
// and default scheduler with 50 steps (SD-Turbo).
//
// For SDXS (euler, 1 step):
//   timestep = 999, alpha_cumprod[999] ≈ 0.00466
//   sqrt_alpha ≈ 0.06827, sqrt(1-alpha) ≈ 0.99766
//
// For SD-Turbo (default, 50 steps, strength=0.5):
//   t_idx = max(0, int(50*(1-0.5))) = 25
//   timestep = timesteps[25] ≈ 501
//   alpha_cumprod[501] ≈ 0.5
//   sqrt_alpha ≈ 0.707, sqrt(1-alpha) ≈ 0.707
//
// These are computed at init time from the scheduler config embedded in the
// model. For now we use the SDXS defaults and allow overriding via strength.

static float compute_alpha_cumprod(int timestep) {
    // Diffusers "scaled_linear" schedule: beta_start=0.00085, beta_end=0.012
    // betas = linspace(sqrt(beta_start), sqrt(beta_end), 1000) ** 2
    float sqrt_beta_start = std::sqrt(0.00085f);
    float sqrt_beta_end   = std::sqrt(0.012f);
    float alpha_cumprod = 1.0f;
    for (int t = 0; t <= timestep; ++t) {
        float sqrt_beta = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * t / 999.0f;
        float beta = sqrt_beta * sqrt_beta;
        alpha_cumprod *= (1.0f - beta);
    }
    return alpha_cumprod;
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;
    latent_size_ = config.render_size / 8;

    std::string dir = config.model_dir;
    // Ensure trailing /
    if (!dir.empty() && dir.back() != '/') dir += '/';

    // Load tokenizer
    fprintf(stderr, "Loading tokenizer...\n");
    if (!tokenizer_.load(dir + "vocab.json", dir + "merges.txt")) return false;

    // Load CoreML models
    // VAE (tiny, ~1.2M params) → ANE to free GPU for UNet
    // UNet (large) → GPU only (ANE is slower for large models)
    ComputeUnit cu_vae  = ComputeUnit::CpuAndNeuralEngine;
    ComputeUnit cu_unet = ComputeUnit::CpuAndGpu;

    fprintf(stderr, "Loading text_encoder...\n");
    if (!text_encoder_.load(dir + "text_encoder.mlmodelc", cu_unet)) return false;

    fprintf(stderr, "Loading vae_encoder...\n");
    if (!vae_encoder_.load(dir + "taesd_encoder_512.mlmodelc", cu_vae)) return false;

    fprintf(stderr, "Loading vae_decoder...\n");
    if (!vae_decoder_.load(dir + "taesd_decoder.mlmodelc", cu_vae)) return false;

    // Resolve UNet filename from model name
    std::string unet_name;
    if (config.model == "sdxs") {
        unet_name = "unet_sdxs_512.mlmodelc";
    } else if (config.model == "sd-turbo") {
        unet_name = "unet_sd_turbo.mlmodelc";
    } else {
        fprintf(stderr, "Unknown model: %s (expected 'sdxs' or 'sd-turbo')\n",
                config.model.c_str());
        return false;
    }

    fprintf(stderr, "Loading unet (%s)...\n", config.model.c_str());
    if (!unet_.load(dir + unet_name, cu_unet)) return false;

    // Allocate buffers
    int rs = config.render_size;
    int ls = latent_size_;
    img_buf_.resize(1 * 3 * rs * rs);
    lat_buf_.resize(1 * 4 * ls * ls);
    out_buf_.resize(1 * 4 * ls * ls);
    prompt_embeds_.resize(1 * 77 * hidden_size_);
    fixed_noise_.resize(1 * 4 * ls * ls);
    prev_denoised_.resize(1 * 4 * ls * ls);

    // Pre-allocate per-frame buffers
    noisy_.resize(4 * ls * ls);
    dec_buf_.resize(1 * 3 * rs * rs);
    resized_buf_.resize(rs * rs * 3);
    // sq_buf_ sized dynamically on first use (depends on camera resolution)

    // Pre-compute pixel→float16 LUT
    for (int i = 0; i < 256; ++i) {
        lut_[i] = f32_to_f16(i / 127.5f - 1.0f);
    }

    // Compute noise schedule based on model type
    int timestep;
    if (config.model == "sdxs") {
        // Euler scheduler, 1 step → timestep 999
        timestep = 999;
    } else {
        // Default scheduler, 50 steps, strength-based
        int t_idx = std::max(0, (int)(50 * (1.0f - config.strength)));
        // Approximate: timesteps are evenly spaced from 999 down
        timestep = 999 - t_idx * (1000 / 50);
        if (timestep < 0) timestep = 0;
    }
    float ap = compute_alpha_cumprod(timestep);
    sqrt_alpha_ = std::sqrt(ap);
    sqrt_one_minus_alpha_ = std::sqrt(1.0f - ap);
    inv_sqrt_alpha_ = 1.0f / sqrt_alpha_;
    timestep_f16_ = f32_to_f16(static_cast<float>(timestep));

    fprintf(stderr, "Noise schedule: t=%d, alpha=%.6f, sqrt_a=%.6f, sqrt_1ma=%.6f\n",
            timestep, ap, sqrt_alpha_, sqrt_one_minus_alpha_);

    // Generate fixed noise (seed 42, matching Python's np.random.RandomState(42))
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < fixed_noise_.size(); ++i) {
        fixed_noise_[i] = f32_to_f16(dist(rng));
    }

    fprintf(stderr, "Pipeline ready: %dx%d → latent %dx%d\n",
            rs, rs, ls, ls);
    return true;
}

bool Pipeline::encode_prompt(const std::string& text) {
    auto it = prompt_cache_.find(text);
    if (it != prompt_cache_.end()) {
        fprintf(stderr, "Encoding prompt (cached): \"%s\"\n", text.c_str());
        memcpy(prompt_embeds_.data(), it->second.data(),
               it->second.size() * sizeof(uint16_t));
        return true;
    }

    fprintf(stderr, "Encoding prompt: \"%s\"\n", text.c_str());

    auto tokens = tokenizer_.tokenize(text);
    fprintf(stderr, "  Tokens (%zu): [", tokens.size());
    for (int i = 0; i < std::min((int)tokens.size(), 10); ++i) {
        fprintf(stderr, "%d%s", tokens[i], i < 9 ? ", " : "");
    }
    fprintf(stderr, "...]\n");

    std::vector<uint16_t> token_buf(77);
    for (int i = 0; i < 77; ++i) {
        token_buf[i] = f32_to_f16(static_cast<float>(tokens[i]));
    }

    TensorDesc in_desc{"input_ids", {1, 77}};
    TensorDesc out_desc{"last_hidden_state", {1, 77, hidden_size_}};

    std::vector<std::pair<TensorDesc, const uint16_t*>> inputs = {
        {in_desc, token_buf.data()}
    };
    std::vector<std::pair<TensorDesc, uint16_t*>> outputs = {
        {out_desc, prompt_embeds_.data()}
    };

    if (!text_encoder_.predict(inputs, outputs)) {
        fprintf(stderr, "Text encoder prediction failed\n");
        return false;
    }

    prompt_cache_[text] = std::vector<uint16_t>(
        prompt_embeds_.data(),
        prompt_embeds_.data() + prompt_embeds_.size());

    fprintf(stderr, "  Prompt encoded: shape (1, 77, %d)\n", hidden_size_);
    return true;
}

// ── Pipeline stages ─────────────────────────────────────────────────────────

void Pipeline::preprocess_bgra(const uint8_t* bgra) {
    int rs = config_.render_size;
    int plane_size = rs * rs;
    uint16_t* r_plane = &img_buf_[0];
    uint16_t* g_plane = &img_buf_[plane_size];
    uint16_t* b_plane = &img_buf_[2 * plane_size];

    float16x8_t v_scale = vdupq_n_f16((__fp16)(2.0f / 255.0f));
    float16x8_t v_offset = vdupq_n_f16((__fp16)(-1.0f));

    int i = 0;
    for (; i + 16 <= plane_size; i += 16) {
        uint8x16x4_t px = vld4q_u8(bgra + i * 4);

        float16x8_t r_lo = vcvtq_f16_u16(vmovl_u8(vget_low_u8(px.val[2])));
        float16x8_t g_lo = vcvtq_f16_u16(vmovl_u8(vget_low_u8(px.val[1])));
        float16x8_t b_lo = vcvtq_f16_u16(vmovl_u8(vget_low_u8(px.val[0])));
        vst1q_f16((__fp16*)&r_plane[i],     vfmaq_f16(v_offset, r_lo, v_scale));
        vst1q_f16((__fp16*)&g_plane[i],     vfmaq_f16(v_offset, g_lo, v_scale));
        vst1q_f16((__fp16*)&b_plane[i],     vfmaq_f16(v_offset, b_lo, v_scale));

        float16x8_t r_hi = vcvtq_f16_u16(vmovl_u8(vget_high_u8(px.val[2])));
        float16x8_t g_hi = vcvtq_f16_u16(vmovl_u8(vget_high_u8(px.val[1])));
        float16x8_t b_hi = vcvtq_f16_u16(vmovl_u8(vget_high_u8(px.val[0])));
        vst1q_f16((__fp16*)&r_plane[i + 8], vfmaq_f16(v_offset, r_hi, v_scale));
        vst1q_f16((__fp16*)&g_plane[i + 8], vfmaq_f16(v_offset, g_hi, v_scale));
        vst1q_f16((__fp16*)&b_plane[i + 8], vfmaq_f16(v_offset, b_hi, v_scale));
    }
    for (; i < plane_size; ++i) {
        int bi = i * 4;
        r_plane[i] = lut_[bgra[bi + 2]];
        g_plane[i] = lut_[bgra[bi + 1]];
        b_plane[i] = lut_[bgra[bi + 0]];
    }
}

bool Pipeline::vae_encode_stage() {
    int rs = config_.render_size;
    int ls = latent_size_;
    return vae_encoder_.predict(
        "image", {1, 3, rs, rs}, img_buf_.data(),
        "latent", {1, 4, ls, ls}, lat_buf_.data());
}

void Pipeline::latent_noise_stage() {
    int ls = latent_size_;
    size_t lat_count = 4 * ls * ls;

    if (has_prev_denoised_ && config_.latent_feedback > 0.0f) {
        float fb = config_.latent_feedback;
        float16x8_t v_fb = vdupq_n_f16((__fp16)fb);
        float16x8_t v_1mfb = vdupq_n_f16((__fp16)(1.0f - fb));
        size_t i = 0;
        for (; i + 8 <= lat_count; i += 8) {
            float16x8_t clean = vld1q_f16((__fp16*)&lat_buf_[i]);
            float16x8_t prev  = vld1q_f16((__fp16*)&prev_denoised_[i]);
            vst1q_f16((__fp16*)&lat_buf_[i],
                      vfmaq_f16(vmulq_f16(v_1mfb, clean), v_fb, prev));
        }
        for (; i < lat_count; ++i) {
            float clean = f16_to_f32(lat_buf_[i]);
            float prev  = f16_to_f32(prev_denoised_[i]);
            lat_buf_[i] = f32_to_f16((1.0f - fb) * clean + fb * prev);
        }
    }

    {
        float16x8_t v_sa = vdupq_n_f16((__fp16)sqrt_alpha_);
        float16x8_t v_s1ma = vdupq_n_f16((__fp16)sqrt_one_minus_alpha_);
        size_t i = 0;
        for (; i + 8 <= lat_count; i += 8) {
            float16x8_t clean = vld1q_f16((__fp16*)&lat_buf_[i]);
            float16x8_t noise = vld1q_f16((__fp16*)&fixed_noise_[i]);
            vst1q_f16((__fp16*)&noisy_[i],
                      vfmaq_f16(vmulq_f16(v_sa, clean), v_s1ma, noise));
        }
        for (; i < lat_count; ++i) {
            float clean = f16_to_f32(lat_buf_[i]);
            float noise = f16_to_f32(fixed_noise_[i]);
            noisy_[i] = f32_to_f16(sqrt_alpha_ * clean + sqrt_one_minus_alpha_ * noise);
        }
    }
}

bool Pipeline::unet_stage() {
    int ls = latent_size_;
    TensorDesc sample_desc{"sample", {1, 4, ls, ls}};
    TensorDesc timestep_desc{"timestep", {1}};
    TensorDesc embeds_desc{"encoder_hidden_states", {1, 77, hidden_size_}};
    TensorDesc npred_desc{"noise_pred", {1, 4, ls, ls}};
    std::vector<std::pair<TensorDesc, const uint16_t*>> unet_inputs = {
        {sample_desc, noisy_.data()},
        {timestep_desc, &timestep_f16_},
        {embeds_desc, prompt_embeds_.data()}
    };
    std::vector<std::pair<TensorDesc, uint16_t*>> unet_outputs = {
        {npred_desc, out_buf_.data()}
    };
    return unet_.predict(unet_inputs, unet_outputs);
}

void Pipeline::denoise_stage() {
    int ls = latent_size_;
    size_t lat_count = 4 * ls * ls;

    float16x8_t v_s1ma = vdupq_n_f16((__fp16)sqrt_one_minus_alpha_);
    float16x8_t v_inv_sa = vdupq_n_f16((__fp16)inv_sqrt_alpha_);
    size_t i = 0;
    for (; i + 8 <= lat_count; i += 8) {
        float16x8_t n = vld1q_f16((__fp16*)&noisy_[i]);
        float16x8_t npred = vld1q_f16((__fp16*)&out_buf_[i]);
        vst1q_f16((__fp16*)&out_buf_[i],
                  vmulq_f16(vfmsq_f16(n, v_s1ma, npred), v_inv_sa));
    }
    for (; i < lat_count; ++i) {
        float n = f16_to_f32(noisy_[i]);
        float npred = f16_to_f32(out_buf_[i]);
        out_buf_[i] = f32_to_f16((n - sqrt_one_minus_alpha_ * npred) * inv_sqrt_alpha_);
    }
    prev_denoised_.swap(out_buf_);
    has_prev_denoised_ = true;
}

bool Pipeline::vae_decode_stage() {
    int rs = config_.render_size;
    int ls = latent_size_;
    return vae_decoder_.predict(
        "latent", {1, 4, ls, ls}, prev_denoised_.data(),
        "image", {1, 3, rs, rs}, dec_buf_.data());
}

void Pipeline::postprocess_bgra(std::vector<float>& bgra_out) {
    int rs = config_.render_size;
    postprocess_frame_bgra(dec_buf_.data(), rs, rs, bgra_out);
}

// ── Convenience wrappers ────────────────────────────────────────────────────

bool Pipeline::run_diffusion() {
    using clock = std::chrono::steady_clock;

    auto t1 = clock::now();
    if (!vae_encode_stage()) { fprintf(stderr, "VAE encode failed\n"); return false; }
    auto t2 = clock::now();
    latent_noise_stage();
    auto t3 = clock::now();
    if (!unet_stage()) { fprintf(stderr, "UNet prediction failed\n"); return false; }
    auto t4 = clock::now();
    denoise_stage();
    auto t5 = clock::now();
    if (!vae_decode_stage()) { fprintf(stderr, "VAE decode failed\n"); return false; }
    auto t6 = clock::now();

    static int prof_count = 0;
    if (++prof_count % 30 == 0) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };
        fprintf(stderr, "[PROF] vae_enc=%.1f latent=%.1f unet=%.1f denoise=%.1f "
                "vae_dec=%.1f total=%.1f ms\n",
                ms(t1,t2), ms(t2,t3), ms(t3,t4),
                ms(t4,t5), ms(t5,t6), ms(t1,t6));
    }
    return true;
}

bool Pipeline::process_frame(const uint8_t* rgb_in, int width, int height,
                             std::vector<uint8_t>& rgb_out) {
    int rs = config_.render_size;
    preprocess_frame(rgb_in, width, height, rs, img_buf_.data(),
                     lut_, sq_buf_, resized_buf_);
    if (!run_diffusion()) return false;
    postprocess_frame(dec_buf_.data(), rs, rs, rgb_out);
    return true;
}

bool Pipeline::process_frame_bgra(const uint8_t* bgra, std::vector<float>& bgra_out) {
    preprocess_bgra(bgra);
    if (!run_diffusion()) return false;
    postprocess_bgra(bgra_out);
    return true;
}
