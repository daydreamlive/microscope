#pragma once
#include "coreml_model.h"
#include "tokenizer.h"
#include "image_utils.h"
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

struct PipelineConfig {
    std::string model_dir;      // Path to models/ directory
    std::string model = "sdxs"; // Model name: "sdxs" or "sd-turbo"
    int render_size = 512;      // Input resolution (320, 384, 512)
    float strength  = 0.5f;     // Diffusion strength
    float latent_feedback = 0.3f;
};

class Pipeline {
public:
    bool init(const PipelineConfig& config);

    // Encode a text prompt → (1,77,hidden_size) float16 embedding
    bool encode_prompt(const std::string& text);

    // Process a single frame: RGB HWC uint8 → RGB HWC uint8
    bool process_frame(const uint8_t* rgb_in, int width, int height,
                       std::vector<uint8_t>& rgb_out);

    // Process from BGRA directly (camera path: skips bgra→rgb + redundant resize)
    // Input must be render_size x render_size BGRA, tightly packed.
    bool process_frame_bgra(const uint8_t* bgra, std::vector<float>& bgra_out);

    // Fine-grained pipeline stages for pipelined execution.
    // Call sequence per frame: preprocess_bgra → vae_encode_stage → latent_noise_stage
    //   → unet_stage → denoise_stage → vae_decode_stage → postprocess_bgra
    // For pipelining: overlap unet_stage(N) with vae_decode_stage(N-1).
    void preprocess_bgra(const uint8_t* bgra);
    bool vae_encode_stage();
    void latent_noise_stage();
    bool unet_stage();
    void denoise_stage();
    bool vae_decode_stage();
    void postprocess_bgra(std::vector<float>& bgra_out);

private:
    bool run_diffusion();

    PipelineConfig config_;
    int latent_size_ = 0;
    int hidden_size_ = 1024; // CLIP hidden size for SD-Turbo/SDXS

    // CoreML models
    CoreMLModel text_encoder_;
    CoreMLModel vae_encoder_;
    CoreMLModel vae_decoder_;
    CoreMLModel unet_;

    CLIPTokenizer tokenizer_;
    std::unordered_map<std::string, std::vector<uint16_t>> prompt_cache_;

    // Pre-allocated buffers (float16 as uint16_t)
    std::vector<uint16_t> img_buf_;       // (1,3,render,render)
    std::vector<uint16_t> lat_buf_;       // (1,4,latent,latent)
    std::vector<uint16_t> out_buf_;       // (1,4,latent,latent)
    std::vector<uint16_t> prompt_embeds_; // (1,77,hidden_size)
    std::vector<uint16_t> fixed_noise_;   // (1,4,latent,latent)
    std::vector<uint16_t> prev_denoised_; // (1,4,latent,latent)
    bool has_prev_denoised_ = false;

    // Pre-allocated per-frame buffers (avoid allocs in hot path)
    std::vector<uint16_t> noisy_;         // (4*latent*latent)
    std::vector<uint16_t> dec_buf_;       // (1*3*render*render)
    std::vector<uint8_t>  sq_buf_;        // square crop buffer
    std::vector<uint8_t>  resized_buf_;   // resized rgb buffer

    // Pre-computed pixel→float16 LUT [0,255] → [-1,1]
    uint16_t lut_[256];

    uint16_t timestep_f16_;
    float sqrt_alpha_;
    float sqrt_one_minus_alpha_;
    float inv_sqrt_alpha_;
};
