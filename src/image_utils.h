#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Float16 ↔ Float32 conversion
uint16_t f32_to_f16(float v);
float f16_to_f32(uint16_t h);

// Load PNG → RGB uint8 buffer. Returns width, height, channels.
bool load_image(const std::string& path, std::vector<uint8_t>& rgb,
                int& width, int& height);

// Save RGB uint8 buffer → PNG
bool save_image(const std::string& path, const uint8_t* rgb,
                int width, int height);

// Preprocess: RGB uint8 HWC image → NCHW float16 buffer [-1,1]
// Crops to square, resizes to render_size, normalizes.
// Output buf must hold 1*3*render_size*render_size uint16_t values.
// lut: pre-computed pixel[0..255]→float16, sq_buf/resized: pre-allocated scratch
void preprocess_frame(const uint8_t* rgb, int width, int height,
                      int render_size, uint16_t* out_nchw,
                      const uint16_t* lut,
                      std::vector<uint8_t>& sq_buf,
                      std::vector<uint8_t>& resized);

// Postprocess: NCHW float16 (1,3,H,W) → HWC RGB uint8
void postprocess_frame(const uint16_t* nchw_f16, int height, int width,
                       std::vector<uint8_t>& rgb_out);

// Postprocess: NCHW float16 (1,3,H,W) → interleaved BGRA float [0,255]
void postprocess_frame_bgra(const uint16_t* nchw_f16, int height, int width,
                            std::vector<float>& bgra_out);

// BGRA helpers for camera/display pipeline
void bgra_to_rgb(const uint8_t* bgra, int w, int h, int stride,
                 uint8_t* rgb);

void rgb_to_bgra(const uint8_t* rgb, int w, int h,
                 uint8_t* bgra);

// Center-crop BGRA to square, then bilinear resize to dst_size x dst_size.
// Output is tightly packed (stride = dst_size * 4).
void crop_and_resize_bgra(const uint8_t* src, int sw, int sh, int stride,
                          uint8_t* dst, int dst_size);
