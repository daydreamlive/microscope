#include "image_utils.h"
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ── Float16 conversion ──────────────────────────────────────────────────────

uint16_t f32_to_f16(float v) {
    uint32_t x;
    memcpy(&x, &v, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (x >> 13) & 0x3FF;

    if (exp <= 0) return sign;             // underflow → ±0
    if (exp >= 31) return sign | 0x7C00;   // overflow → ±inf
    return sign | (exp << 10) | frac;
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    if (exp == 0) {
        if (frac == 0) {
            float r; uint32_t v = sign;
            memcpy(&r, &v, 4);
            return r;
        }
        // Denormalized
        float r;
        uint32_t v = sign;
        memcpy(&r, &v, 4);
        return r; // treat denorms as zero for simplicity
    }
    if (exp == 31) {
        uint32_t v = sign | 0x7F800000 | (frac << 13);
        float r;
        memcpy(&r, &v, 4);
        return r;
    }

    uint32_t v = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    float r;
    memcpy(&r, &v, 4);
    return r;
}

// ── Image I/O ───────────────────────────────────────────────────────────────

bool load_image(const std::string& path, std::vector<uint8_t>& rgb,
                int& width, int& height) {
    int channels;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) return false;
    rgb.assign(data, data + width * height * 3);
    stbi_image_free(data);
    return true;
}

bool save_image(const std::string& path, const uint8_t* rgb,
                int width, int height) {
    return stbi_write_png(path.c_str(), width, height, 3, rgb, width * 3) != 0;
}

// ── vImage-accelerated resize ────────────────────────────────────────────────

static void resize_rgb(const uint8_t* src, int sw, int sh,
                       uint8_t* dst, int dw, int dh) {
    if (sw == dw && sh == dh) {
        memcpy(dst, src, (size_t)dw * dh * 3);
        return;
    }
    // vImage needs 4-channel; expand RGB→ARGB, scale, compress back
    size_t src_count = (size_t)sw * sh;
    size_t dst_count = (size_t)dw * dh;
    std::vector<uint8_t> src4(src_count * 4);
    std::vector<uint8_t> dst4(dst_count * 4);
    for (size_t i = 0; i < src_count; ++i) {
        src4[i * 4 + 0] = 255;
        src4[i * 4 + 1] = src[i * 3 + 0];
        src4[i * 4 + 2] = src[i * 3 + 1];
        src4[i * 4 + 3] = src[i * 3 + 2];
    }
    vImage_Buffer srcBuf = { src4.data(), (vImagePixelCount)sh, (vImagePixelCount)sw, (size_t)sw * 4 };
    vImage_Buffer dstBuf = { dst4.data(), (vImagePixelCount)dh, (vImagePixelCount)dw, (size_t)dw * 4 };
    vImageScale_ARGB8888(&srcBuf, &dstBuf, NULL, kvImageNoFlags);
    for (size_t i = 0; i < dst_count; ++i) {
        dst[i * 3 + 0] = dst4[i * 4 + 1];
        dst[i * 3 + 1] = dst4[i * 4 + 2];
        dst[i * 3 + 2] = dst4[i * 4 + 3];
    }
}

// ── Preprocess: RGB HWC uint8 → NCHW float16 [-1,1] ────────────────────────

void preprocess_frame(const uint8_t* rgb, int width, int height,
                      int render_size, uint16_t* out_nchw,
                      const uint16_t* lut,
                      std::vector<uint8_t>& sq_buf,
                      std::vector<uint8_t>& resized) {
    int rs = render_size;
    const uint8_t* src = rgb;

    // Fast path: input already at render_size — skip crop+resize entirely
    if (width == rs && height == rs) {
        // Convert HWC RGB → NCHW float16 using pre-computed LUT
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < rs; ++y) {
                for (int x = 0; x < rs; ++x) {
                    out_nchw[c * rs * rs + y * rs + x] = lut[src[(y * rs + x) * 3 + c]];
                }
            }
        }
        return;
    }

    // General path: center crop to square, resize, then convert
    int off_x = 0, off_y = 0;
    int crop_w = width, crop_h = height;
    if (width > height) {
        off_x = (width - height) / 2;
        crop_w = height;
    } else if (height > width) {
        off_y = (height - width) / 2;
        crop_h = width;
    }
    int sq = std::min(crop_w, crop_h);

    size_t sq_size = (size_t)sq * sq * 3;
    if (sq_buf.size() < sq_size) sq_buf.resize(sq_size);

    size_t resized_size = (size_t)rs * rs * 3;
    if (resized.size() < resized_size) resized.resize(resized_size);

    for (int y = 0; y < sq; ++y) {
        memcpy(&sq_buf[y * sq * 3],
               &rgb[((y + off_y) * width + off_x) * 3],
               sq * 3);
    }
    resize_rgb(sq_buf.data(), sq, sq, resized.data(), rs, rs);

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < rs; ++y) {
            for (int x = 0; x < rs; ++x) {
                out_nchw[c * rs * rs + y * rs + x] = lut[resized[(y * rs + x) * 3 + c]];
            }
        }
    }
}

// ── Postprocess: NCHW float16 → HWC RGB uint8 ──────────────────────────────

void postprocess_frame(const uint16_t* nchw_f16, int height, int width,
                       std::vector<uint8_t>& rgb_out) {
    int plane_size = height * width;
    rgb_out.resize(plane_size * 3);

    const uint16_t* r_plane = &nchw_f16[0];
    const uint16_t* g_plane = &nchw_f16[plane_size];
    const uint16_t* b_plane = &nchw_f16[2 * plane_size];

    float16x8_t v_scale = vdupq_n_f16((__fp16)127.5f);
    float16x8_t v_zero  = vdupq_n_f16((__fp16)0.0f);
    float16x8_t v_255   = vdupq_n_f16((__fp16)255.0f);

    int i = 0;
    for (; i + 8 <= plane_size; i += 8) {
        float16x8_t r = vld1q_f16((const __fp16*)&r_plane[i]);
        float16x8_t g = vld1q_f16((const __fp16*)&g_plane[i]);
        float16x8_t b = vld1q_f16((const __fp16*)&b_plane[i]);

        r = vminq_f16(vmaxq_f16(vfmaq_f16(v_scale, r, v_scale), v_zero), v_255);
        g = vminq_f16(vmaxq_f16(vfmaq_f16(v_scale, g, v_scale), v_zero), v_255);
        b = vminq_f16(vmaxq_f16(vfmaq_f16(v_scale, b, v_scale), v_zero), v_255);

        uint8x8x3_t rgb;
        rgb.val[0] = vqmovn_u16(vcvtq_u16_f16(r));
        rgb.val[1] = vqmovn_u16(vcvtq_u16_f16(g));
        rgb.val[2] = vqmovn_u16(vcvtq_u16_f16(b));
        vst3_u8(&rgb_out[i * 3], rgb);
    }
    for (; i < plane_size; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = f16_to_f32(nchw_f16[c * plane_size + i]);
            v = (v + 1.0f) * 127.5f;
            rgb_out[i * 3 + c] = (uint8_t)std::clamp(v, 0.0f, 255.0f);
        }
    }
}

void postprocess_frame_bgra(const uint16_t* nchw_f16, int height, int width,
                            std::vector<float>& bgra_out) {
    int plane_size = height * width;
    bgra_out.resize(plane_size * 4);

    const uint16_t* r_plane = &nchw_f16[0];
    const uint16_t* g_plane = &nchw_f16[plane_size];
    const uint16_t* b_plane = &nchw_f16[2 * plane_size];

    float16x8_t v_scale = vdupq_n_f16((__fp16)127.5f);
    float16x8_t v_zero  = vdupq_n_f16((__fp16)0.0f);
    float16x8_t v_255   = vdupq_n_f16((__fp16)255.0f);

    int i = 0;
    for (; i + 8 <= plane_size; i += 8) {
        float16x8_t r = vld1q_f16((const __fp16*)&r_plane[i]);
        float16x8_t g = vld1q_f16((const __fp16*)&g_plane[i]);
        float16x8_t b = vld1q_f16((const __fp16*)&b_plane[i]);

        r = vminq_f16(vmaxq_f16(vfmaq_f16(v_scale, r, v_scale), v_zero), v_255);
        g = vminq_f16(vmaxq_f16(vfmaq_f16(v_scale, g, v_scale), v_zero), v_255);
        b = vminq_f16(vmaxq_f16(vfmaq_f16(v_scale, b, v_scale), v_zero), v_255);

        float32x4_t a_val = vdupq_n_f32(255.0f);

        float32x4x4_t lo = {{ vcvt_f32_f16(vget_low_f16(b)),
                               vcvt_f32_f16(vget_low_f16(g)),
                               vcvt_f32_f16(vget_low_f16(r)), a_val }};
        vst4q_f32(&bgra_out[i * 4], lo);

        float32x4x4_t hi = {{ vcvt_f32_f16(vget_high_f16(b)),
                               vcvt_f32_f16(vget_high_f16(g)),
                               vcvt_f32_f16(vget_high_f16(r)), a_val }};
        vst4q_f32(&bgra_out[(i + 4) * 4], hi);
    }
    for (; i < plane_size; ++i) {
        float r = std::clamp((f16_to_f32(r_plane[i]) + 1.0f) * 127.5f, 0.0f, 255.0f);
        float g = std::clamp((f16_to_f32(g_plane[i]) + 1.0f) * 127.5f, 0.0f, 255.0f);
        float b = std::clamp((f16_to_f32(b_plane[i]) + 1.0f) * 127.5f, 0.0f, 255.0f);
        bgra_out[i * 4 + 0] = b;
        bgra_out[i * 4 + 1] = g;
        bgra_out[i * 4 + 2] = r;
        bgra_out[i * 4 + 3] = 255.0f;
    }
}

// ── BGRA helpers for camera/display pipeline ────────────────────────────────

void bgra_to_rgb(const uint8_t* bgra, int w, int h, int stride,
                 uint8_t* rgb) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgra + y * stride;
        for (int x = 0; x < w; ++x) {
            rgb[(y * w + x) * 3 + 0] = row[x * 4 + 2]; // R
            rgb[(y * w + x) * 3 + 1] = row[x * 4 + 1]; // G
            rgb[(y * w + x) * 3 + 2] = row[x * 4 + 0]; // B
        }
    }
}

void rgb_to_bgra(const uint8_t* rgb, int w, int h,
                 uint8_t* bgra) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int si = (y * w + x) * 3;
            int di = (y * w + x) * 4;
            bgra[di + 0] = rgb[si + 2]; // B
            bgra[di + 1] = rgb[si + 1]; // G
            bgra[di + 2] = rgb[si + 0]; // R
            bgra[di + 3] = 255;         // A
        }
    }
}

// vImage-accelerated BGRA resize
static void resize_bgra(const uint8_t* src, int sw, int sh,
                         uint8_t* dst, int dw, int dh) {
    if (sw == dw && sh == dh) {
        memcpy(dst, src, (size_t)dw * dh * 4);
        return;
    }
    vImage_Buffer srcBuf = { (void*)src, (vImagePixelCount)sh, (vImagePixelCount)sw, (size_t)sw * 4 };
    vImage_Buffer dstBuf = { dst, (vImagePixelCount)dh, (vImagePixelCount)dw, (size_t)dw * 4 };
    vImageScale_ARGB8888(&srcBuf, &dstBuf, NULL, kvImageNoFlags);
}

void crop_and_resize_bgra(const uint8_t* src, int sw, int sh, int stride,
                          uint8_t* dst, int dst_size) {
    // Center crop to square
    int sq = std::min(sw, sh);
    int off_x = (sw - sq) / 2;
    int off_y = (sh - sq) / 2;

    // Use vImageScale directly from cropped region (stride-aware)
    vImage_Buffer srcBuf = {
        (void*)(src + off_y * stride + off_x * 4),
        (vImagePixelCount)sq, (vImagePixelCount)sq, (size_t)stride
    };
    vImage_Buffer dstBuf = { dst, (vImagePixelCount)dst_size, (vImagePixelCount)dst_size, (size_t)dst_size * 4 };
    vImageScale_ARGB8888(&srcBuf, &dstBuf, NULL, kvImageNoFlags);
}
