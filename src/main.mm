#include "pipeline.h"
#include "image_utils.h"
#include "camera.h"
#include "display.h"
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <memory>
#include <future>
#include <dispatch/dispatch.h>
#include <sys/stat.h>
#include <mach-o/dyld.h>
#include <libgen.h>
#import <Cocoa/Cocoa.h>

static const char* DEFAULT_PROMPT = "oil painting style, masterpiece, highly detailed";

static const char* GITHUB_RELEASE_BASE =
    "https://github.com/livepeer/microscope/releases/latest/download/";

static std::string get_models_app_support_dir() {
    const char* home = getenv("HOME");
    if (!home) return "";
    return std::string(home) + "/Library/Application Support/microscope/models";
}

static bool download_and_extract(NSWindow* window, NSTextField* statusLabel,
                                 NSProgressIndicator* progressBar,
                                 const std::string& filename,
                                 const std::string& dest_dir) {
    std::string url_str = std::string(GITHUB_RELEASE_BASE) + filename;
    NSURL* url = [NSURL URLWithString:
        [NSString stringWithUTF8String:url_str.c_str()]];

    dispatch_sync(dispatch_get_main_queue(), ^{
        statusLabel.stringValue = [NSString stringWithFormat:@"Downloading %s...",
            filename.c_str()];
        progressBar.indeterminate = YES;
        [progressBar startAnimation:nil];
    });

    // Download synchronously (called from background thread)
    NSURLSessionConfiguration* config = [NSURLSessionConfiguration defaultSessionConfiguration];
    NSURLSession* session = [NSURLSession sessionWithConfiguration:config];

    __block NSData* data = nil;
    __block NSError* error = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);

    NSURLSessionDataTask* task = [session dataTaskWithURL:url
        completionHandler:^(NSData* d, NSURLResponse* r, NSError* e) {
            data = d;
            error = e;
            dispatch_semaphore_signal(sem);
        }];
    [task resume];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

    if (error || !data) {
        dispatch_sync(dispatch_get_main_queue(), ^{
            statusLabel.stringValue = [NSString stringWithFormat:@"Failed to download %s",
                filename.c_str()];
        });
        return false;
    }

    dispatch_sync(dispatch_get_main_queue(), ^{
        statusLabel.stringValue = [NSString stringWithFormat:@"Extracting %s...",
            filename.c_str()];
    });

    // Write zip to temp file
    NSString* tmpPath = [NSTemporaryDirectory()
        stringByAppendingPathComponent:
            [NSString stringWithUTF8String:filename.c_str()]];
    [data writeToFile:tmpPath atomically:YES];

    // Extract using ditto (handles zip on macOS)
    NSString* destPath = [NSString stringWithUTF8String:dest_dir.c_str()];
    NSTask* unzip = [[NSTask alloc] init];
    unzip.launchPath = @"/usr/bin/ditto";
    unzip.arguments = @[@"-xk", tmpPath, destPath];
    [unzip launch];
    [unzip waitUntilExit];

    [[NSFileManager defaultManager] removeItemAtPath:tmpPath error:nil];

    return unzip.terminationStatus == 0;
}

static bool has_file(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool show_download_dialog(const std::string&) {
    std::string dest = get_models_app_support_dir();
    if (dest.empty()) return false;

    // Figure out what needs downloading
    bool need_base = !has_file(dest + "/vocab.json");
    bool need_turbo = !has_file(dest + "/unet_sd_turbo.mlmodelc");

    if (!need_base && !need_turbo) return true;

    // Create directory
    [[NSFileManager defaultManager]
        createDirectoryAtPath:[NSString stringWithUTF8String:dest.c_str()]
        withIntermediateDirectories:YES attributes:nil error:nil];

    // Setup progress window
    NSWindow* window = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, 420, 140)
                  styleMask:NSWindowStyleMaskTitled
                    backing:NSBackingStoreBuffered
                      defer:NO];
    window.title = @"Microscope — Downloading Models";
    [window center];

    NSTextField* statusLabel = [NSTextField labelWithString:@"Preparing..."];
    statusLabel.frame = NSMakeRect(20, 90, 380, 20);
    [window.contentView addSubview:statusLabel];

    NSProgressIndicator* progressBar = [[NSProgressIndicator alloc]
        initWithFrame:NSMakeRect(20, 55, 380, 20)];
    progressBar.style = NSProgressIndicatorStyleBar;
    progressBar.indeterminate = YES;
    [window.contentView addSubview:progressBar];

    [window makeKeyAndOrderFront:nil];

    __block bool success = true;

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        if (need_base) {
            if (!download_and_extract(window, statusLabel, progressBar,
                                      "models-base.zip", dest))
                success = false;
        }
        if (success && need_turbo) {
            if (!download_and_extract(window, statusLabel, progressBar,
                                      "models-sd-turbo.zip", dest))
                success = false;
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            [window close];
            [NSApp stopModal];
        });
    });

    [NSApp runModalForWindow:window];
    return success;
}

static bool is_valid_model_dir(const std::string& path) {
    struct stat st;
    std::string vocab = path + "/vocab.json";
    std::string merges = path + "/merges.txt";
    return (stat(vocab.c_str(), &st) == 0 && stat(merges.c_str(), &st) == 0);
}

static std::string get_executable_dir() {
    char buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return "";
    char* resolved = realpath(buf, nullptr);
    if (!resolved) return "";
    std::string dir = dirname(resolved);
    free(resolved);
    return dir;
}

static std::string find_model_dir() {
    // 1. ~/Library/Application Support/microscope/models/
    const char* home = getenv("HOME");
    if (home) {
        std::string app_support = std::string(home) +
            "/Library/Application Support/microscope/models";
        if (is_valid_model_dir(app_support)) {
            fprintf(stderr, "Models found: %s\n", app_support.c_str());
            return app_support;
        }
    }

    std::string exe_dir = get_executable_dir();
    if (!exe_dir.empty()) {
        // 2. Adjacent to .app bundle (exe is at Microscope.app/Contents/MacOS/Microscope)
        std::string app_adjacent = exe_dir + "/../../../models";
        char* resolved = realpath(app_adjacent.c_str(), nullptr);
        if (resolved) {
            std::string path(resolved);
            free(resolved);
            if (is_valid_model_dir(path)) {
                fprintf(stderr, "Models found: %s\n", path.c_str());
                return path;
            }
        }

        // 3. Adjacent to CLI binary (exe is at build/microscope)
        std::string cli_adjacent = exe_dir + "/../models";
        resolved = realpath(cli_adjacent.c_str(), nullptr);
        if (resolved) {
            std::string path(resolved);
            free(resolved);
            if (is_valid_model_dir(path)) {
                fprintf(stderr, "Models found: %s\n", path.c_str());
                return path;
            }
        }
    }

    return "";
}

static void usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --model-dir <path> --image <input.png> --prompt <text>\n", prog);
    fprintf(stderr, "  %s --model-dir <path> --prompt <text> [--camera <id>]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --model-dir   Path to models/ directory\n");
    fprintf(stderr, "  --image       Input image (PNG) — static mode\n");
    fprintf(stderr, "  --camera      Camera device ID (default: 0) — camera mode\n");
    fprintf(stderr, "  --prompt      Text prompt for style transfer\n");
    fprintf(stderr, "  --output      Output image path (default: output.png)\n");
    fprintf(stderr, "  --strength    Diffusion strength (default: 0.5)\n");
    fprintf(stderr, "  --feedback    Latent feedback ratio (default: 0.3)\n");
    fprintf(stderr, "  --blend       Camera/AI blend ratio (default: 0.0, 0=AI only)\n");
    fprintf(stderr, "  --ema         EMA smoothing factor (default: 0.3)\n");
    fprintf(stderr, "  --model       Model name: sdxs or sd-turbo (default: sdxs)\n");
    fprintf(stderr, "  --render-size Render resolution (default: 512)\n");
}

static int run_image_mode(const std::string& model_dir,
                          const std::string& model_name,
                          const std::string& image_path,
                          const std::string& prompt,
                          const std::string& output_path,
                          int render_size, float strength, float feedback)
{
    fprintf(stderr, "microscope — Static Image Pipeline\n");
    fprintf(stderr, "  Model dir:   %s\n", model_dir.c_str());
    fprintf(stderr, "  Model:       %s\n", model_name.c_str());
    fprintf(stderr, "  Image:       %s\n", image_path.c_str());
    fprintf(stderr, "  Prompt:      %s\n", prompt.c_str());
    fprintf(stderr, "  Output:      %s\n", output_path.c_str());
    fprintf(stderr, "  Render size: %d\n", render_size);
    fprintf(stderr, "  Strength:    %.2f\n", strength);
    fprintf(stderr, "  Feedback:    %.2f\n", feedback);
    fprintf(stderr, "\n");

    PipelineConfig config;
    config.model_dir = model_dir;
    config.model = model_name;
    config.render_size = render_size;
    config.strength = strength;
    config.latent_feedback = feedback;

    Pipeline pipeline;
    if (!pipeline.init(config)) {
        fprintf(stderr, "Failed to initialize pipeline\n");
        return 1;
    }
    if (!pipeline.encode_prompt(prompt)) {
        fprintf(stderr, "Failed to encode prompt\n");
        return 1;
    }

    std::vector<uint8_t> rgb;
    int w, h;
    if (!load_image(image_path, rgb, w, h)) {
        fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
        return 1;
    }
    fprintf(stderr, "Loaded image: %dx%d\n", w, h);

    std::vector<uint8_t> result;
    if (!pipeline.process_frame(rgb.data(), w, h, result)) {
        fprintf(stderr, "Failed to process frame\n");
        return 1;
    }

    if (!save_image(output_path, result.data(), render_size, render_size)) {
        fprintf(stderr, "Failed to save output: %s\n", output_path.c_str());
        return 1;
    }

    fprintf(stderr, "Output saved: %s (%dx%d)\n", output_path.c_str(),
            render_size, render_size);
    return 0;
}

struct CameraState {
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    std::vector<uint8_t> cam_write_buf;
    std::vector<uint8_t> latest_frame_bgra;
    int frame_w = 0, frame_h = 0, frame_stride = 0;
    bool frame_ready = false;

    std::mutex crop_mutex;
    std::vector<uint8_t> cropped_bgra;
    bool crop_ready = false;

    std::mutex result_mutex;
    std::vector<uint8_t> result_ai_pane;
    bool result_ready = false;

    std::vector<uint8_t> cam_pane;
    std::vector<uint8_t> ai_pane;
    std::vector<uint8_t> blend_out;
    bool ai_valid = false;
    std::vector<float> blend_fa;
    std::vector<float> blend_fb;

    std::atomic<bool> running{true};
    std::atomic<bool> display_dirty{false};

    float blend = 0.0f;
    float ema_factor = 0.3f;
    int rs = 512;
};

static int run_camera_mode(const std::string& model_dir,
                           const std::string& model_name,
                           int camera_id,
                           const std::string& prompt,
                           int render_size, float strength, float feedback,
                           float blend_init, float ema_init)
{
    int rs = render_size;
    int display_w = rs * 2 + 10;
    int display_h = rs;

    fprintf(stderr, "microscope — Camera Mode\n");
    fprintf(stderr, "  Model dir:   %s\n", model_dir.c_str());
    fprintf(stderr, "  Model:       %s\n", model_name.c_str());
    fprintf(stderr, "  Camera:      %d\n", camera_id);
    fprintf(stderr, "  Prompt:      %s\n", prompt.c_str());
    fprintf(stderr, "  Render size: %d\n", rs);
    fprintf(stderr, "  Blend:       %.2f\n", blend_init);
    fprintf(stderr, "  EMA:         %.2f\n", ema_init);
    fprintf(stderr, "\n");

    PipelineConfig config;
    config.model_dir = model_dir;
    config.model = model_name;
    config.render_size = rs;
    config.strength = strength;
    config.latent_feedback = feedback;

    Pipeline pipeline;
    if (!pipeline.init(config)) {
        fprintf(stderr, "Failed to initialize pipeline\n");
        return 1;
    }
    if (!pipeline.encode_prompt(prompt)) {
        fprintf(stderr, "Failed to encode prompt\n");
        return 1;
    }

    auto state_ptr = std::make_unique<CameraState>();
    CameraState* st = state_ptr.get();
    st->rs = rs;
    st->blend = blend_init;
    st->ema_factor = ema_init;
    st->cropped_bgra.resize(rs * rs * 4, 0);
    st->result_ai_pane.resize(rs * rs * 4, 0);
    st->cam_pane.resize(rs * rs * 4, 0);
    st->ai_pane.resize(rs * rs * 4, 0);
    st->blend_out.resize(rs * rs * 4, 0);
    st->blend_fa.resize(rs * rs * 4, 0.0f);
    st->blend_fb.resize(rs * rs * 4, 0.0f);

    Camera camera;
    if (!camera.open(camera_id)) {
        fprintf(stderr, "Failed to open camera %d\n", camera_id);
        return 1;
    }

    camera.set_callback([st](const uint8_t* bgra, int w, int h, int stride) {
        int row_bytes = w * 4;
        st->cam_write_buf.resize(h * row_bytes);
        for (int y = 0; y < h; ++y)
            memcpy(&st->cam_write_buf[y * row_bytes], bgra + y * stride, row_bytes);

        // Update camera display pane at camera-native fps (decoupled from inference)
        {
            std::lock_guard<std::mutex> lock(st->crop_mutex);
            crop_and_resize_bgra(st->cam_write_buf.data(), w, h, row_bytes,
                                 st->cropped_bgra.data(), st->rs);
            st->crop_ready = true;
        }
        st->display_dirty.store(true, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lock(st->frame_mutex);
            st->latest_frame_bgra.swap(st->cam_write_buf);
            st->frame_w = w;
            st->frame_h = h;
            st->frame_stride = row_bytes;
            st->frame_ready = true;
        }
        st->frame_cv.notify_one();
    });

    if (!camera.start()) {
        fprintf(stderr, "Failed to start camera\n");
        return 1;
    }

    Display display;
    if (!display.create("microscope", display_w, display_h)) {
        fprintf(stderr, "Failed to create display\n");
        camera.stop();
        return 1;
    }

    std::thread inference_thread([st, &pipeline]() {
        int rs = st->rs;
        int pixel_count_4 = rs * rs * 4;
        std::vector<uint8_t> local_bgra;
        int lw, lh, lstride;
        std::vector<uint8_t> cropped_bgra(pixel_count_4);
        std::vector<float> result(pixel_count_4);

        std::vector<float> ema_buf(pixel_count_4, 0.0f);
        std::vector<float> scratch(pixel_count_4);
        std::vector<uint8_t> ai_ready(pixel_count_4);
        bool ema_initialized = false;
        bool has_pending_decode = false;

        int frame_count = 0;
        auto fps_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point prev_capture_time;
        float latency_sum_ms = 0.0f;
        int latency_count = 0;

        float ema_f = st->ema_factor;

        auto publish_result = [&]() {
            auto pub_time = std::chrono::steady_clock::now();
            float lat = std::chrono::duration<float, std::milli>(pub_time - prev_capture_time).count();
            latency_sum_ms += lat;
            latency_count++;

            pipeline.postprocess_bgra(result);

            if (!ema_initialized) {
                memcpy(ema_buf.data(), result.data(),
                       pixel_count_4 * sizeof(float));
                ema_initialized = true;
            } else {
                float one_minus_ema = 1.0f - ema_f;
                vDSP_vsmul(ema_buf.data(), 1, &ema_f,
                           ema_buf.data(), 1, pixel_count_4);
                vDSP_vsma(result.data(), 1, &one_minus_ema,
                          ema_buf.data(), 1,
                          ema_buf.data(), 1, pixel_count_4);
            }

            float lo = 0.0f, hi = 255.0f;
            vDSP_vclip(ema_buf.data(), 1, &lo, &hi,
                       scratch.data(), 1, pixel_count_4);
            vDSP_vfixru8(scratch.data(), 1,
                         ai_ready.data(), 1, pixel_count_4);

            {
                std::lock_guard<std::mutex> lock(st->result_mutex);
                st->result_ai_pane.swap(ai_ready);
                st->result_ready = true;
            }
            st->display_dirty.store(true, std::memory_order_release);
        };

        while (st->running.load()) {
            {
                std::unique_lock<std::mutex> lock(st->frame_mutex);
                st->frame_cv.wait(lock, [st]{ return st->frame_ready || !st->running.load(); });
                if (!st->running.load()) break;
                local_bgra.swap(st->latest_frame_bgra);
                lw = st->frame_w;
                lh = st->frame_h;
                lstride = st->frame_stride;
                st->frame_ready = false;
            }
            auto capture_time = std::chrono::steady_clock::now();

            crop_and_resize_bgra(local_bgra.data(), lw, lh, lstride,
                                 cropped_bgra.data(), rs);

            pipeline.preprocess_bgra(cropped_bgra.data());
            if (!pipeline.vae_encode_stage()) continue;
            pipeline.latent_noise_stage();

            auto unet_future = std::async(std::launch::async,
                [&pipeline]{ return pipeline.unet_stage(); });

            if (has_pending_decode) {
                if (pipeline.vae_decode_stage()) {
                    publish_result();
                }
            }

            if (!unet_future.get()) {
                fprintf(stderr, "UNet prediction failed\n");
                has_pending_decode = false;
                continue;
            }
            pipeline.denoise_stage();
            has_pending_decode = true;
            prev_capture_time = capture_time;

            if (++frame_count % 30 == 0) {
                auto now = std::chrono::steady_clock::now();
                float elapsed = std::chrono::duration<float>(now - fps_start).count();
                float avg_lat = latency_count > 0 ? latency_sum_ms / latency_count : 0.0f;
                fprintf(stderr, "[FPS] %.1f fps (%.1f ms/frame) | [Latency] %.1f ms avg\n",
                        frame_count / elapsed, elapsed / frame_count * 1000.0f, avg_lat);
                latency_sum_ms = 0.0f;
                latency_count = 0;
            }
        }

        if (has_pending_decode) {
            if (pipeline.vae_decode_stage()) {
                publish_result();
            }
        }
    });

    Display* disp = &display;

    dispatch_source_t timer = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_main_queue());
    dispatch_source_set_timer(timer,
        dispatch_time(DISPATCH_TIME_NOW, 0),
        NSEC_PER_SEC / 60,
        NSEC_PER_MSEC);

    dispatch_source_set_event_handler(timer, ^{
        if (!st->running.load()) return;
        if (!st->display_dirty.exchange(false)) return;

        int rs = st->rs;

        {
            std::lock_guard<std::mutex> lock(st->crop_mutex);
            if (st->crop_ready) {
                memcpy(st->cam_pane.data(), st->cropped_bgra.data(), rs * rs * 4);
                st->crop_ready = false;
            }
        }

        {
            std::lock_guard<std::mutex> lock(st->result_mutex);
            if (st->result_ready) {
                memcpy(st->ai_pane.data(), st->result_ai_pane.data(), rs * rs * 4);
                st->ai_valid = true;
                st->result_ready = false;
            }
        }

        const uint8_t* right = st->ai_pane.data();
        if (st->ai_valid && st->blend > 0.0f) {
            int n = rs * rs * 4;
            float inv_b = 1.0f - st->blend;
            vDSP_vfltu8(st->ai_pane.data(), 1,
                        st->blend_fa.data(), 1, n);
            vDSP_vfltu8(st->cam_pane.data(), 1,
                        st->blend_fb.data(), 1, n);
            vDSP_vsmul(st->blend_fa.data(), 1, &inv_b,
                       st->blend_fa.data(), 1, n);
            vDSP_vsma(st->blend_fb.data(), 1, &st->blend,
                      st->blend_fa.data(), 1,
                      st->blend_fa.data(), 1, n);
            vDSP_vfixru8(st->blend_fa.data(), 1,
                         st->blend_out.data(), 1, n);
            right = st->blend_out.data();
        }

        disp->present_frame_split(st->cam_pane.data(), right, rs, 10);
    });

    dispatch_resume(timer);

    display.run_loop();

    st->running.store(false);
    st->frame_cv.notify_one();
    camera.stop();
    dispatch_source_cancel(timer);
    if (inference_thread.joinable()) {
        inference_thread.join();
    }

    fprintf(stderr, "Done.\n");
    return 0;
}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string model_name = "sd-turbo";
    std::string image_path;
    std::string prompt;
    std::string output_path = "output.png";
    float strength = 0.5f;
    float feedback = 0.3f;
    float blend = 0.0f;
    float ema = 0.3f;
    int render_size = 512;
    int camera_id = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_name = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--strength" && i + 1 < argc) {
            strength = std::atof(argv[++i]);
        } else if (arg == "--feedback" && i + 1 < argc) {
            feedback = std::atof(argv[++i]);
        } else if (arg == "--blend" && i + 1 < argc) {
            blend = std::atof(argv[++i]);
        } else if (arg == "--ema" && i + 1 < argc) {
            ema = std::atof(argv[++i]);
        } else if (arg == "--camera" && i + 1 < argc) {
            camera_id = std::atoi(argv[++i]);
        } else if (arg == "--render-size" && i + 1 < argc) {
            render_size = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (model_dir.empty()) {
        model_dir = find_model_dir();
    }
    if (model_dir.empty()) {
        fprintf(stderr, "Models not found — starting automatic download\n");

        @autoreleasepool {
            [NSApplication sharedApplication];
            [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

            if (show_download_dialog(model_name)) {
                model_dir = get_models_app_support_dir();
                fprintf(stderr, "Models installed: %s\n", model_dir.c_str());
            } else {
                NSAlert* errAlert = [[NSAlert alloc] init];
                errAlert.messageText = @"Download Failed";
                errAlert.informativeText =
                    @"Could not download models. Check your internet connection "
                     "and try again.";
                errAlert.alertStyle = NSAlertStyleCritical;
                [errAlert runModal];
                return 1;
            }
        }
    }

    if (!image_path.empty()) {
        if (prompt.empty()) {
            fprintf(stderr, "Error: --prompt is required for image mode\n\n");
            usage(argv[0]);
            return 1;
        }
        return run_image_mode(model_dir, model_name, image_path, prompt,
                              output_path, render_size, strength, feedback);
    }

    if (camera_id < 0) camera_id = 0;
    if (prompt.empty()) prompt = DEFAULT_PROMPT;

    return run_camera_mode(model_dir, model_name, camera_id, prompt,
                           render_size, strength, feedback, blend, ema);
}
