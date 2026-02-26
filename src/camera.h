#pragma once
#include <functional>
#include <cstdint>

class Camera {
public:
    Camera();
    ~Camera();
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    using FrameCallback = std::function<void(const uint8_t* bgra, int w, int h, int stride)>;

    bool open(int device_id = 0, int width = 640, int height = 480);
    void set_callback(FrameCallback cb);
    bool start();
    void stop();

private:
    void* impl_; // opaque ObjC pointer
};
