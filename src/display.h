#pragma once
#include <string>
#include <cstdint>

class Display {
public:
    Display();
    ~Display();
    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    bool create(const std::string& title, int width, int height);
    void present_frame(const uint8_t* bgra, int width, int height);
    void present_frame_split(const uint8_t* left_pane, const uint8_t* right_pane,
                             int pane_size, int gap);
    void run_loop();
    void request_stop();

private:
    void* impl_;
};
