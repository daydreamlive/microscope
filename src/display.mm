#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "display.h"
#include <vector>
#include <cstdio>
#include <cstring>

@interface DisplayView : NSView
@end

@implementation DisplayView
- (BOOL)acceptsFirstResponder { return YES; }
@end

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@implementation AppDelegate

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)app {
    return YES;
}

- (void)windowWillClose:(NSNotification*)notification {
    [NSApp terminate:nil];
}

@end

struct DisplayImpl {
    NSWindow* window = nil;
    DisplayView* view = nil;
    CAMetalLayer* metalLayer = nil;
    AppDelegate* appDelegate = nil;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    std::vector<uint8_t> frame_buf;
};

Display::Display() : impl_(nullptr) {}

Display::~Display() {
    if (impl_) {
        delete static_cast<DisplayImpl*>(impl_);
        impl_ = nullptr;
    }
}

bool Display::create(const std::string& title, int width, int height) {
    @autoreleasepool {
        auto* di = new DisplayImpl();

        di->device = MTLCreateSystemDefaultDevice();
        if (!di->device) {
            fprintf(stderr, "Metal not available\n");
            delete di;
            return false;
        }
        di->commandQueue = [di->device newCommandQueue];

        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        di->appDelegate = [[AppDelegate alloc] init];
        [NSApp setDelegate:di->appDelegate];

        NSRect frame = NSMakeRect(100, 100, width, height);
        di->window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:NSWindowStyleMaskTitled |
                                NSWindowStyleMaskClosable |
                                NSWindowStyleMaskMiniaturizable
                        backing:NSBackingStoreBuffered
                          defer:NO];

        NSString* nsTitle = [NSString stringWithUTF8String:title.c_str()];
        [di->window setTitle:nsTitle];
        [di->window setDelegate:di->appDelegate];

        di->view = [[DisplayView alloc] initWithFrame:frame];

        di->metalLayer = [CAMetalLayer layer];
        di->metalLayer.device = di->device;
        di->metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        di->metalLayer.framebufferOnly = NO;
        di->metalLayer.drawableSize = CGSizeMake(width, height);
        di->metalLayer.contentsScale = di->window.backingScaleFactor;

        [di->view setLayer:di->metalLayer];
        [di->view setWantsLayer:YES];

        [di->window setContentView:di->view];
        [di->window makeFirstResponder:di->view];
        [di->window makeKeyAndOrderFront:nil];

        impl_ = di;
        fprintf(stderr, "Display created: %dx%d (scale=%.1f)\n",
                width, height, di->window.backingScaleFactor);
        return true;
    }
}

void Display::present_frame(const uint8_t* bgra, int width, int height) {
    if (!impl_) return;
    auto* di = static_cast<DisplayImpl*>(impl_);

    @autoreleasepool {
        di->metalLayer.drawableSize = CGSizeMake(width, height);

        id<CAMetalDrawable> drawable = [di->metalLayer nextDrawable];
        if (!drawable) return;

        id<MTLTexture> texture = drawable.texture;

        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [texture replaceRegion:region
                   mipmapLevel:0
                     withBytes:bgra
                   bytesPerRow:width * 4];

        id<MTLCommandBuffer> cmdBuf = [di->commandQueue commandBuffer];
        [cmdBuf presentDrawable:drawable];
        [cmdBuf commit];
    }
}

void Display::present_frame_split(const uint8_t* left_pane, const uint8_t* right_pane,
                                  int pane_size, int gap) {
    if (!impl_) return;
    auto* di = static_cast<DisplayImpl*>(impl_);

    @autoreleasepool {
        int total_w = pane_size * 2 + gap;
        size_t row_bytes = (size_t)total_w * 4;
        size_t total_bytes = row_bytes * pane_size;
        if (di->frame_buf.size() != total_bytes) {
            di->frame_buf.resize(total_bytes);
            memset(di->frame_buf.data(), 30, total_bytes);
        }

        size_t left_row = (size_t)pane_size * 4;
        size_t right_off = (size_t)(pane_size + gap) * 4;
        for (int y = 0; y < pane_size; ++y) {
            uint8_t* dst = &di->frame_buf[y * row_bytes];
            memcpy(dst, left_pane + y * left_row, left_row);
            memcpy(dst + right_off, right_pane + y * left_row, left_row);
        }

        id<CAMetalDrawable> drawable = [di->metalLayer nextDrawable];
        if (!drawable) return;

        MTLRegion region = MTLRegionMake2D(0, 0, total_w, pane_size);
        [drawable.texture replaceRegion:region
                            mipmapLevel:0
                              withBytes:di->frame_buf.data()
                            bytesPerRow:row_bytes];

        id<MTLCommandBuffer> cmdBuf = [di->commandQueue commandBuffer];
        [cmdBuf presentDrawable:drawable];
        [cmdBuf commit];
    }
}

void Display::run_loop() {
    @autoreleasepool {
        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];
    }
}

void Display::request_stop() {
    dispatch_async(dispatch_get_main_queue(), ^{
        [NSApp terminate:nil];
    });
}
