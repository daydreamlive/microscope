#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#include "camera.h"
#include <cstdio>

@interface CameraDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property (nonatomic) Camera::FrameCallback callback;
@end

@implementation CameraDelegate

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection
{
    if (!self.callback) return;

    CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!pb) return;

    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    int w = (int)CVPixelBufferGetWidth(pb);
    int h = (int)CVPixelBufferGetHeight(pb);
    int stride = (int)CVPixelBufferGetBytesPerRow(pb);
    const uint8_t* data = (const uint8_t*)CVPixelBufferGetBaseAddress(pb);
    self.callback(data, w, h, stride);
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
}

@end

struct CameraImpl {
    AVCaptureSession* session = nil;
    CameraDelegate* delegate = nil;
    dispatch_queue_t queue = nil;
};

Camera::Camera() : impl_(nullptr) {}

Camera::~Camera() {
    stop();
    if (impl_) {
        delete static_cast<CameraImpl*>(impl_);
        impl_ = nullptr;
    }
}

void Camera::set_callback(FrameCallback cb) {
    if (!impl_) return;
    static_cast<CameraImpl*>(impl_)->delegate.callback = std::move(cb);
}

bool Camera::open(int device_id, int width, int height) {
    @autoreleasepool {
        auto* ci = new CameraImpl();

        // Find camera device by index
        AVCaptureDeviceDiscoverySession* discovery = [AVCaptureDeviceDiscoverySession
            discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera,
                                              AVCaptureDeviceTypeExternal]
                                  mediaType:AVMediaTypeVideo
                                   position:AVCaptureDevicePositionUnspecified];

        NSArray<AVCaptureDevice*>* devices = discovery.devices;
        if (device_id >= (int)devices.count) {
            fprintf(stderr, "Camera %d not found (%d available)\n",
                    device_id, (int)devices.count);
            delete ci;
            return false;
        }

        AVCaptureDevice* device = devices[device_id];
        fprintf(stderr, "Camera: %s\n", [[device localizedName] UTF8String]);

        NSError* error = nil;
        AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:device
                                                                            error:&error];
        if (error || !input) {
            fprintf(stderr, "Camera input error: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            delete ci;
            return false;
        }

        ci->session = [[AVCaptureSession alloc] init];
        ci->session.sessionPreset = AVCaptureSessionPreset640x480;

        if (![ci->session canAddInput:input]) {
            fprintf(stderr, "Cannot add camera input\n");
            delete ci;
            return false;
        }
        [ci->session addInput:input];

        AVCaptureVideoDataOutput* output = [[AVCaptureVideoDataOutput alloc] init];
        output.alwaysDiscardsLateVideoFrames = YES;
        output.videoSettings = @{
            (NSString*)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
        };

        ci->delegate = [[CameraDelegate alloc] init];
        ci->queue = dispatch_queue_create("camera.capture", DISPATCH_QUEUE_SERIAL);
        [output setSampleBufferDelegate:ci->delegate queue:ci->queue];

        if (![ci->session canAddOutput:output]) {
            fprintf(stderr, "Cannot add camera output\n");
            delete ci;
            return false;
        }
        [ci->session addOutput:output];

        impl_ = ci;
        fprintf(stderr, "Camera opened: %dx%d BGRA\n", width, height);
        return true;
    }
}

bool Camera::start() {
    if (!impl_) return false;
    auto* ci = static_cast<CameraImpl*>(impl_);
    [ci->session startRunning];
    fprintf(stderr, "Camera started\n");
    return true;
}

void Camera::stop() {
    if (!impl_) return;
    auto* ci = static_cast<CameraImpl*>(impl_);
    if (ci->session && ci->session.isRunning) {
        [ci->session stopRunning];
        fprintf(stderr, "Camera stopped\n");
    }
}
