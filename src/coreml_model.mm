#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "coreml_model.h"
#include <cstdio>

CoreMLModel::CoreMLModel() : impl_(nullptr), prepared_provider_(nullptr), prepared_options_(nullptr) {}

CoreMLModel::~CoreMLModel() {
    if (prepared_options_) {
        CFBridgingRelease(prepared_options_);
        prepared_options_ = nullptr;
    }
    if (prepared_provider_) {
        CFBridgingRelease(prepared_provider_);
        prepared_provider_ = nullptr;
    }
    if (impl_) {
        CFBridgingRelease(impl_);
        impl_ = nullptr;
    }
}

bool CoreMLModel::load(const std::string& path, ComputeUnit cu) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:path.c_str()]];

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        switch (cu) {
            case ComputeUnit::CpuAndGpu:
                config.computeUnits = MLComputeUnitsCPUAndGPU;
                break;
            case ComputeUnit::CpuAndNeuralEngine:
                config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                break;
            case ComputeUnit::All:
                config.computeUnits = MLComputeUnitsAll;
                break;
        }

        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url
                                           configuration:config
                                                   error:&error];
        if (error || !model) {
            fprintf(stderr, "CoreML load error: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        impl_ = (void*)CFBridgingRetain(model);
        return true;
    }
}

// Zero-copy: wrap an existing buffer as MLMultiArray without alloc or memcpy.
// Caller must ensure `data` outlives the returned MLMultiArray.
static MLMultiArray* wrapMultiArray(const std::vector<int>& shape,
                                     void* data) {
    NSMutableArray<NSNumber*>* nsShape = [NSMutableArray arrayWithCapacity:shape.size()];
    NSMutableArray<NSNumber*>* nsStrides = [NSMutableArray arrayWithCapacity:shape.size()];

    // Compute C-contiguous strides (in elements)
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        [nsShape addObject:@(shape[i])];
        [nsStrides addObject:@(strides[i])];
    }

    NSError* error = nil;
    MLMultiArray* arr = [[MLMultiArray alloc]
        initWithDataPointer:data
                      shape:nsShape
                   dataType:MLMultiArrayDataTypeFloat16
                    strides:nsStrides
                deallocator:nil
                      error:&error];
    if (error) return nil;
    return arr;
}

bool CoreMLModel::prepare(
    const std::vector<std::pair<TensorDesc, const uint16_t*>>& inputs,
    std::vector<std::pair<TensorDesc, uint16_t*>>& outputs)
{
    @autoreleasepool {
        NSMutableDictionary<NSString*, MLFeatureValue*>* inputDict =
            [NSMutableDictionary dictionaryWithCapacity:inputs.size()];

        for (auto& [desc, data] : inputs) {
            MLMultiArray* arr = wrapMultiArray(desc.shape, (void*)data);
            if (!arr) return false;
            NSString* name = [NSString stringWithUTF8String:desc.name.c_str()];
            inputDict[name] = [MLFeatureValue featureValueWithMultiArray:arr];
        }

        NSError* error = nil;
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict
                                                             error:&error];
        if (error) return false;

        MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
        NSMutableDictionary<NSString*, id>* backings =
            [NSMutableDictionary dictionaryWithCapacity:outputs.size()];
        for (auto& [desc, data] : outputs) {
            MLMultiArray* arr = wrapMultiArray(desc.shape, data);
            if (!arr) return false;
            NSString* name = [NSString stringWithUTF8String:desc.name.c_str()];
            backings[name] = arr;
        }
        options.outputBackings = backings;

        if (prepared_provider_) CFBridgingRelease(prepared_provider_);
        if (prepared_options_) CFBridgingRelease(prepared_options_);
        prepared_provider_ = (void*)CFBridgingRetain(provider);
        prepared_options_ = (void*)CFBridgingRetain(options);

        return true;
    }
}

bool CoreMLModel::predict_prepared() {
    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)impl_;
        if (!model || !prepared_provider_ || !prepared_options_) return false;

        MLDictionaryFeatureProvider* provider =
            (__bridge MLDictionaryFeatureProvider*)prepared_provider_;
        MLPredictionOptions* options =
            (__bridge MLPredictionOptions*)prepared_options_;

        NSError* error = nil;
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                             options:options
                                                               error:&error];
        if (error || !result) {
            fprintf(stderr, "CoreML predict error: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }
        return true;
    }
}

bool CoreMLModel::predict(
    const std::vector<std::pair<TensorDesc, const uint16_t*>>& inputs,
    std::vector<std::pair<TensorDesc, uint16_t*>>& outputs)
{
    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)impl_;
        if (!model) return false;

        // Build input feature provider (zero-copy: wrap existing buffers)
        NSMutableDictionary<NSString*, MLFeatureValue*>* inputDict =
            [NSMutableDictionary dictionaryWithCapacity:inputs.size()];

        for (auto& [desc, data] : inputs) {
            MLMultiArray* arr = wrapMultiArray(desc.shape, (void*)data);
            if (!arr) return false;
            NSString* name = [NSString stringWithUTF8String:desc.name.c_str()];
            inputDict[name] = [MLFeatureValue featureValueWithMultiArray:arr];
        }

        NSError* error = nil;
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict
                                                             error:&error];
        if (error) return false;

        // Set up output backings (zero-copy: CoreML writes directly to our buffers)
        MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
        NSMutableDictionary<NSString*, id>* backings =
            [NSMutableDictionary dictionaryWithCapacity:outputs.size()];
        for (auto& [desc, data] : outputs) {
            MLMultiArray* arr = wrapMultiArray(desc.shape, data);
            if (!arr) return false;
            NSString* name = [NSString stringWithUTF8String:desc.name.c_str()];
            backings[name] = arr;
        }
        options.outputBackings = backings;

        // Run prediction (zero-copy in + out)
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                             options:options
                                                               error:&error];
        if (error || !result) {
            fprintf(stderr, "CoreML predict error: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        // Output data is already in our buffers via outputBackings — no copy needed
        return true;
    }
}

bool CoreMLModel::predict(
    const std::string& in_name, const std::vector<int>& in_shape, const uint16_t* in_data,
    const std::string& out_name, const std::vector<int>& out_shape, uint16_t* out_data)
{
    std::vector<std::pair<TensorDesc, const uint16_t*>> inputs = {
        {TensorDesc{in_name, in_shape}, in_data}
    };
    std::vector<std::pair<TensorDesc, uint16_t*>> outputs = {
        {TensorDesc{out_name, out_shape}, out_data}
    };
    return predict(inputs, outputs);
}
