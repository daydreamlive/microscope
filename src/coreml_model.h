#pragma once
#include <string>
#include <vector>
#include <cstdint>

enum class ComputeUnit { CpuAndGpu, CpuAndNeuralEngine, All };

struct TensorDesc {
    std::string name;
    std::vector<int> shape;
    int64_t count() const {
        int64_t n = 1;
        for (int s : shape) n *= s;
        return n;
    }
};

class CoreMLModel {
public:
    CoreMLModel();
    ~CoreMLModel();
    CoreMLModel(const CoreMLModel&) = delete;
    CoreMLModel& operator=(const CoreMLModel&) = delete;

    bool load(const std::string& path, ComputeUnit cu = ComputeUnit::CpuAndGpu);

    // Run prediction with float16 (uint16_t*) buffers.
    // Each input/output pairs a TensorDesc with a raw buffer pointer.
    bool predict(
        const std::vector<std::pair<TensorDesc, const uint16_t*>>& inputs,
        std::vector<std::pair<TensorDesc, uint16_t*>>& outputs
    );

    // Convenience: single-input single-output with pre-allocated output buffer
    bool predict(
        const std::string& in_name, const std::vector<int>& in_shape, const uint16_t* in_data,
        const std::string& out_name, const std::vector<int>& out_shape, uint16_t* out_data
    );

private:
    void* impl_; // opaque ObjC pointer (MLModel*)
};
