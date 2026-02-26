# Microscope

On-device real-time Stable Diffusion style transfer for Apple Silicon. Processes camera frames through CoreML models (SDXS or SD-Turbo) and displays the result live in a Metal window.

## Requirements

- macOS with Apple Silicon
- Xcode Command Line Tools
- CMake 3.14+
- Python 3.10+ (for model conversion)

## Model Setup

```bash
pip install torch diffusers coremltools transformers

# Convert models
python scripts/convert_models.py --model sdxs --output-dir models/
python scripts/convert_models.py --model sd-turbo --output-dir models/

# Compile for CoreML
xcrun coremlcompiler compile models/text_encoder.mlpackage models/
xcrun coremlcompiler compile models/unet_sdxs_512.mlpackage models/
xcrun coremlcompiler compile models/unet_sd_turbo.mlpackage models/
xcrun coremlcompiler compile models/taesd_encoder_512.mlpackage models/
xcrun coremlcompiler compile models/taesd_decoder.mlpackage models/
```

## Build

```bash
make
```

## Usage

Camera mode:

```bash
./build/microscope --model-dir models/ --prompt "oil painting style"
./build/microscope --model-dir models/ --model sd-turbo --strength 0.6
```

Static image:

```bash
./build/microscope --model-dir models/ --image input.png --prompt "watercolor"
```

### Options

| Flag            | Description                 | Default    |
| --------------- | --------------------------- | ---------- |
| `--model-dir`   | Path to models/ directory   | (required) |
| `--model`       | `sdxs` or `sd-turbo`        | `sdxs`     |
| `--prompt`      | Style transfer prompt       | —          |
| `--image`       | Input PNG (static mode)     | —          |
| `--camera`      | Camera device ID            | `0`        |
| `--render-size` | Resolution                  | `512`      |
| `--strength`    | Diffusion strength          | `0.5`      |
| `--feedback`    | Latent feedback ratio       | `0.3`      |
| `--blend`       | Camera/AI blend (0=AI only) | `0.0`      |
| `--ema`         | Temporal smoothing          | `0.3`      |
