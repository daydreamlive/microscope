#!/usr/bin/env python3
"""
Convert SD models to CoreML for microscope.

Converts: UNet, TAESD encoder/decoder, and CLIP text encoder.
Output: .mlpackage files (compile with `xcrun coremlcompiler compile`)

Usage:
    python convert_models.py --model sdxs --output-dir models/
    python convert_models.py --model sd-turbo --output-dir models/

After conversion, compile for instant loading:
    xcrun coremlcompiler compile models/text_encoder.mlpackage models/
    xcrun coremlcompiler compile models/unet_sdxs_512.mlpackage models/
    xcrun coremlcompiler compile models/taesd_encoder_512.mlpackage models/
    xcrun coremlcompiler compile models/taesd_decoder.mlpackage models/
"""

import os
import sys
import gc
import argparse
import numpy as np

MODEL_CONFIGS = {
    "sdxs": {
        "model_id": "IDKiro/sdxs-512-0.9",
        "hidden_size": 1024,
        "unet_prefix": "unet_sdxs_512",
        "scheduler": "euler",
    },
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "hidden_size": 1024,
        "unet_prefix": "unet_sd_turbo",
        "scheduler": "default",
    },
}


def convert_text_encoder(model_id, output_dir):
    """Convert CLIP text encoder to CoreML."""
    import torch
    import coremltools as ct
    from transformers import CLIPTextModel, CLIPTokenizer

    path = os.path.join(output_dir, "text_encoder.mlpackage")
    if os.path.exists(path):
        print(f"  Text encoder already exists: {path}")
        return path

    print("  Converting CLIP text encoder...")
    model = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    model.eval()

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids):
            return self.encoder(input_ids)[0]  # last_hidden_state

    wrapper = TextEncoderWrapper(model).eval()
    dummy_input = torch.randint(0, 49408, (1, 77))

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, 77), dtype=np.float16)],
        outputs=[ct.TensorType(name="last_hidden_state", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.save(path)
    print(f"  Saved: {path}")

    del mlmodel, traced, wrapper, model
    gc.collect()
    return path


def convert_vae_encoder(render_size, output_dir):
    """Convert TinyVAE encoder to CoreML."""
    import torch
    import coremltools as ct
    from diffusers import AutoencoderTiny

    path = os.path.join(output_dir, f"taesd_encoder_{render_size}.mlpackage")
    if os.path.exists(path):
        print(f"  VAE encoder already exists: {path}")
        return path

    print(f"  Converting TinyVAE encoder ({render_size}x{render_size})...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.encoder = v.encoder

        def forward(self, x):
            return self.encoder(x)

    wrapper = Wrapper(vae).eval()
    dummy = torch.randn(1, 3, render_size, render_size)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="latent", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.save(path)
    print(f"  Saved: {path}")

    del mlmodel, traced, wrapper, vae
    gc.collect()
    return path


def convert_vae_decoder(render_size, output_dir):
    """Convert TinyVAE decoder to CoreML."""
    import torch
    import coremltools as ct
    from diffusers import AutoencoderTiny

    latent_size = render_size // 8
    path = os.path.join(output_dir, "taesd_decoder.mlpackage")
    if os.path.exists(path):
        print(f"  VAE decoder already exists: {path}")
        return path

    print(f"  Converting TinyVAE decoder ({render_size}x{render_size})...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.decoder = v.decoder

        def forward(self, x):
            return self.decoder(x)

    wrapper = Wrapper(vae).eval()
    dummy = torch.randn(1, 4, latent_size, latent_size)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=dummy.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.save(path)
    print(f"  Saved: {path}")

    del mlmodel, traced, wrapper, vae
    gc.collect()
    return path


def convert_unet(model_id, cfg, render_size, output_dir):
    """Convert UNet to CoreML."""
    import torch
    import coremltools as ct
    from diffusers import StableDiffusionPipeline

    prefix = cfg["unet_prefix"]
    path = os.path.join(output_dir, f"{prefix}.mlpackage")
    if os.path.exists(path):
        print(f"  UNet already exists: {path}")
        return path

    latent_size = render_size // 8
    print(f"  Converting UNet ({prefix})...")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    unet = pipe.unet.eval().float().cpu()

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample, timestep, encoder_hidden_states).sample

    wrapper = UNetWrapper(unet).eval()
    hidden_size = cfg["hidden_size"]
    dummy_sample = torch.randn(1, 4, latent_size, latent_size)
    dummy_timestep = torch.tensor([999.0])
    dummy_hidden = torch.randn(1, 77, hidden_size)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_sample, dummy_timestep, dummy_hidden))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=dummy_sample.shape, dtype=np.float16),
            ct.TensorType(name="timestep", shape=(1,), dtype=np.float16),
            ct.TensorType(name="encoder_hidden_states", shape=dummy_hidden.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="noise_pred", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.save(path)
    print(f"  Saved: {path}")

    del mlmodel, traced, wrapper, unet, pipe
    gc.collect()
    return path


def export_tokenizer(model_id, output_dir):
    """Export CLIP tokenizer vocab.json and merges.txt."""
    from transformers import CLIPTokenizer

    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"  Tokenizer files already exist")
        return

    print("  Exporting tokenizer files...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    import json
    with open(vocab_path, "w") as f:
        json.dump(tokenizer.encoder, f)

    with open(merges_path, "w") as f:
        f.write("#version: 0.2\n")
        for merge in tokenizer.bpe_ranks.keys():
            f.write(f"{merge[0]} {merge[1]}\n")

    print(f"  Saved: {vocab_path}, {merges_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SD models to CoreML")
    parser.add_argument("--model", type=str, default="sdxs",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--render-size", type=int, default=512)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id = cfg["model_id"]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Converting {args.model} ({model_id}) → {args.output_dir}/")
    print()

    export_tokenizer(model_id, args.output_dir)
    convert_text_encoder(model_id, args.output_dir)
    convert_vae_encoder(args.render_size, args.output_dir)
    convert_vae_decoder(args.render_size, args.output_dir)
    convert_unet(model_id, cfg, args.render_size, args.output_dir)

    print()
    print("Done! Now compile for instant loading:")
    print(f"  for f in {args.output_dir}/*.mlpackage; do")
    print(f"    xcrun coremlcompiler compile \"$f\" {args.output_dir}/")
    print(f"  done")


if __name__ == "__main__":
    main()
