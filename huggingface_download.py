from huggingface_hub import hf_hub_download
# Download default FP16 MLX safetensors weight file
weights_path = hf_hub_download(
    repo_id="uqer1244/mlx_lingbot-map",
    filename="lingbot-map-fp16.safetensors",
    local_dir="checkpoints"
)
print(f"Weights downloaded to: {weights_path}")
