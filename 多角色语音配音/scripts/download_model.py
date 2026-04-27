from pathlib import Path
model_dir = Path("Qwen3-TTS-CustomVoice-0.6B-fp16-ov")
if not model_dir.exists():
    print(f"模型不存在，开始下载到: {model_dir}")
    from modelscope import snapshot_download
    snapshot_download(
        model_id="snake7gun/Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
        local_dir=str(model_dir)
    )
    print(f"模型已下载到: {model_dir}")
else:
    print(f"模型已存在: {model_dir}，跳过下载")