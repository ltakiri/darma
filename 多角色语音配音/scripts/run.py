# 1. 导入依赖
from pathlib import Path
from qwen_3_tts_helper import OVQwen3TTSModel
from gradio_drama_tab import make_drama_demo  # 独立多语音UI的创建函数

# 2. 加载模型（和单语音UI的模型加载逻辑一致）
# 替换为你的模型路径
model_dir = Path("Qwen3-TTS-CustomVoice-0.6B-fp16-ov")

# 初始化模型（device根据实际情况选 "CPU"/"GPU"/"NPU"）
ov_model = OVQwen3TTSModel.from_pretrained(
    model_dir=model_dir,
    device="CPU",
)

# 3. 创建多语音UI（Audio Drama独立界面）
demo = make_drama_demo(ov_model, model_type=ov_model.tts_model_type)

# 4. 启动UI（Jupyter中会直接嵌入显示，或点击输出的本地链接打开）
try:
    # 本地启动（默认端口7860）
    demo.launch(debug=True)
except Exception:
    # 若本地启动失败，生成临时公网链接
    demo.launch(debug=True, share=True)