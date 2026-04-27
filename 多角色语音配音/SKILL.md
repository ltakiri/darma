---
name:多角色语音配音
description: 这是一个专为TTS（文本转语音）场景设计的剧本解析工具，可将带角色、情感、语音风格标记的戏剧
配音稿，自动解析为标准化语音合成片段，并提供文本清洗、片段摘要、多段音频智能拼接能力，完美适配多角色配音与
单语音播报两大场景，是语音生成流水线的核心预处理模块。
---
## 工作目录结构
---
drama-voice-skills/
├── requirements.txt           # 项目依赖清单
├── scripts/
│   ├── download_model.py      # Qwen3-TTS模型下载脚本
│   ├── gradio_drama_tab.py   # Gradio多角色配音UI核心模块
│   ├── qwen_3_tts_helper.py  # TTS推理核心封装（加载OV模型、生成音频）
│   ├── run.py                # 应用启动入口
│   ├── drama_script_parser.py # 短剧剧本解析工具（支持多角色台词拆分）
│   ├── Qwen3-TTS-CustomVoice-0.6B-fp16-ov/  # 预转换OpenVINO模型目录
│   ├── input/                 # 输入目录（存放参考音频、剧本文件）
│   │   ├── ref_voice.mp3      # 音色克隆参考音频
│   │   └── drama_script.txt   # 多角色剧本文件
│   └── output/                # 输出目录
│       └── drama_audio.wav    # 多角色合成音频
└── SKILL.md                   # 本技能说明文档
---
## 环境搭建
# 进入项目根目录
cd drama-voice-skills

# 创建并激活虚拟环境
# Windows
python -m venv ov_tts_env
ov_tts_env\Scripts\activate
# 升级pip并安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt
## 第二步：下载模型

```bash

cd scripts
python download_model.py
```
## 第三步：启动 Gradio UI

```python


python run.py 
```
