[中文](README-CN.md)|[English](README.md)

# EraX-WoW-Turbo 的 ComfyUI 节点: 超快多语言语音识别

![](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo/blob/main/images/2025-03-23_06-38-22.png)

基于 `Whisper Large-v3 Turbo` 训练，以下八种语言上进行特别训练：

- 越南语
- 印地语
- 中文
- 英语
- 俄语
- 德语
- 乌克兰语
- 日语
- 法语
- 荷兰语
- 韩语

## 📣 更新

[2025-03-25]⚒️: 新增 `Whisper Large-v3 Turbo` 模型，可识别语音, 生成带时间戳的文本。

![](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo/blob/main/images/2025-03-25_17-53-56.png)

[2025-03-23]⚒️: 发布版本 v1.0.0. 

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo.git
cd ComfyUI_EraX-WoW-Turbo
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- [EraX-WoW-Turbo-V1.0](https://huggingface.co/erax-ai/EraX-WoW-Turbo-V1.0): 下载放到 `ComfyUI/models/TTS` 目录下.
- [whisper-large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt): 下载放到 `ComfyUI/models/TTS/whisper-large-v3-turbo` 目录下.

## 鸣谢

[EraX Team](https://huggingface.co/erax-ai/EraX-WoW-Turbo-V1.0)
[Whisper](https://github.com/openai/whisper)