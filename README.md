[中文](README-CN.md)|[English](README.md)

# ComfyUI Nodes for EraX-WoW-Turbo: Ultra-Fast Multi-Language Speech Recognition

![](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo/blob/main/images/2025-03-23_06-38-22.png)

Based `on Whisper Large-v3 Turbo` training, special training was conducted on the following eight languages:

- Vietnamese
- Hindi
- Chinese
- English
- Russian
- German
- Ukrainian
- Japanese
- French
- Dutch
- Korean

## 📣 Updates

[2025-03-25]⚒️: Added `Whisper Large-v3 Turbo` model, capable of speech recognition and generating timestamped text.

![](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo/blob/main/images/2025-03-25_17-53-56.png)

[2025-03-23]⚒️: Released version v1.0.0.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo.git
```

## Model Download

- [EraX-WoW-Turbo-V1.0](https://huggingface.co/erax-ai/EraX-WoW-Turbo-V1.0): Download and place in the `ComfyUI/models/TTS` directory.
- [whisper-large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt): Download and place in the `ComfyUI/models/TTS/whisper-large-v3-turbo` directory.

## Acknowledgements

[EraX Team](https://huggingface.co/erax-ai/EraX-WoW-Turbo-V1.0)
[Whisper](https://github.com/openai/whisper)
