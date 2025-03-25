import torch
import torchaudio
import os
import folder_paths
import whisper


class WhisperTurboRun:
    models_dir = folder_paths.models_dir
    whisper_model_id = os.path.join(models_dir, "TTS", "whisper-large-v3-turbo")

    model_cache = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # "sample_len": ("INT", {"default": 300, "min": 1, "step": 1}),
                "logprob_threshold": ("FLOAT", {"default": -0.10, "min": -2.0, "max": -0.01, "step": 0.01}),
                "no_speech_threshold": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}),
                # "task": (["transcribe", "translate"], {"default": "transcribe", "tooltip": "translate: → English."}), 
                # "initial_prompt": ("STRING", {"default": "", "multiline": True}),
                "timestamp": ("BOOLEAN", {"default": False}),
                "word_timestamps": ("BOOLEAN", {"default": False}),
                # "hallucination_silence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 3.0, "step": 0.01}),
                "unload_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)  
    RETURN_NAMES = ("text",)  
    FUNCTION = "process_audio"
    CATEGORY = "🎤MW/MW-EraXWoW"

    def split_into_sentences(self, segments):
        sentences = []
        current_sentence =  {"timestamp": None, "text": ""}
        
        for segment in segments:
            for word in segment["words"]:
                if current_sentence["timestamp"] is None:
                    current_sentence["timestamp"] = []
                    current_sentence["timestamp"].append(round(word["start"], 2))
                current_sentence["text"] += word["word"]
                
                # 如果遇到句号或问号，结束当前句子
                if word["word"].endswith(("。", "，", "、", "：", "；", "？", "！", 
                                          "”", "’", "）", "——", "……", "》", ".", 
                                          ",", ";", ":", "?", "!", ")", "--", "…")):
                    current_sentence["timestamp"].append(round(word["end"], 2))
                    sentences.append(current_sentence)
                    current_sentence = {"timestamp": None, "text": ""}
        
        # 处理未结束的句子
        if current_sentence["text"]:
            sentences.append(current_sentence)
        
        return sentences
    def process_audio(self, audio, 
                    #   sample_len = 300,
                      logprob_threshold = -1.0, 
                      no_speech_threshold = 0.1,
                      word_timestamps = False,
                    #   hallucination_silence_threshold = 0.1,
                    #   task, 
                    #   initial_prompt="", 
                      unload_model=False, 
                      timestamp=False
                      ):
        if self.model_cache is None:
            model = whisper.load_model(f"{self.whisper_model_id}/large-v3-turbo.pt").to(self.device)
            self.model_cache = model
        else:
            model = self.model_cache

        waveform, sample_rate = audio["waveform"], audio["sample_rate"]
        waveform = waveform.squeeze(0)
        waveform = waveform.to(self.device)
                
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(self.device)
            waveform = resampler(waveform)
                
        if len(waveform.shape) > 1:
            waveform = torch.mean(waveform, dim=0)

        audio = whisper.pad_or_trim(waveform.float()).to(self.device)

        if not timestamp:
            word_timestamps = False
        
        result = model.transcribe(audio, 
                                # sample_len = sample_len,
                                # hallucination_silence_threshold = hallucination_silence_threshold,
                                logprob_threshold = logprob_threshold,
                                no_speech_threshold = no_speech_threshold,
                                # initial_prompt=initial_prompt, 
                                word_timestamps = word_timestamps,
                                # task=task, 
                                fp16=False if self.device == "cpu" else True
                                )

        if unload_model:
            import gc
            del model
            self.model_cache = None
            gc.collect()
            torch.cuda.empty_cache()

        if not timestamp:
            texts = []
            for segment in result["segments"]:
                text = segment['text'] + ", "
                texts.append(text)
            return ("".join(texts).removesuffix(", ") + ".",)
        else:
            if not word_timestamps:
                timestamped_segments = []
                for segment in result["segments"]:
                    timestamped_segments.append({
                        "timestamp": (round(segment['start'], 2), round(segment['end'], 2)), 
                        "text": segment["text"]
                    })
                return (str(timestamped_segments),)
            else:
                return (str(self.split_into_sentences(result["segments"])),)

