import torch
import torchaudio
import os
import folder_paths
import whisper


models_dir = folder_paths.models_dir
whisper_model_id = os.path.join(models_dir, "TTS", "whisper-large-v3-turbo")


def convert_subtitle_format(data):
  lines = []
  for entry in data:
    # We only need the start timestamp for this format
    start_time_seconds = entry['timestamp'][0]
    text = entry['text']

    # Convert seconds to minutes, seconds, and milliseconds
    # Work with total milliseconds to avoid floating point issues
    total_milliseconds = int(start_time_seconds * 1000)

    minutes = total_milliseconds // (1000 * 60)
    remaining_milliseconds = total_milliseconds % (1000 * 60)
    seconds = remaining_milliseconds // 1000
    milliseconds = remaining_milliseconds % 1000

    # Format the timestamp string as [MM:SS.mmm]
    # Use f-string formatting with zero-padding
    timestamp_str = f"[{minutes:02d}:{seconds:02d}.{milliseconds:03d}]"

    # Combine timestamp and text
    line = f"{timestamp_str}{text}"
    lines.append(line)

  # Join all lines with a newline character
  return "\n".join(lines)

import tempfile
from typing import Optional
import folder_paths
cache_dir = folder_paths.get_temp_directory()
def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")

CACHE_MODEL = None
class WhisperTurboRun:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_num_words_per_page": ("INT", {"default": 24, "min": 1, "max": 50}),
                # "sample_len": ("INT", {"default": 300, "min": 1, "step": 1}),
                "logprob_threshold": ("FLOAT", {"default": -0.10, "min": -2.0, "max": -0.01, "step": 0.01}),
                "no_speech_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                # "task": (["transcribe", "translate"], {"default": "transcribe", "tooltip": "translate: → English."}), 
                "initial_prompt": ("STRING", {"default": "如果有中文, 严格使用简体中文:", "multiline": True}),
                "timestamp": ("BOOLEAN", {"default": False}),
                "word_timestamps": ("BOOLEAN", {"default": False}),
                # "hallucination_silence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 3.0, "step": 0.01}),
                "unload_model": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")  
    RETURN_NAMES = ("json_text", "subtitle_text")  
    FUNCTION = "process_audio"
    CATEGORY = "🎤MW/MW-EraXWoW"

    def split_into_sentences(self, segments, max_num_words_per_page):
        sentences = []
        current_sentence =  {"timestamp": None, "text": ""}
        
        num_words = 0
        for segment in segments:
            for word in segment["words"]:
                if current_sentence["timestamp"] is None:
                    current_sentence["timestamp"] = []
                    current_sentence["timestamp"].append(round(word["start"], 2))
                current_sentence["text"] += word["word"]

                num_words += 1
                # 如果遇到句号或问号，结束当前句子
                if word["word"].endswith(("。", "，", "、", "：", "；", "？", "！", 
                                          "”", "’", "）", "——", "……", "》", ".", 
                                          ",", ";", ":", "?", "!", ")", "--", "…")):
                    num_words = 0
                    current_sentence["timestamp"].append(round(word["end"], 2))
                    current_sentence["text"] = current_sentence["text"].strip()
                    sentences.append(current_sentence)
                    current_sentence = {"timestamp": None, "text": ""}

                elif num_words > max_num_words_per_page:
                    num_words = 0
                    current_sentence["timestamp"].append(round(word["end"], 2))
                    current_sentence["text"] = current_sentence["text"].strip()
                    sentences.append(current_sentence)
                    current_sentence = {"timestamp": None, "text": ""}
        
        # 处理未结束的句子
        if current_sentence["text"]:
            current_sentence["timestamp"].append(round(word["end"], 2))
            current_sentence["text"] = current_sentence["text"].strip()
            sentences.append(current_sentence)
        
        return sentences

    def process_audio(self, audio, 
                    #   sample_len = 300,
                      max_num_words_per_page = 24,
                      logprob_threshold = -1.0, 
                      no_speech_threshold = 0.1,
                      word_timestamps = False,
                    #   hallucination_silence_threshold = 0.1,
                    #   task, 
                      initial_prompt="", 
                      unload_model=False, 
                      timestamp=False
                      ):
        global CACHE_MODEL
        if CACHE_MODEL is None:
            CACHE_MODEL = whisper.load_model(f"{whisper_model_id}/large-v3-turbo.pt").to(self.device)

        waveform, sample_rate = audio["waveform"].squeeze(0), audio["sample_rate"]
                
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
                
        audio = cache_audio_tensor(cache_dir, waveform, sample_rate)

        if not timestamp:
            word_timestamps = False
        
        result = CACHE_MODEL.transcribe(audio, 
                                # sample_len = sample_len,
                                # hallucination_silence_threshold = hallucination_silence_threshold,
                                logprob_threshold = logprob_threshold,
                                no_speech_threshold = no_speech_threshold,
                                initial_prompt=initial_prompt, 
                                word_timestamps = word_timestamps,
                                # task=task, 
                                fp16=False if self.device == "cpu" else True
                                )

        if unload_model:
            import gc
            CACHE_MODEL = None
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
                return (str(timestamped_segments), convert_subtitle_format(timestamped_segments))
            else:
                data = self.split_into_sentences(result["segments"], max_num_words_per_page)
                return (str(data), convert_subtitle_format(data))

