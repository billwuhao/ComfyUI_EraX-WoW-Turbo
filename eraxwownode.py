import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import folder_paths
import os


LANGUAGES = {
    'vietnamese': 'vi',
    'english': 'en',
    'chinese': 'zh',
    'german': 'de',
    'russian': 'ru',
    'korean': 'ko',
    'french': 'fr',
    'japanese': 'ja',
    'dutch': 'nl',
    'hindi': 'hi',
    'ukrainian': 'uk'
}


models_dir = folder_paths.models_dir
model_id = os.path.join(models_dir, "TTS", "EraX-WoW-Turbo-V1.0")


PROCESSOR = None
CACHE_MODEL = None
class EraXWoWRUN:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "language": (list(LANGUAGES.keys()), {"default": "chinese"}),
                "max_length": ("INT", {"default": 200, "min": 1,}),
                "num_beams": ("INT", {"default": 1, "min": 1,}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "🎤MW/MW-EraXWoW"
    
    def transcribe(self, audio, language, num_beams, max_length, unload_model):
        global PROCESSOR, CACHE_MODEL
        if CACHE_MODEL is None:
            PROCESSOR = WhisperProcessor.from_pretrained(model_id)
            CACHE_MODEL = WhisperForConditionalGeneration.from_pretrained(model_id)
            CACHE_MODEL.to(self.device).eval()
        
        model = CACHE_MODEL
        processor = PROCESSOR

        waveform, sample_rate = audio["waveform"], audio["sample_rate"]
        waveform = waveform.squeeze(0)
        waveform = waveform.to(self.device)
                
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(self.device)
            waveform = resampler(waveform)
                
        if len(waveform.shape) > 1:
            waveform = torch.mean(waveform, dim=0)

        inputs = processor(waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        processor.feature_extractor.language = LANGUAGES[language]
        processor.feature_extractor.task = "transcribe"
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGES[language], task="transcribe")
                
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_length=max_length,
                num_beams=num_beams,
                forced_decoder_ids=forced_decoder_ids,
                return_dict_in_generate=True,
            ).sequences

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        if unload_model:
            import gc
            del model
            del processor
            PROCESSOR = None
            CACHE_MODEL = None
            gc.collect()
            torch.cuda.empty_cache()

        return (transcription,)



from .whisper_turbo_node import WhisperTurboRun

NODE_CLASS_MAPPINGS = {
    "EraXWoWRUN": EraXWoWRUN,
    "WhisperTurboRun": WhisperTurboRun
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EraXWoWRUN": "EraX WoW Run",
    "WhisperTurboRun": "Whisper Turbo Run"
}