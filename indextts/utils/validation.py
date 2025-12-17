import warnings
import numpy as np
import os

# Try importing dependencies, but don't crash if missing (lazy load check)
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import Levenshtein
    DEPENDENCIES_INSTALLED = True
except ImportError as e:
    DEPENDENCIES_INSTALLED = False
    MISSING_DEP_ERROR = str(e)

class AudioValidator:
    def __init__(self, device='cpu'):
        if not DEPENDENCIES_INSTALLED:
            raise ImportError(f"Validation dependencies missing: {MISSING_DEP_ERROR}. Please install 'funasr', 'modelscope', 'python-Levenshtein'")
        
        print(f"✅ Initializing STT Validator on {device.upper()}...")
        # Suppress ModelScope warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use Paraformer-Large (highly accurate, fast)
            # This model is small enough for CPU (< 200MB)
            self.pipeline = pipeline(
                task=Tasks.auto_speech_recognition,
                model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                device=device
            )
        print("✅ STT Validator Ready")

    def validate(self, reference_text, audio_input):
        """
        Validates audio against text.
        audio_input: path to wav file OR numpy array
        reference_text: original text
        Returns: (score [0-100], transcript)
        """
        try:
            # ModelScope pipeline handles filepath strings natively. 
            # If numpy, might need handling, but we save chunks to disk anyway.
            result = self.pipeline(audio_input)
            
            if 'text' in result:
                transcript = result['text']
            else:
                transcript = ""
            
            # Normalize for comparison (remove punctuation, lower case)
            # Simple normalization
            ref_norm = self._normalize(reference_text)
            trans_norm = self._normalize(transcript)
            
            if not ref_norm and not trans_norm:
                return 100.0, transcript # Both empty
            
            # Levenshtein Ratio: 0 to 1
            ratio = Levenshtein.ratio(ref_norm, trans_norm)
            score = round(ratio * 100, 1)
            
            return score, transcript
            
        except Exception as e:
            print(f"❌ Validation Error: {e}")
            return 0.0, f"[Error] {str(e)}"

    def _normalize(self, text):
        import re
        # Remove punctuation and newlines
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip().lower()

# Global instance
validator_instance = None

def get_validator():
    global validator_instance
    if validator_instance is None:
        try:
            validator_instance = AudioValidator(device='cpu') # Force CPU
        except Exception as e:
            print(f"⚠️ Could not load Validator: {e}")
            return None
    return validator_instance
