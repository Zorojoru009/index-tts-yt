import html
import json
import os
import sys
import threading
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--accel", action="store_true", default=False, help="Use acceleration engine for GPT (requires flash_attn)")
parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile for optimization")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import torch
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from indextts.infer_v2 import IndexTTS2
from indextts.utils.validation import get_validator
from tools.i18n.i18n import I18nAuto
import numpy as np
import torchaudio
import torchaudio
import scipy.io.wavfile
import soundfile as sf

class MultiGPUManager:
    def __init__(self, cmd_args):
        self.cmd_args = cmd_args
        self.models = {}  # {device_id: model_instance}
        # Initialize default model on GPU 0 (or default device)
        self.default_device_id = 0 if torch.cuda.is_available() else -1
        
        # Determine available devices
        self.available_gpus = []
        if torch.cuda.is_available():
            self.available_gpus = [i for i in range(torch.cuda.device_count())]
        
        # We start with just the default model loaded (for backward compatibility)
        # The existing global 'tts' will be treated as model 0
        
    def get_model(self, device_id):
        if device_id not in self.models:
            print(f">> Loading model on GPU {device_id}...")
            # Create new instance for this device
            device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"
            model = IndexTTS2(
                model_dir=self.cmd_args.model_dir,
                cfg_path=os.path.join(self.cmd_args.model_dir, "config.yaml"),
                use_fp16=self.cmd_args.fp16,
                use_deepspeed=self.cmd_args.deepspeed,
                use_cuda_kernel=self.cmd_args.cuda_kernel,
                use_accel=self.cmd_args.accel,
                use_torch_compile=self.cmd_args.compile,
                device=device_str
            )
            self.models[device_id] = model
        return self.models[device_id]

    def preload_all(self, selected_gpu_ids, prompt_audio, emo_upload, emo_control_method, emo_weight, vec, emo_text, progress):
        """Preload models and voice on ALL selected GPUs"""
        print(f">> preload_all called with GPU IDs: {selected_gpu_ids}")
        total_steps = len(selected_gpu_ids)
        for i, gpu_id in enumerate(selected_gpu_ids):
            print(f">> Preloading on GPU {gpu_id} ({i+1}/{total_steps})")
            progress((i / total_steps), desc=f"Loading on GPU {gpu_id}...")
            model = self.get_model(gpu_id)
            
            # Run a dummy inference to trigger caching on this specific model instance
            # We must duplicate the logic from preload_voice here because each model instance
            # has its own independent cache!
            
            # Determine emotion vector logic (duplicated for safety)
            emo_vector = vec
            if emo_control_method == 3 and emo_text: # text
                 # Re-use the master model's qwen_emo (it's CPU based usually or shared)
                 # actually each model has its own qwen_emo, but we computed vec passed in.
                 pass

            # Determine emotion audio prompt based on mode
            if emo_control_method == 0:  # speaker emotion
                emo_audio_prompt = None
            elif emo_control_method == 1:  # reference audio emotion
                emo_audio_prompt = emo_upload
            elif emo_control_method == 2:  # vector emotion (no audio needed)
                emo_audio_prompt = None
            else:  # text emotion (mode 3)
                emo_audio_prompt = None

            dummy_text = "test"
            print(f"[DEBUG] Preloading GPU {gpu_id}: prompt={prompt_audio}, emo_audio={emo_audio_prompt}, emo_vec={emo_vector}")
            model.infer(
                spk_audio_prompt=prompt_audio,
                text=dummy_text,
                output_path=None,
                emo_audio_prompt=emo_audio_prompt,
                emo_alpha=emo_weight,
                emo_vector=emo_vector,
                verbose=False,
                max_text_tokens_per_segment=20,
                stream_return=False,
                do_sample=True,
                max_mel_tokens=100
            )

    def infer_distributed(self, selected_gpu_ids, text, **kwargs):
        """
        Generator that yields chunks in order, but processes them in parallel across GPUs.
        """
        # 1. Use primary model to tokenize/segment
        # We assume model 0 is always available or use the first selected one
        primary_gpu = selected_gpu_ids[0] if selected_gpu_ids else self.default_device_id
        primary_model = self.get_model(primary_gpu)
        
        text_tokens_list = primary_model.tokenizer.tokenize(text)
        max_tokens = kwargs.get('max_text_tokens_per_segment', 120)
        segments = primary_model.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_tokens))
        
        # 2. Define worker function
        def process_segment(gpu_id, segment, seg_idx):
            model = self.get_model(gpu_id)
            # We must convert segment (list of tokens) back to string or pass tokens directly?
            # infer_generator expects 'text' string usually, but helper methods might use tokens.
            # actually IndexTTS2.infer_generator takes 'text' and re-tokenizes.
            # To avoid re-tokenizing overhead/mismatch, we should rebuild string from tokens.
            # But IndexTTS2 tokenizer is simple. Let's just pass the string segment.
            seg_text = "".join(segment)
            
            # We need to call a version of infer that handles a SINGLE segment.
            # infer_generator normally splits text. We want to force it to treat this as 1 segment.
            # We can pass max_text_tokens_per_segment=VeryLarge to prevent splitting.
            
            # Update kwargs for this specific call
            local_kwargs = kwargs.copy()
            local_kwargs['max_text_tokens_per_segment'] = 999999 # Force single chunk
            
            # Call generator but consume it immediately to get the tensor
            # We expect exactly one chunk (plus silence maybe)
            gen = model.infer_generator(text=seg_text, **local_kwargs)
            
            results = []
            for item in gen:
                results.append(item)
            return results

        # 3. Schedule Tasks
        futures = []
        with ThreadPoolExecutor(max_workers=len(selected_gpu_ids)) as executor:
            for i, seg in enumerate(segments):
                # Round robin assignment
                gpu_id = selected_gpu_ids[i % len(selected_gpu_ids)]
                futures.append(executor.submit(process_segment, gpu_id, seg, i))
            
            # 4. Yield Results in Order
            # We iterate futures in order (which matches segment order)
            for i, future in enumerate(futures):
                try:
                    results = future.result()
                    # yield all items from this chunk (audio tensors)
                    for res in results:
                        yield res
                except Exception as e:
                    print(f"Error in distributed chunk {i}: {e}")
                    # We might want to yield an error or silence to keep stream alive?
                    pass

if cmd_args.accel:
    try:
        import flash_attn
    except ImportError:
        print("Warning: --accel was specified but 'flash_attn' is not installed. Disabling acceleration.")
        cmd_args.accel = False

i18n = I18nAuto(language="Auto")
MODE = 'local'
# Initialize Multi-GPU Manager
multi_gpu_manager = MultiGPUManager(cmd_args)

# Debug: Print GPU detection info
print(f">> CUDA Available: {torch.cuda.is_available()}")
print(f">> GPU Count: {torch.cuda.device_count()}")
print(f">> Available GPUs: {multi_gpu_manager.available_gpus}")
for i, gpu_id in enumerate(multi_gpu_manager.available_gpus):
    try:
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        print(f"   GPU {gpu_id}: {gpu_name} ({gpu_memory:.2f} GB)")
    except Exception as e:
        print(f"   GPU {gpu_id}: Error getting info - {e}")

# For backward compatibility with existing code that references global `tts`
# We use device 0 as default
if len(multi_gpu_manager.available_gpus) > 0:
    tts = multi_gpu_manager.get_model(multi_gpu_manager.available_gpus[0])
else:
    # CPU case
    tts = multi_gpu_manager.get_model(-1)

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("Same emotion as speaker"),
                i18n("Reference audio emotion"),
                i18n("Emotion vector control"),
                i18n("Text-based emotion control")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)

def gen_single(emo_control_method,prompt, text, emo_ref_path, emo_weight, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text,emo_random, max_text_tokens_per_segment=120, interval_silence=200, *args, progress=gr.Progress()):
    # Returns (sample_rate, audio_data) for use in regeneration
    audio_result = None
    gen = gen_single_streaming(
        None, # selected_gpus (None = default)
        emo_control_method, prompt, text,
        emo_ref_path, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        max_text_tokens_per_segment,
        interval_silence,
        *args, progress=progress
    )
    for item in gen:
        # consume generator to get final audio data
        if isinstance(item, dict) and 'output_audio' in item and item['output_audio'] is not None:
             audio_result = item['output_audio']
    
    # Ensure we return valid data
    if audio_result and isinstance(audio_result, tuple) and len(audio_result) == 2:
        sr, data = audio_result
        if isinstance(data, np.ndarray) and data.size > 0:
            return audio_result
    
    # Fallback: return empty array
    print("[WARNING] gen_single: No valid audio generated")
    return (22050, np.array([], dtype=np.float32))

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('Note: This feature is experimental. Results may be unstable. We are continuously optimizing.'))

def preload_voice(selected_gpus, prompt_audio, emo_upload, emo_control_method, emo_weight,
                  vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text,
                  progress=gr.Progress()):
    """Preload voice embeddings to speed up generation"""
    # Validate audio input
    if not prompt_audio:
        yield "❌ " + i18n("Please upload a voice reference audio first")
        return

    # If prompt_audio is a dict (from Gradio File component), extract the file path
    if isinstance(prompt_audio, dict):
        prompt_audio = prompt_audio.get("name") or prompt_audio.get("path")

    # Validate that file exists
    if not prompt_audio or not os.path.exists(prompt_audio):
        yield "❌ " + i18n("Audio file not found or invalid")
        return

    try:
        yield "⏳ " + i18n("Analyzing voice...")
        progress(0.1, desc="Analyzing voice...")
        
        # Determine emotion control method
        # emo_control_method comes from Gradio Radio as a string value, convert to index
        if isinstance(emo_control_method, str):
            try:
                emo_control_method = EMO_CHOICES_ALL.index(emo_control_method)
            except ValueError:
                emo_control_method = 0  # Default to speaker emotion
        elif not isinstance(emo_control_method, int):
            # Handle other types (shouldn't happen, but be safe)
            emo_control_method = 0
        
        # Set up emotion reference
        emo_ref_path = None
        emo_vector = None
        
        if emo_control_method == 0:  # emotion from speaker
            emo_ref_path = None
        elif emo_control_method == 1:  # emotion from reference audio
            emo_ref_path = emo_upload
        elif emo_control_method == 2:  # emotion from custom vectors
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            emo_vector = tts.normalize_emo_vec(vec, apply_bias=True)
        elif emo_control_method == 3:  # emotion from text
            if emo_text and emo_text.strip():
                yield "⏳ " + i18n("Analyzing emotion text...")
                progress(0.2, desc="Analyzing emotion text...")
                emo_dict = tts.qwen_emo.inference(emo_text)
                emo_vector = list(emo_dict.values())
                print(f"[DEBUG] Extracted emotion from text: {emo_dict}")
        
        print(f"[DEBUG] Preload params: emo_method={emo_control_method}, emo_ref={emo_ref_path}, emo_vec={emo_vector}")
        
        yield "⏳ " + i18n("Caching embeddings on selected GPUs...")
        progress(0.4, desc="Caching embeddings...")
        
        # Handle GPU selection
        print(f">> DEBUG: selected_gpus input = {selected_gpus}")
        print(f">> DEBUG: selected_gpus type = {type(selected_gpus)}")

        target_gpus = []
        if selected_gpus:
            for g in selected_gpus:
                print(f">> DEBUG: Processing GPU selection: {g}")
                if "CPU" in g:
                    target_gpus.append(-1)  # CPU device ID
                else:
                    try:
                        gpu_id = int(g.split(":")[0].replace("GPU ", ""))
                        target_gpus.append(gpu_id)
                    except (ValueError, IndexError):
                        target_gpus.append(multi_gpu_manager.default_device_id)
        else:
            target_gpus = [multi_gpu_manager.default_device_id]

        if not target_gpus: target_gpus = [0] # Fallback
        print(f">> DEBUG: Final target_gpus = {target_gpus}")
        
        multi_gpu_manager.preload_all(
            target_gpus,
            prompt_audio=prompt_audio,
            emo_upload=emo_upload,
            emo_control_method=emo_control_method,
            emo_weight=emo_weight,
            vec=emo_vector,
            emo_text=emo_text,
            progress=progress
        )
        
        progress(1.0, desc="Done!")
        yield "✅ " + i18n("Voice preloaded on: ") + f"{target_gpus}"
    except Exception as e:
        print(f"Preload error: {e}")
        yield f"❌ " + i18n("Preload failed") + f": {str(e)}"

def gen_single_streaming(selected_gpus, emo_control_method, prompt, text,
                        emo_ref_path, emo_weight,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                        emo_text, emo_random,
                        max_text_tokens_per_segment=120,
                        interval_silence=200,
                        *args, progress=gr.Progress()):
    """Streaming generation with real-time progress updates"""
    # Create outputs directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = None
    if not output_path:
        output_path = os.path.join(output_dir, f"spk_{int(time.time())}.wav")
    
    # set gradio progress
    tts.gr_progress = progress
    
    # CRITICAL: Upack args to match advanced_params from UI
    # UI sends: [do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens]
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "max_mel_tokens": int(max_mel_tokens),
        "interval_silence": int(interval_silence)
    }
    
    # DEBUG: Log generation parameters with high visibility
    print("\n" + "="*50)
    print(" 🔥 [DEBUG] GENERATION PARAMETERS RECEIVED 🔥")
    print(f" - do_sample: {kwargs['do_sample']}")
    print(f" - temperature: {kwargs['temperature']}")
    print(f" - top_p: {kwargs['top_p']}")
    print(f" - top_k: {kwargs['top_k']}")
    print(f" - repetition_penalty: {kwargs['repetition_penalty']}")
    print(f" - interval_silence: {kwargs['interval_silence']}ms")
    print("="*50 + "\n")


    # Convert emo_control_method from string (Gradio Radio value) to index
    if isinstance(emo_control_method, str):
        try:
            emo_control_method = EMO_CHOICES_ALL.index(emo_control_method)
        except ValueError:
            emo_control_method = 0  # Default to speaker emotion
    elif not isinstance(emo_control_method, int):
        emo_control_method = 0

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    # Handle emotion control vectors
    vec = None
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
        vec = None
    elif emo_control_method == 1:  # emotion from reference audio
        vec = None
    elif emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    elif emo_control_method == 3:  # emotion from text
        if emo_text and emo_text.strip():
            emo_dict = tts.qwen_emo.inference(emo_text)
            vec = list(emo_dict.values())
            # CRITICAL: Always normalize/bias even if coming from text
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
            print(f"[DEBUG] Normalized Text Emo Vector: {vec}")

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    print(f"[DEBUG] Generation params: prompt={prompt}, emo_ref_path={emo_ref_path}")
    
    # Calculate segments for progress tracking using primary model
    text_tokens_list = tts.tokenizer.tokenize(text)
    segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
    total_segments = len(segments)
    
    # Determine target GPUs
    target_gpus = []
    if selected_gpus:
        for g in selected_gpus:
            if "CPU" in g:
                target_gpus.append(-1)  # CPU device ID
            else:
                try:
                    gpu_id = int(g.split(":")[0].replace("GPU ", ""))
                    target_gpus.append(gpu_id)
                except (ValueError, IndexError):
                    target_gpus.append(multi_gpu_manager.default_device_id)
    else:
        target_gpus = [multi_gpu_manager.default_device_id]

    if not target_gpus: target_gpus = [0]
    
    # Initialize log
    log_lines = []
    log_lines.append(f"🎙️ Starting generation...")
    log_lines.append(f"🖥️ Using GPUs: {target_gpus}")
    log_lines.append(f"📊 Total segments: {total_segments}")
    log_lines.append(f"📝 Text length: {len(text)} characters, {len(text_tokens_list)} tokens")
    log_lines.append(f"⚙️ Chunk size: {max_text_tokens_per_segment} tokens/segment")
    log_lines.append("-" * 50)
    
    # Show initial log
    yield {
        streaming_log: gr.update(value="\n".join(log_lines), visible=True),
        output_audio: None,
        download_file: None
    }
    
    # Start streaming generation
    start_time = time.time()
    chunk_times = []
    
    try:
        # Use distributed inference if multiple GPUs selected
        if len(target_gpus) > 1:
            # For distributed mode, don't pass output_path to avoid race conditions
            # We'll save the concatenated file ourselves after all chunks are done
            print(f">> Using distributed inference on GPUs: {target_gpus}")
            generator = multi_gpu_manager.infer_distributed(
                target_gpus,
                text=text,
                spk_audio_prompt=prompt,
                output_path=None,  # Don't save in distributed mode - we'll concatenate and save after
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method==3),
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                stream_return=True,
                **kwargs
            )
        else:
            # Single GPU - can use model's internal saving
            print(f">> Using single GPU inference on GPU: {target_gpus[0]}")
            model = multi_gpu_manager.get_model(target_gpus[0])
            generator = model.infer_generator(
                spk_audio_prompt=prompt,
                text=text,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method==3),
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                stream_return=True,
                **kwargs
            )
        
        chunk_idx = 0
        all_audio_chunks = []  # Accumulate all chunks for final concatenation
        
        # Phase 1: Chunk Data Accumulator
        chunk_data_accumulator = [] 
        chunks_dir = os.path.join(output_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Phase 2: Initialize Validator (Lazy)
        validator = None
        try:
             validator = get_validator()
        except:
             pass

        for item in generator:
            if isinstance(item, torch.Tensor):
                # Check silence
                is_silence = torch.all(item == 0)
                chunk_end_time = time.time()
                audio_duration = item.shape[-1] / 22050 # Assume 24k

                if not is_silence:
                    chunk_idx += 1
                    chunk_duration = chunk_end_time - start_time if chunk_idx == 1 else chunk_end_time - chunk_times[-1]
                    chunk_times.append(chunk_end_time)

                    rtf = chunk_duration / audio_duration if audio_duration > 0 else 0

                    raw_text = ''.join(segments[chunk_idx-1]) if chunk_idx <= len(segments) else "End"
                    segment_text = raw_text.replace(' ', ' ').replace(' ', ' ')

                    log_lines.append(f"✅ Chunk {chunk_idx}/{total_segments} completed in {chunk_duration:.2f}s")
                    log_lines.append(f"   📊 RTF: {rtf:.4f} | Audio: {audio_duration:.2f}s")
                    log_lines.append(f"   📝 Text: {segment_text[:50]}...")
                    
                    # Save individual chunk
                    chunk_filename = f"chunk_{int(time.time())}_{chunk_idx}.wav"
                    chunk_filepath = os.path.join(chunks_dir, chunk_filename)
                    # Save individual chunk using soundfile with safe float->int16 conversion
                    chunk_np = item.detach().cpu().numpy().flatten()
                    
                    # DEBUG: Log final sampling params reaching the model with high visibility
                    print(f" 🔍 [MODEL-SAMPLING] do_sample={do_sample}, temp={temperature}, top_p={top_p}, top_k={top_k}")
                    print(f" 🔍 [MODEL-SAMPLING] repetition_penalty={repetition_penalty}, max_mel_tokens={max_mel_tokens}")
                    print(f"  - Numpy shape: {chunk_np.shape}, dtype: {chunk_np.dtype}")
                    print(f"  - Value range: [{chunk_np.min():.4f}, {chunk_np.max():.4f}]")
                    
                    # CRITICAL FIX: Model outputs int16-scale floats, normalize to -1.0..1.0
                    chunk_np_normalized = chunk_np / 32767.0
                    print(f"  - After normalization: [{chunk_np_normalized.min():.4f}, {chunk_np_normalized.max():.4f}]")
                    
                    # Ensure Float32/64 is safely clipped and saved as standard PCM_16
                    sf.write(chunk_filepath, chunk_np_normalized, 22050, subtype='PCM_16')
                    
                    # DEBUG: Verify saved file
                    test_load, test_sr = sf.read(chunk_filepath)
                    print(f"  - After save/load: shape={test_load.shape}, dtype={test_load.dtype}, sr={test_sr}")
                    print(f"  - After save/load range: [{test_load.min():.4f}, {test_load.max():.4f}]")
                    
                    # Phase 2: Validate
                    val_score = 0.0
                    status_text = "Generated"
                    transcription = ""
                    
                    print(f"[DEBUG] Starting validation for chunk {chunk_idx}...")
                    print(f"  - Segment text: {segment_text[:50]}...")
                    print(f"  - Audio file: {chunk_filepath}")
                    print(f"  - File exists: {os.path.exists(chunk_filepath)}")
                    
                    if validator:
                        try:
                            print(f"  - Validator instance: {validator}")
                            val_score, transcription = validator.validate(segment_text, chunk_filepath)
                            print(f"  - Validation result: score={val_score}, transcription='{transcription[:50] if transcription else 'None'}...'")
                            log_lines.append(f"   🔍 Match: {val_score}%")
                            if val_score >= 90:
                                status_text = "✅ Exact"
                            elif val_score >= 75:
                                status_text = "⚠️ Good"
                            elif val_score > 0:
                                status_text = "❌ Low"
                            else:
                                status_text = "❓ Error"
                        except Exception as e:
                            print(f"  - ❌ Validation exception: {type(e).__name__}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            status_text = "❓ Error"
                            val_score = 0.0
                    else:
                        print(f"  - ⚠️ No validator instance available")
                        status_text = "⚠️ No STT"
                    
                    chunk_info = {
                        "index": chunk_idx,
                        "text": segment_text,
                        "audio_path": chunk_filepath,
                        "status": status_text,
                        "score": val_score
                    }
                    chunk_data_accumulator.append(chunk_info)
                    
                    # Format for Dataframe: [Index, Text, Status, Score]
                    df_data = [[c["index"], c["text"], c["status"], c["score"]] for c in chunk_data_accumulator]

                    # Estimate remaining time
                    if chunk_idx < total_segments:
                        avg_chunk_time = (chunk_end_time - start_time) / chunk_idx
                        remaining_chunks = total_segments - chunk_idx
                        eta = avg_chunk_time * remaining_chunks
                        log_lines.append(f"   ⏳ ETA: {eta:.1f}s")

                    # CRITICAL FIX: Use normalized audio for accumulation
                    all_audio_chunks.append(chunk_np_normalized)
                    
                    # DEBUG: Log accumulation
                    accumulated = np.concatenate(all_audio_chunks)
                    print(f"  - Accumulated audio: shape={accumulated.shape}, range=[{accumulated.min():.4f}, {accumulated.max():.4f}]")

                    yield {
                        streaming_log: gr.update(value="\n".join(log_lines)),
                        # Send accumulated audio (Gradio Audio streaming=True REPLACES, doesn't append)
                        output_audio: (22050, np.concatenate(all_audio_chunks)),
                        download_file: None,
                        chunk_state: chunk_data_accumulator,
                        chunk_list: gr.update(value=df_data)
                    }
                else:
                    # Silence chunk.. handle if needed
                    pass
            elif isinstance(item, dict):
                 # Handle status updates from generator if any
                 pass
        if len(all_audio_chunks) > 0:
            final_audio = np.concatenate(all_audio_chunks)
            # Use soundfile for consistent, safe saving
            sf.write(output_path, final_audio, 22050, subtype='PCM_16')
            
            log_lines.append(f"✅ Generation Complete! Saved to {output_path}")
            # CRITICAL FIX: Don't re-yield audio (already sent during streaming)
            # Only update log and download file to avoid duplication
            yield {
                streaming_log: gr.update(value="\n".join(log_lines)),
                output_audio: gr.update(),  # Don't update (keeps last streamed audio)
                download_file: output_path,
                chunk_state: chunk_data_accumulator,
                chunk_list: gr.update(value=df_data)
            }

    except Exception as e:
        log_lines.append("=" * 50)
        log_lines.append(f"❌ Error: {str(e)}")
        log_lines.append("=" * 50)
        yield {
            streaming_log: gr.update(value="\n".join(log_lines)),
            output_audio: None,
            download_file: None
        }

def on_select_chunk(evt: gr.SelectData, chunk_state):
    if not chunk_state:
        return "", None, -1
    
    # evt.index is [row, col]
    row_idx = evt.index[0]
    if row_idx < 0 or row_idx >= len(chunk_state):
        return "", None, -1
        
    chunk = chunk_state[row_idx]
    # Return text, audio_path, index
    return chunk["text"], chunk["audio_path"], chunk["index"]

def merge_chunks(chunk_state):
    """Concatenate all chunks in the state into one file"""
    if not chunk_state:
        return "⚠️ No chunks to merge", None, None
    
    try:
        # Sort by index just in case
        sorted_chunks = sorted(chunk_state, key=lambda x: x["index"])
        all_data = []
        
        for chunk in sorted_chunks:
            path = chunk.get("audio_path")
            if path and os.path.exists(path):
                data, sr = sf.read(path)
                all_data.append(data)
            else:
                print(f"Details: Missing chunk path {path}")
        
        if not all_data:
            return "❌ No valid audio chunks found", None, None
            
        final_audio = np.concatenate(all_data)
        
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"merged_{int(time.time())}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Enforce PCM_16 for merged file compatibility
        sf.write(output_path, final_audio, 22050, subtype='PCM_16')
        
        status_msg = f"✅ Merged {len(all_data)} chunks! Saved to: {output_path}"
        print(status_msg)
        
        # Return status, audio tuple for preview, and file path for download
        return status_msg, (22050, final_audio), output_path
        
    except Exception as e:
        error_msg = f"❌ Merge error: {str(e)}"
        print(error_msg)
        return error_msg, None, None

def regenerate_chunk_handler(chunk_idx, new_text, chunk_state, 
                           emo_control_method, prompt, 
                           emo_ref_path, emo_weight,
                           vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                           emo_text, emo_random,
                           max_text_tokens_per_segment,
                           interval_silence,
                           *args): # args = advanced_params
    
    if chunk_idx is None or chunk_idx < 0 or not chunk_state:
        return chunk_state, gr.update(), None

    # Call gen_single with NEW text
    # Note: gen_single signature must match passed args.
    # We pass 'new_text' as the 'text' argument.
    # prompt is 'prompt_audio'
    
    try:
        print(f"🔄 Regenerating chunk {chunk_idx}: {new_text[:20]}...")
        result = gen_single(
            emo_control_method, prompt, new_text,
            emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment,
            interval_silence,
            *args
        )
        # result is (sr, audio_data)
        sr, audio_data = result
        
        # Find chunk
        target_chunk = None
        target_idx = -1
        chunk_idx = int(chunk_idx)
        for i, c in enumerate(chunk_state):
            if c["index"] == chunk_idx:
                target_chunk = c
                target_idx = i
                break
        
        if not target_chunk:
            print("Chunk not found in state")
            return chunk_state, gr.update(), None

        path = target_chunk["audio_path"]
        
        # CRITICAL: Safety check for empty audio
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            print("⚠️ Regeneration produced empty audio")
            return chunk_state, gr.update(), None
        
        # CRITICAL: Normalize audio like in main generation
        # Audio from gen_single should already be normalized, but verify
        audio_normalized = audio_data / 32767.0 if np.abs(audio_data).max() > 1.0 else audio_data
        
        # Save audio (Overwrite) using soundfile (safe PCM_16)
        sf.write(path, audio_normalized, sr, subtype='PCM_16')
        
        # Re-Validate
        score = 0
        status = "Regenerated"
        try:
            validator = get_validator()
            if validator:
                 score, _ = validator.validate(new_text, path)
                 if score >= 90: status = "✅ Regen"
                 elif score >= 75: status = "⚠️ Regen"
                 else: status = "❌ Regen"
        except: pass
        
        # Update State
        target_chunk["text"] = new_text
        target_chunk["status"] = status
        target_chunk["score"] = score
        chunk_state[target_idx] = target_chunk
        
        # Update Dataframe
        # Headers: ["Index", "Text Segment", "Status", "Score"]
        df_data = [[c["index"], c["text"], c["status"], c.get("score", 0)] for c in chunk_state]
        
        # CRITICAL FIX: Return (sr, audio_data) tuple for Gradio Audio component
        return chunk_state, gr.update(value=df_data), (sr, audio_normalized)

    except Exception as e:
        print(f"Regeneration error: {e}")
        import traceback
        traceback.print_exc()
        return chunk_state, gr.update(), None




def gen_wrapper(streaming_mode, selected_gpus, emo_control_method, prompt, text,
                emo_ref_path, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment,
                interval_silence, # Added interval_silence here
                *args,
                progress=gr.Progress()):
    """Wrapper that switches between streaming and non-streaming modes"""
    if streaming_mode:
        # Use streaming mode
        yield from gen_single_streaming(
            selected_gpus,
            emo_control_method, prompt, text,
            emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment,
            interval_silence,
            *args, progress=progress
        )
    else:
        # Use non-streaming mode
        result = gen_single(
            emo_control_method, prompt, text,
            emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment,
            interval_silence,
            *args, progress=progress
        )
        yield {
            streaming_log: gr.update(value="", visible=False),
            output_audio: result,
            download_file: result,  # In batch mode, result is the file path
            chunk_state: [],
            chunk_list: gr.update(value=None)
        }

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("Audio Generation")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("Voice Reference Audio"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("Text"),key="input_text_single", placeholder=i18n("Enter target text here"), info=f"{i18n('Current model version')}{tts.model_version or '1.0'}")
                with gr.Row():
                    preload_voice_btn = gr.Button("🔄 " + i18n("Preload Voice"), scale=1, variant="secondary")
                    gen_button = gr.Button(i18n("Generate Speech"), key="gen_button",interactive=True, scale=2, variant="primary")
                voice_status = gr.Markdown("⏳ " + i18n("Voice not preloaded"))
            
            with gr.Column():

                output_audio = gr.Audio(label=i18n("Generation Result"), visible=True,key="output_audio", streaming=True)
                download_file = gr.File(label=i18n("Download Audio"), visible=True)
        
        # Review Panel (Phase 1 MVP)
        with gr.Row():
            with gr.Accordion(i18n("Review & Edit Chunks"), open=True, visible=True) as review_accordion:
                chunk_state = gr.State([]) # Stores list of dicts: [{'index', 'text', 'audio_path', 'status'}]
                with gr.Row():
                    with gr.Column(scale=3):
                        chunk_list = gr.Dataframe(
                            headers=["Index", "Text Segment", "Status", "Score"],
                            datatype=["number", "str", "str", "number"],
                            interactive=False,
                            label=i18n("Chunks List"),
                            max_height=400
                        )
                    with gr.Column(scale=2):
                        selected_chunk_idx = gr.Number(label=i18n("Chunk Index"), visible=False, value=-1)
                        selected_chunk_text = gr.Textbox(label=i18n("Edit Text Segment"), lines=4, interactive=True)
                        selected_chunk_audio = gr.Audio(label=i18n("Chunk Preview"), type="filepath")
                        btn_regen_chunk = gr.Button(i18n("Regenerate This Chunk"), variant="secondary")
                        
                        # Merge Section
                        gr.Markdown("---")
                        merge_status = gr.Textbox(label=i18n("Merge Status"), lines=2, interactive=False, visible=True)
                        merged_audio_preview = gr.Audio(label=i18n("Merged Audio Preview"), type="numpy")
                        btn_merge_all = gr.Button(i18n("Merge All Chunks"), variant="primary")

        with gr.Row():
            streaming_log = gr.Textbox(
                label=i18n("Generation Log"),
                lines=8,
                max_lines=15,
                interactive=False,
                visible=False,
                show_copy_button=True
            )

        with gr.Row():
            experimental_checkbox = gr.Checkbox(label=i18n("Show Experimental Features"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("Enable Custom Terminology Pronunciation"), value=tts.normalizer.enable_glossary)
            streaming_mode_checkbox = gr.Checkbox(
                label=i18n("Enable Streaming"),
                value=True,
                info=i18n("Recommended for long text (5+ min), shows real-time progress and audio playback")
            )
            
            # GPU Selection for Multi-GPU environments
            gpu_choices = [f"GPU {i}" for i in multi_gpu_manager.available_gpus]
            if not gpu_choices: gpu_choices = ["CPU"]

            # Debug GPU selection
            print(f">> GPU Choices: {gpu_choices}")
            print(f">> GPU Selection Visible: {len(gpu_choices) > 1}")

            gpu_selection = gr.CheckboxGroup(
                choices=gpu_choices,
                value=[gpu_choices[0]] if gpu_choices else [],
                label=i18n("Select GPUs"),
                info=i18n("Select multiple GPUs for parallel processing (faster generation, higher VRAM usage)"),
                visible=len(gpu_choices) > 1
            )
            
        with gr.Accordion(i18n("Emotion Control")):
            # Emotion control options section
            with gr.Row():
                emo_control_method = gr.Radio(choices=EMO_CHOICES_ALL, value=EMO_CHOICES_ALL[0], label=i18n("Emotion Control Mode"), interactive=True)
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("Emotion Control Method"),
                    visible=False)  # do not render
                emo_upload = gr.Audio(label=i18n("Emotion Reference Audio"), sources=["upload", "microphone"], type="filepath", visible=False)
                emo_weight = gr.Slider(minimum=0, maximum=1, value=1, label=i18n("Emotion Strength"), visible=False)

            with gr.Row(visible=False) as emo_vec_row:
                 with gr.Column():
                    vec1 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Happy"))
                    vec2 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Angry"))
                    vec3 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Sad"))
                    vec4 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Afraid"))
                 with gr.Column():
                    vec5 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Disgusted"))
                    vec6 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Melancholic"))
                    vec7 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Surprised"))
                    vec8 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("Calm"))

            emo_text = gr.Textbox(label=i18n("Emotion Description"), visible=False, placeholder=i18n("Enter emotion description (e.g., sad, happy, angry...)"))

            emo_random = gr.Checkbox(label=i18n("Random Emotion"), value=False, visible=False)

        with gr.Accordion(i18n("Advanced Settings"), open=False):
             with gr.Row():
                max_text_tokens_per_segment = gr.Slider(minimum=10, maximum=500, value=120, step=1, label=i18n("Segment Length"), info=i18n("Text segmentation length for long text processing. Too long may cause memory overflow"))
                interval_silence = gr.Slider(minimum=0, maximum=2000, value=200, step=50, label=i18n("Interval Silence (ms)"), info=i18n("Silence between segments"))
                
             with gr.Row():
                do_sample = gr.Checkbox(label=i18n("Do Sample"), value=True)
                top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.8, step=0.01, label=i18n("Top P"))
                top_k = gr.Slider(minimum=1, maximum=100, value=30, step=1, label=i18n("Top K"))
                temperature = gr.Slider(minimum=0.01, maximum=2.0, value=0.8, step=0.01, label=i18n("Temperature"))
                
             with gr.Row():
                length_penalty = gr.Slider(minimum=-2.0, maximum=2.0, value=0.0, step=0.1, label=i18n("Length Penalty"))
                num_beams = gr.Slider(minimum=1, maximum=5, value=3, step=1, label=i18n("Num Beams"))
                repetition_penalty = gr.Slider(minimum=1.0, maximum=5.0, value=10.0, step=0.1, label=i18n("Repetition Penalty"))
                max_mel_tokens = gr.Slider(minimum=10, maximum=2000, value=1500, step=1, label=i18n("Max Mel Tokens"))

        # Glossary/Terminology management
        with gr.Accordion(i18n("Custom Terminology Pronunciation"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("Customize pronunciation for specific professional terms"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=i18n("Term"),
                        placeholder="IndexTTS2",
                    )
                    glossary_reading_zh = gr.Textbox(
                        label=i18n("Chinese Pronunciation"),
                        placeholder="Index T-T-S 二",
                    )
                    glossary_reading_en = gr.Textbox(
                        label=i18n("English Pronunciation"),
                        placeholder="Index T-T-S two",
                    )
                    btn_add_term = gr.Button(i18n("Add Term"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(
                        value=format_glossary_markdown()
                    )

        with gr.Accordion(i18n("Advanced Generation Parameters"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 Sampling Settings')}** _{i18n('Parameters affect audio diversity and generation speed. See')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("Whether to perform sampling"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.61, step=0.01)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.91, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=0, step=1, visible=False) # Unexposed/High (0 usually means unrestricted or max)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("Maximum tokens to generate. Too small will truncate audio"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="Not recommended")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("Sentence Segmentation")}** _{i18n("Parameters affect audio quality and generation speed")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("Max tokens per segment"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("Recommended 80~200. Larger = longer segments, smaller = shorter. Too extreme values affect quality"),
                        )
                    with gr.Accordion(i18n("Segment Preview"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("Index"), i18n("Segment Content"), i18n("Tokens")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
        # binds tightly to UI and always restores the initial state of all components,
        # such as the list of available choices in emo_control_method.
        example_table = gr.Dataset(label="Examples",
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            # these components are NOT "connected". it just reads the column labels/available
            # states from them, so we MUST link to the "all options" versions of all components,
            # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
            components=[prompt_audio,
                        emo_control_method_all,  # important: support all mode labels!
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    # click() event works on both desktop and mobile UI
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio,
                                 emo_control_method,
                                 input_text_single,
                                 emo_upload,
                                 emo_weight,
                                 emo_text,
                                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    # 术语词汇表事件处理函数
    def on_add_glossary_term(term, reading_zh, reading_en):
        """添加术语到词汇表并自动保存"""
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()

        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
            
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
            return gr.update()
        

        # 构建读法数据
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en

        # 添加到词汇表
        tts.normalizer.term_glossary[term] = reading

        # 自动保存到文件
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
            print(f"Error details: {e}")
            return gr.update()

        # 更新Markdown表格
        return gr.update(value=format_glossary_markdown())
        
    def update_emo_ui(method_val):
        """
        Update UI visibility based on the selected emotion control method.
        metrics:
        0: Speaker (default) -> hide all
        1: Ref Audio -> show upload, weight
        2: Vector -> show vec_row, weight
        3: Text -> show text, weight
        """
        # method_val might be index or value depending on gradio version/setup, 
        # but here we used choices list, so it receives the value string.
        # We need to map it back to index or check string.
        
        idx = 0
        if method_val in EMO_CHOICES_ALL:
            idx = EMO_CHOICES_ALL.index(method_val)
        
        # Returns: [emo_upload, emo_weight, emo_vec_row, emo_text, emo_random]
        if idx == 0: # Speaker
            return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
        elif idx == 1: # Ref Audio
            return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
        elif idx == 2: # Vector
            return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]
        elif idx == 3: # Text
            return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
        
        return [gr.update(visible=False)] * 5

    # Old handler removed

    emo_control_method.change(update_emo_ui,
        inputs=[emo_control_method],
        outputs=[emo_upload,
                 emo_weight,
                 emo_vec_row,
                 emo_text,
                 emo_random]
    )

    def on_experimental_change(is_experimental, current_mode_value):
        # Switch emotion control options based on experimental mode
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL

        # Convert the string value to index
        try:
            current_index = EMO_CHOICES_ALL.index(current_mode_value) if current_mode_value in EMO_CHOICES_ALL else 0
        except (ValueError, AttributeError):
            current_index = 0

        # If current selection doesn't exist in new choices, reset to first option
        new_index = current_index if current_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    def on_glossary_checkbox_change(is_enabled):
        """控制术语词汇表的可见性"""
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    def on_demo_load():
        """页面加载时重新加载glossary数据"""
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("加载词汇表时出错"))
            print(f"Failed to reload glossary on page load: {e}")
        return gr.update(value=format_glossary_markdown())

    # 术语词汇表事件绑定
    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table]
    )

    # 页面加载时重新加载glossary
    demo.load(
        on_demo_load,
        inputs=[],
        outputs=[glossary_table]
    )

    # Preload voice button handler
    preload_voice_btn.click(
        preload_voice,
        inputs=[gpu_selection, prompt_audio, emo_upload, emo_control_method, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text],
        outputs=[voice_status]
    )

    # Generate button with streaming
    gen_button.click(
        gen_wrapper,
        inputs=[streaming_mode_checkbox, gpu_selection, emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment,
                interval_silence,
                *advanced_params,
        ],
        outputs=[streaming_log, output_audio, download_file, chunk_state, chunk_list]
    )
    
    btn_merge_all.click(
        fn=merge_chunks,
        inputs=[chunk_state],
        outputs=[merge_status, merged_audio_preview, download_file]
    )
    # Phase 1: Chunk List Event
    chunk_list.select(
        fn=on_select_chunk,
        inputs=[chunk_state],
        outputs=[selected_chunk_text, selected_chunk_audio, selected_chunk_idx]
    )



    btn_regen_chunk.click(
        fn=regenerate_chunk_handler,
        inputs=[
            selected_chunk_idx, selected_chunk_text, chunk_state,
            emo_control_method, prompt_audio, emo_upload, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment,
            interval_silence,
            *advanced_params
        ],
        outputs=[chunk_state, chunk_list, selected_chunk_audio]
    )

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
