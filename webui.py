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

import gradio as gr
import torch
from concurrent.futures import ThreadPoolExecutor
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

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
        total_steps = len(selected_gpu_ids)
        for i, gpu_id in enumerate(selected_gpu_ids):
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

            dummy_text = "test"
            model.infer(
                spk_audio_prompt=prompt_audio,
                text=dummy_text,
                output_path=None,
                emo_audio_prompt=emo_upload if emo_control_method==1 else (None if emo_control_method==0 else prompt_audio),
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
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
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

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    # Legacy wrapper for single generation (uses default device)
    # We can just call gen_single_streaming and consume it
    result = None
    gen = gen_single_streaming(
        None, # selected_gpus (None = default)
        emo_control_method, prompt, text,
        emo_ref_path, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        max_text_tokens_per_segment,
        *args, progress=progress
    )
    for item in gen:
        # consume generator to get final file
        if 'output_audio' in item:
             # Streaming updates...
             pass
        if 'download_file' in item and item['download_file']:
            result = item['download_file']
    return result

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

def preload_voice(selected_gpus, prompt_audio, emo_upload, emo_control_method, emo_weight,
                  vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text, 
                  progress=gr.Progress()):
    """Preload voice embeddings to speed up generation"""
    if not prompt_audio:
        yield "❌ " + i18n("Please upload a voice reference audio first")
        return
    
    try:
        yield "⏳ " + i18n("Analyzing voice...")
        progress(0.1, desc="Analyzing voice...")
        
        # Determine emotion control method
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        
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
        
        yield "⏳ " + i18n("Caching embeddings on selected GPUs...")
        progress(0.4, desc="Caching embeddings...")
        
        # Handle GPU selection
        target_gpus = [int(g.split(":")[0].replace("GPU ", "")) for g in selected_gpus] if selected_gpus else [multi_gpu_manager.default_device_id]
        if not target_gpus: target_gpus = [0] # Fallback
        
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
                        *args, progress=gr.Progress()):
    """Streaming generation with real-time progress updates"""
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    
    # Calculate segments for progress tracking using primary model
    text_tokens_list = tts.tokenizer.tokenize(text)
    segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
    total_segments = len(segments)
    
    # Determine target GPUs
    target_gpus = [int(g.split(":")[0].replace("GPU ", "")) for g in selected_gpus] if selected_gpus else [multi_gpu_manager.default_device_id]
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
            generator = multi_gpu_manager.infer_distributed(
                target_gpus,
                text=text,
                spk_audio_prompt=prompt,
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
        else:
            # Single GPU fallback (legacy path but using manager)
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
        for item in generator:
            if isinstance(item, torch.Tensor):
                # Check silence
                is_silence = torch.all(item == 0)
                chunk_end_time = time.time()
                audio_duration = item.shape[-1] / 22050
                
                if not is_silence:
                    chunk_idx += 1
                    chunk_duration = chunk_end_time - start_time if chunk_idx == 1 else chunk_end_time - chunk_times[-1]
                    chunk_times.append(chunk_end_time)
                    
                    rtf = chunk_duration / audio_duration if audio_duration > 0 else 0
                    
                    raw_text = ''.join(segments[chunk_idx-1]) if chunk_idx <= len(segments) else ""
                    segment_text = raw_text.replace(' ', ' ').replace(' ', ' ')
                    
                    log_lines.append(f"✅ Chunk {chunk_idx}/{total_segments} completed in {chunk_duration:.2f}s")
                    log_lines.append(f"   📊 RTF: {rtf:.4f} | Audio: {audio_duration:.2f}s")
                    log_lines.append(f"   📝 Text: {segment_text[:50]}...")
                    
                    # Estimate remaining time
                    if chunk_idx < total_segments:
                        # Distributed estimation is tricky, simple linear approx
                        avg_chunk_time = (chunk_end_time - start_time) / chunk_idx
                        remaining_time = avg_chunk_time * (total_segments - chunk_idx) / len(target_gpus) # Simple factor
                        log_lines.append(f"   ⏱️ Est. remaining: {remaining_time:.1f}s")
                    
                    log_lines.append("-" * 50)
                
                if len(log_lines) > 100: log_lines = log_lines[-100:]
                
                yield {
                    streaming_log: gr.update(value="\n".join(log_lines)),
                    output_audio: (22050, item.numpy().T) if hasattr(item, 'numpy') else item,
                    download_file: None
                }
        
        # Final summary
        avg_rtf = sum([(chunk_times[i] - (chunk_times[i-1] if i > 0 else start_time)) for i in range(len(chunk_times))]) / ((chunk_times[-1] - start_time) / len(chunk_times)) if chunk_times else 0
        total_time = time.time() - start_time
        log_lines.append("=" * 50)
        log_lines.append(f"🎉 Generation complete!")
        log_lines.append(f"⏱️ Total time: {total_time:.2f}s")
        log_lines.append(f"📊 Average RTF: {avg_rtf:.4f}")
        log_lines.append(f"💾 Saved to: {output_path}")
        log_lines.append("=" * 50)
        
        # NOTE: Do NOT yield output_path to output_audio, as it causes repetition in streaming mode
        # Instead, yield it to download_file component
        yield {
            streaming_log: gr.update(value="\n".join(log_lines)),
            output_audio: None,
            download_file: output_path
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

def gen_wrapper(streaming_mode, selected_gpus, emo_control_method, prompt, text,
                emo_ref_path, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
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
            *args, progress=progress
        )
        yield {
            streaming_log: gr.update(value="", visible=False),
            output_audio: result,
            download_file: result  # In batch mode, result is the file path
        }

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                with gr.Row():
                    preload_voice_btn = gr.Button("🔄 " + i18n("Preload Voice"), scale=1, variant="secondary")
                    gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True, scale=2, variant="primary")
                voice_status = gr.Markdown("⏳ " + i18n("Voice not preloaded"))
            with gr.Column():
                output_audio = gr.Audio(label=i18n("生成结果"), visible=True,key="output_audio", streaming=True)
                download_file = gr.File(label=i18n("Download Audio"), visible=True)
        
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
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
            streaming_mode_checkbox = gr.Checkbox(
                label=i18n("Enable Streaming"), 
                value=True,
                info=i18n("Recommended for long text (5+ min), shows real-time progress and audio playback")
            )
            
            # GPU Selection for Multi-GPU environments
            gpu_choices = [f"GPU {i}" for i in multi_gpu_manager.available_gpus]
            if not gpu_choices: gpu_choices = ["CPU"]
            
            gpu_selection = gr.CheckboxGroup(
                choices=gpu_choices,
                value=[gpu_choices[0]] if gpu_choices else [],
                label=i18n("Select GPUs"),
                info=i18n("Select multiple GPUs for parallel processing (faster generation, higher VRAM usage)"),
                visible=len(gpu_choices) > 1
            )
            
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(choices=EMO_CHOICES_ALL, value=EMO_CHOICES_ALL[0], label=i18n("情感控制模式"), interactive=True)
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
                emo_upload = gr.Audio(label=i18n("情感参考音频"), sources=["upload", "microphone"], type="filepath", visible=False)
                emo_weight = gr.Slider(minimum=0, maximum=1, value=1, label=i18n("情感强度"), visible=False)
            
            with gr.Row(visible=False) as emo_vec_row:
                 with gr.Column():
                    vec1 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 1"))
                    vec2 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 2"))
                    vec3 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 3"))
                    vec4 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 4"))
                 with gr.Column():
                    vec5 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 5"))
                    vec6 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 6"))
                    vec7 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 7"))
                    vec8 = gr.Slider(minimum=-5, maximum=5, value=0, label=i18n("情感向量 8"))
            
            emo_text = gr.Textbox(label=i18n("情感描述文本"), visible=False, placeholder=i18n("请输入情感描述，例如：悲伤、开心、愤怒..."))
            
            emo_random = gr.Checkbox(label=i18n("随机情感"), value=False, visible=False)

        with gr.Accordion(i18n("高级设置"), open=False):
             with gr.Row():
                max_text_tokens_per_segment = gr.Slider(minimum=10, maximum=500, value=120, step=1, label=i18n("分段长度"), info=i18n("长文本分段处理的长度，过长可能会导致显存溢出"))
                
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

        # 术语词汇表管理
        with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("自定义个别专业术语的读音"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=i18n("术语"),
                        placeholder="IndexTTS2",
                    )
                    glossary_reading_zh = gr.Textbox(
                        label=i18n("中文读法"),
                        placeholder="Index T-T-S 二",
                    )
                    glossary_reading_en = gr.Textbox(
                        label=i18n("英文读法"),
                        placeholder="Index T-T-S two",
                    )
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(
                        value=format_glossary_markdown()
                    )

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
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
        

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

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
        inputs=[prompt_audio, emo_upload, emo_control_method, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text],
        outputs=[voice_status]
    )

    # Generate button with streaming
    gen_button.click(
        gen_wrapper,
        inputs=[streaming_mode_checkbox, emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment,
                *advanced_params,
        ],
        outputs=[streaming_log, output_audio, download_file]
    )



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
