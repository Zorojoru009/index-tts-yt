# Multi-GPU Support in IndexTTS

## Overview

IndexTTS now supports parallel audio generation across multiple GPUs, significantly speeding up inference for long-form content. The implementation includes voice preloading, streaming generation, and intelligent GPU distribution.

## Architecture

### Core Components

#### 1. **MultiGPUManager** (lines 58-183)
Manages all GPU instances and distributes inference workload.

```python
class MultiGPUManager:
    def __init__(self, cmd_args):
        self.models = {}              # {device_id: model_instance}
        self.default_device_id = 0 if torch.cuda.is_available() else -1
        self.available_gpus = [0, 1, 2, ...]  # List of available GPU IDs
```

**Key Methods:**
- `get_model(device_id)`: Returns or creates a model instance for a specific GPU
- `preload_all(gpu_ids, audio, emotion_params, ...)`: Preloads voice embeddings on all selected GPUs
- `infer_distributed(gpu_ids, text, ...)`: Distributes inference across GPUs in order

---

## Features

### 1. Voice Preloading

**Purpose:** Cache speaker embeddings on GPU memory to speed up generation.

**Function:** `preload_voice()` (lines 316-372)

**Process:**
1. User uploads audio reference
2. Click "Preload Voice" button
3. System analyzes voice on selected GPUs
4. Generates speaker embeddings cached in VRAM

**Supports All Emotion Control Modes:**
- Mode 0: Speaker emotion (no extra audio needed)
- Mode 1: Reference audio emotion (uses uploaded emotion audio)
- Mode 2: Vector-based emotion (uses manual sliders)
- Mode 3: Text-based emotion (uses emotion description)

**Example:**
```python
preload_voice(
    selected_gpus=['GPU 0', 'GPU 1'],
    prompt_audio='speaker.wav',
    emo_control_method=1,  # Reference audio
    emo_upload='emotion.wav',
    emo_weight=0.7
)
```

---

### 2. GPU Selection UI

**Component:** `gpu_selection` (lines 690-696)

A CheckboxGroup that dynamically shows available GPUs:
- **Single GPU systems**: Hidden (only one choice)
- **Multi-GPU systems**: Shows all GPUs with selection options
- **CPU-only systems**: Shows "CPU" option

```python
gpu_choices = [f"GPU {i}" for i in multi_gpu_manager.available_gpus]
if not gpu_choices:
    gpu_choices = ["CPU"]  # Fallback for CPU-only

gpu_selection = gr.CheckboxGroup(
    choices=gpu_choices,
    value=[gpu_choices[0]],  # Default: first GPU
    label="Select GPUs",
    visible=len(gpu_choices) > 1
)
```

**Parsing GPU Selection:**
```python
target_gpus = []
for g in selected_gpus:
    if "CPU" in g:
        target_gpus.append(-1)  # CPU device ID
    else:
        gpu_id = int(g.split(":")[0].replace("GPU ", ""))
        target_gpus.append(gpu_id)
```

---

### 3. Streaming Audio Generation

**Function:** `gen_single_streaming()` (lines 377-540)

Generates audio in real-time with progress updates.

**Key Features:**
- Segments long text into chunks (default: 120 tokens)
- Processes segments sequentially or in parallel across GPUs
- Streams audio chunks as they're generated
- Shows real-time progress log

**Emotion Control Logic:**

| Mode | Control | Audio Needed | Example |
|------|---------|--------------|---------|
| 0 | Speaker | No | Use speaker's natural emotion |
| 1 | Reference Audio | Yes (upload) | "Make it sound sad like this audio" |
| 2 | Vector Sliders | No | Manually adjust 8 emotion dimensions |
| 3 | Text Description | No | "sad, crying, desperate" |

---

### 4. Emotion Control System

#### Mode 0: Speaker Emotion (Default)
- Uses the speaker's natural emotion from the reference audio
- No additional emotion parameters needed
- **Setting:** `emo_control_method = 0`

#### Mode 1: Reference Audio Emotion
- Extracts emotion from a separate reference audio file
- User uploads both speaker audio AND emotion reference audio
- **Setting:** `emo_control_method = 1`
- **Required:** `emo_upload` (emotion reference audio file)
- **Optional:** `emo_weight` (0-1, blending strength)

#### Mode 2: Emotion Vector Control
- Manually control 8 emotion dimensions with sliders
- Each dimension ranges from -5 to +5
- Vectors are normalized and applied to generation
- **Setting:** `emo_control_method = 2`
- **Parameters:** `vec1` through `vec8` (emotion vectors)
- **Optional:** `emo_weight` (blend strength)

#### Mode 3: Text-Based Emotion (Experimental)
- Describe emotions in natural language
- System uses Qwen LLM to convert text to emotion vectors
- **Setting:** `emo_control_method = 3`
- **Required:** `emo_text` (emotion description)
- **Optional:** `emo_weight` (blend strength)
- **Optional:** `emo_random` (randomize emotion)

**Examples:**
```python
# Mode 0: Use speaker's emotion
emo_control_method = 0

# Mode 1: Use reference audio emotion
emo_control_method = 1
emo_upload = "sad_voice.wav"
emo_weight = 0.8

# Mode 2: Use vector control
emo_control_method = 2
vec1, vec2, vec3, vec4 = 0.5, -0.3, 0.2, 0.1  # Manual values
vec5, vec6, vec7, vec8 = 0.0, 0.1, -0.2, 0.3

# Mode 3: Use text description
emo_control_method = 3
emo_text = "sad, melancholic, with a hint of anger"
emo_weight = 0.6
```

---

## Usage Guide

### Basic Setup

1. **Single GPU (Default)**
   - Just upload speaker audio and text
   - GPU 0 is used automatically

2. **Multi-GPU Setup**
   - GPU checkbox appears automatically
   - Select multiple GPUs for parallel processing
   - Higher VRAM usage but faster generation

### Workflow

#### Step 1: Select GPUs
```
 GPU 0
 GPU 1
```

#### Step 2: Upload Speaker Audio
- Click "Voice Reference Audio" upload box
- Upload WAV/MP3 file (any speaker)

#### Step 3: (Optional) Preload Voice
- Click "= Preload Voice" button
- Status shows: " Voice preloaded on: GPU 0, GPU 1"
- This caches embeddings to speed up generation

#### Step 4: Configure Emotion Control
**Option A - Speaker Emotion (Default):**
- Leave emotion control as "Same emotion as speaker"

**Option B - Reference Audio:**
- Select "Reference audio emotion"
- Upload emotion reference audio
- Adjust "Emotion Strength" slider (0-1)

**Option C - Vector Control:**
- Select "Emotion vector control"
- Adjust 8 emotion dimension sliders (-5 to +5)
- Adjust "Emotion Strength" slider

**Option D - Text Description:**
- Select "Text-based emotion control"
- Enter emotion description (e.g., "happy, energetic, excited")
- Optionally enable "Random Emotion"

#### Step 5: Enter Text
- Type or paste text in "Text" field
- Configure segmentation in Advanced Settings
- Default: 120 tokens per segment

#### Step 6: Generate
- Click "Generate Speech" button
- Watch progress in "Generation Log"
- Audio plays and downloads automatically

---

## Advanced Configuration

### Command Line Arguments

```bash
python webui.py --help

# Multi-GPU specific:
--fp16              # Use FP16 precision (saves VRAM)
--deepspeed         # Use DeepSpeed acceleration
--cuda_kernel       # Use CUDA kernels for optimization
--accel             # Acceleration engine (requires flash_attn)
--compile           # Use torch.compile for optimization
```

### Example: Run with FP16 + DeepSpeed

```bash
python webui.py --fp16 --deepspeed --cuda_kernel
```

---

## Performance Characteristics

### Single GPU (GPU 0)
- Baseline generation speed
- Suitable for: Short texts, real-time demos

### Multi-GPU (GPU 0 + GPU 1)
- ~2x speedup for long texts
- Better for: Long-form content, batch processing
- Trade-off: Higher VRAM usage

### Streaming Mode
- Real-time audio output
- Progress tracking
- Recommended for: Interactive use, user feedback

### Batch Mode (Non-Streaming)
- Faster overall (no UI overhead)
- Recommended for: Offline processing

---

## Troubleshooting

### Issue: "Please upload a voice reference audio first"
**Solution:** Upload audio file to "Voice Reference Audio" field

### Issue: GPU selection doesn't work
**Solutions:**
1. Check `torch.cuda.is_available()` returns True
2. Verify GPU drivers: `nvidia-smi`
3. Check VRAM availability: `nvidia-smi --query-gpu=memory.free`

### Issue: Audio preloading fails
**Solutions:**
1. Ensure audio file is valid WAV/MP3
2. Check audio duration (should be 2-20 seconds)
3. Verify VRAM is sufficient for model + audio embeddings
4. Check emotion mode settings match UI selection

### Issue: Out of Memory errors
**Solutions:**
1. Use `--fp16` flag to reduce VRAM usage
2. Reduce `max_text_tokens_per_segment` in Advanced Settings
3. Use single GPU instead of multi-GPU
4. Reduce `max_mel_tokens` in generation parameters

### Issue: Text-based emotion (Mode 3) doesn't work
**Solutions:**
1. Enable "Show Experimental Features" checkbox
2. Verify Qwen LLM model is loaded
3. Enter emotion description in English or Chinese
4. Check that emotion text is not empty

---

## Architecture Details

### GPU Memory Layout

**Per GPU Instance:**
```
GPU Memory Usage:
   Model Weights: ~8GB (quantized) or ~16GB (full precision)
   Speaker Embeddings Cache: ~100MB
   Emotion Embeddings Cache: ~50MB
   Generation Buffers: ~2-4GB (depends on max_mel_tokens)

Total per GPU: ~10-24GB depending on configuration
```

### Data Flow

```
User Input
    “
                                 
   GPU Selection & Preload       
                                $
  GPU 0: Load model + cache     
  GPU 1: Load model + cache     
  GPU N: Load model + cache     
                                 
    “
                                 
   Text Processing               
 Split into segments (120 tokens)
                                 
    “
                                 
   Parallel Generation           
 GPU 0 ’ Segment 1 ’ Audio chunk 
 GPU 1 ’ Segment 2 ’ Audio chunk 
 GPU 0 ’ Segment 3 ’ Audio chunk 
 ...                             
                                 
    “
                                 
   Audio Concatenation           
   Stream to user                
                                 
```

---

## Emotion Control Implementation

### Emotion Processing Pipeline

```python
# Mode selection and processing
if emo_control_method == 0:
    # No emotion reference needed
    emo_vector = None

elif emo_control_method == 1:
    # Extract emotion from audio
    emo_audio_prompt = emo_upload
    emo_vector = None

elif emo_control_method == 2:
    # Use manual vectors
    vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    emo_vector = tts.normalize_emo_vec(vec, apply_bias=True)

elif emo_control_method == 3:
    # Convert text to vector (experimental)
    if emo_text:
        emo_dict = tts.qwen_emo.inference(emo_text)
        emo_vector = list(emo_dict.values())
```

### Emotion Vector Normalization

```python
emo_vector = tts.normalize_emo_vec(
    input_vector,
    apply_bias=True
)
# Output: Normalized 8-dimensional emotion vector
# Range: [-1, 1] per dimension
```

---

## Recent Fixes (December 2024)

### Fixed Issues:

1. **TypeError in experimental mode toggle**
   - Fixed string/int comparison error

2. **GPU selection not passed to functions**
   - Added `gpu_selection` to preload and generate button handlers

3. **CPU mode parsing crash**
   - Fixed int("CPU") error with proper type handling

4. **Emotion mode 3 vector overwrite**
   - Fixed logic to preserve vectors for text-based emotion

5. **Wrong emotion reference for vector control**
   - Fixed mode 2 to not use audio reference

6. **Audio validation**
   - Added file existence checking

7. **Chinese UI translation**
   - Translated all labels to English
   - Updated emotion control mode descriptions

---

## Future Enhancements

- [ ] Dynamic GPU memory management
- [ ] Queue-based parallel processing
- [ ] Emotion vector visualization
- [ ] Batch file processing
- [ ] Model checkpoint switching per GPU
- [ ] Advanced emotion blending (cross-fade emotions)
