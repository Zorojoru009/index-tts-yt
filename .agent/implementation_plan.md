# Multi-GPU Implementation Plan

## Goal
Enable parallel processing of text segments across multiple GPUs (e.g., 2x T4 on Kaggle) to double generation speed for long-form content.

## Architecture

### 1. `GPUPool` Class (New)
A singleton manager in `webui.py` (or new module) that manages multiple `IndexTTS2` instances.
- **Attributes**:
  - `models`: Dict[int, IndexTTS2] keys are GPU indices
  - `lock`: Threading lock for valid initialization
- **Methods**:
  - `init_model(device_id, args)`: Loads model into specific GPU memory if not present.
  - `get_model(device_id)`: Returns the instance.
  - `unload_model(device_id)`: Optional, for memory management.

### 2. Distributed Inference Logic
New function `gen_distributed(...)` or modified `gen_wrapper`:
1.  **Tokenizer**: Run on CPU (main thread) to split text into `N` segments.
2.  **Scheduling**: Assign segments round-robin:
    - Seg 0 -> GPU 0
    - Seg 1 -> GPU 1
    - Seg 2 -> GPU 0
    - ...
3.  **Execution**: Use `concurrent.futures.ThreadPoolExecutor`.
    - Submit tasks for all segments immediately.
    - Each task runs `model.infer_generator` (or single chunk infer) on its assigned GPU.
4.  **Streaming Assembly**:
    - Main loop waits for Future 0, then Future 1, etc.
    - Yields results as they become available.
    - This ensures `Stream` order is correct (User hears sentence 1, then 2) even if 2 finishes before 1.

### 3. UI Changes
- **New Component**: `gpu_selector = gr.CheckboxGroup(...)`
  - Populated via `torch.cuda.device_count()` at launch.
  - Labels: "GPU 0: Tesla T4", "GPU 1: Tesla T4"
- **Logic**: Use selected GPUs for generation. If 1 selected -> Standard behavior. If multiple -> Distributed behavior.

## Implementation Details

### Step 1: Detect and List GPUs
```python
def get_available_gpus():
    if not torch.cuda.is_available():
        return []
    return [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
```

### Step 2: Model Management
Instead of one global `tts`, we will have a `tts_pool`.
The existing `tts` global can act as "GPU 0 default" or be deprecated in favor of the pool.
For backward compatibility, we can keep `tts` as the instance on GPU 0.

### Step 3: Distributed Generator Function
```python
def distributed_generator(text, selected_gpu_indices, ...):
    # 1. Tokenize (Main Thread) to get segments
    segments = tts_pool[0].tokenizer.split_segments(text)
    
    # 2. Define Worker Function
    def process_segment(gpu_id, segment, index):
        model = tts_pool[gpu_id]
        # Return audio chunk
        return model.infer_segment(segment, ...)

    # 3. Submit Jobs
    futures = []
    with ThreadPoolExecutor() as executor:
        for i, seg in enumerate(segments):
             gpu_id = selected_gpu_indices[i % len(selected_gpu_indices)]
             futures.append(executor.submit(process_segment, gpu_id, seg, i))
        
        # 4. Yield in Order
        for future in futures:
             yield future.result()
```

## Risks & Mitigations
- **VRAM Usage**: Loading 2 models (4GB+ each) + CUDA context.
  - *Mitigation*: 2x16GB T4 is plenty of room.
- **Latency**: First chunk might take slightly longer if waiting for heavy pool init.
  - *Mitigation*: Pre-load models at startup or on "Preload Voice" click.
- **Race Conditions**: `IndexTTS2` has some shared caches?
  - *Check*: `cache_spk_cond` etc. are instance attributes. Should be safe if instances are distinct.

## Execution Steps
1. [ ] Create `MultiGPUManager` in `webui.py`.
2. [ ] Replace global `tts` init with Manager init (loading at least GPU 0).
3. [ ] Add GPU Selector to UI.
4. [ ] Implement `infer_distributed` logic.
5. [ ] Wire up to `gen_button`.
