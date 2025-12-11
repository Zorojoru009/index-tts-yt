# Index-TTS-YT Streaming Enhancement

## 🎉 New Features

This enhanced version of Index-TTS-YT adds powerful streaming capabilities for
- **Long-form Audio Generation**: Break free from the 2-minute limit. Generate 20-50+ minute audio files.
- **Streaming Playback**: Audio starts playing within seconds, no matter how long the text is.
- **Real-time Feedback**: Live log shows generation progress, current text segment, and processing speed.
- **Voice Preloading**: Cache voice embeddings to speed up subsequent generations.
- **Multi-GPU Support**: Distribute generation across multiple GPUs (e.g., 2x T4) for up to 2x speed.
- **Improved UI**: Cleaner, localized (English), and more intuitive interface.
- Significantly speeds up generation when using the same voice multiple times

## ✨ What's New

### 2. **Streaming Audio Generation** 🎵
- Real-time audio playback starts within seconds
- No more waiting for entire generation to complete
- Perfect for long-form content (20-50 minutes)
- Prevents timeout issues with very long texts

### 3. **Live Progress Logging** 📊
- Real-time generation log shows:
  - Current chunk being processed (e.g., "Chunk 5/20")
  - Processing time per chunk
  - RTF (Real-Time Factor) metrics
  - Estimated time remaining
  - Preview of text being synthesized
- Helps monitor progress during long generations

### 4. **Flexible Generation Modes** ⚡
- **Streaming Mode** (default): Best for long texts (5+ minutes)
  - Real-time playback
  - Live progress updates
  - Memory efficient
- **Batch Mode**: Best for short texts (<5 minutes)
  - Traditional all-at-once generation
  - Simpler output

### 5. **Performance Optimizations** 🚀
Already included from previous enhancements:
- `--accel`: Acceleration engine (PagedAttention, FlashAttention, CUDA Graphs)
- `--compile`: torch.compile optimization
- `--fp16`: FP16 precision for faster inference

## 🎯 Usage Guide

### Basic Workflow

1. **Upload Voice Reference Audio**
   - Click on "音色参考音频" to upload your reference voice

2. **Preload Voice (Optional but Recommended)**
   - Click "🔄 预加载音色" button
   - Wait for "✅ 音色已预加载" status
   - This speeds up generation, especially for long texts

3. **Enter Your Text**
   - Paste or type your text in the text area
   - The preview will show how it will be segmented

4. **Choose Generation Mode**
   - ✅ **Enable Streaming** (recommended for long texts)
     - Real-time playback
     - Live progress updates
   - ⬜ **Disable Streaming** (for short texts)
     - Traditional batch generation

5. **Click Generate**
   - Watch the live log for progress
   - Audio starts playing as it's generated (in streaming mode)

### Advanced Settings

#### Chunk Size Optimization
Located in "高级生成参数设置" → "分句设置"

- **Default: 120 tokens** - Good balance for most cases
- **Smaller (40-80)**: 
  - ✅ More granular control
  - ❌ May sound choppy
  - ❌ Slower overall generation
- **Larger (150-200)**:
  - ✅ More natural prosody
  - ❌ Risk hitting max_mel_tokens limit
  - ❌ May cause audio cutoff

**Recommendations:**
- Short texts (1-5 min): 80-120 tokens
- Medium texts (5-20 min): 100-140 tokens
- Long texts (20-50 min): 120-160 tokens

#### Generation Parameters
- **temperature**: Controls randomness (0.1-2.0)
  - Lower = more consistent
  - Higher = more varied
- **top_p**: Nucleus sampling (0.0-1.0)
- **max_mel_tokens**: Maximum output length per chunk
  - Default: 1500
  - Increase if you see warnings about truncation

## 🚀 Performance Tips

### For Long-Form Generation (20-50 minutes)

1. **Always preload voice first**
   - Saves time on every chunk

2. **Use streaming mode**
   - Prevents timeouts
   - Provides real-time feedback

3. **Optimize chunk size**
   - Start with 120 tokens
   - Adjust based on results

4. **Enable acceleration flags**
   ```bash
   python webui.py --accel --compile --fp16
   ```

5. **Monitor the live log**
   - Watch RTF (Real-Time Factor)
   - RTF < 1.0 = faster than real-time
   - RTF > 1.0 = slower than real-time

### Memory Management

For very long texts (50+ minutes):
- Close other applications
- Use `--fp16` flag to reduce memory usage
- Consider breaking into multiple sessions

## 🎬 Example Workflows

### Workflow 1: Quick Short Audio
```
1. Upload voice reference
2. Enter text (1-2 minutes)
3. Disable streaming mode
4. Click generate
5. Wait for complete audio
```

### Workflow 2: Long-Form Narration
```
1. Upload voice reference
2. Click "🔄 预加载音色"
3. Wait for "✅ 音色已预加载"
4. Paste long text (20-50 minutes)
5. Enable streaming mode
6. Adjust chunk size if needed (120-140)
7. Click generate
8. Watch live log and listen as it generates
```

### Workflow 3: Multiple Generations with Same Voice
```
1. Upload voice reference
2. Click "🔄 预加载音色" (once)
3. For each text:
   - Enter new text
   - Click generate
   - Voice cache is reused (faster!)
```

## 📊 Understanding the Live Log

Example log output:
```
🎙️ Starting generation...
📊 Total segments: 15
📝 Text length: 2500 characters, 450 tokens
⚙️ Chunk size: 120 tokens/segment
--------------------------------------------------
✅ Chunk 1/15 completed in 3.45s
   📊 RTF: 0.8234 | Audio: 4.19s
   📝 Text: 欢迎大家来体验IndexTTS2，这是一个突破性的...
   ⏱️ Est. remaining: 48.3s
--------------------------------------------------
```

**Key Metrics:**
- **RTF (Real-Time Factor)**: 
  - 0.8234 = generating faster than real-time
  - 1.0 = generating at real-time speed
  - 2.0 = taking twice as long as audio duration
- **Est. remaining**: Estimated time to complete all chunks

## 🔧 Command-Line Options

```bash
# Basic usage
python webui.py

# With all optimizations
python webui.py --accel --compile --fp16

# Custom port and host
python webui.py --host 0.0.0.0 --port 7860

# Verbose logging
python webui.py --verbose

# Custom chunk size default
python webui.py --gui_seg_tokens 140
```

## 🐛 Troubleshooting

### Issue: Audio sounds choppy
**Solution**: Increase chunk size (try 140-160 tokens)

### Issue: Audio gets cut off
**Solution**: 
- Decrease chunk size (try 80-100 tokens)
- OR increase max_mel_tokens (try 2000)

### Issue: Generation is slow
**Solution**:
- Enable acceleration: `--accel --compile --fp16`
- Preload voice before generation
- Check RTF in live log (should be < 1.0)

### Issue: Out of memory
**Solution**:
- Use `--fp16` flag
- Reduce chunk size
- Close other applications
- Break text into smaller parts

### Issue: Streaming not working
**Solution**:
- Make sure "启用流式生成" checkbox is enabled
- Check browser console for errors
- Try refreshing the page

## 📝 Technical Details

### Architecture Changes

1. **Voice Preloading**
   - Triggers voice analysis early
   - Populates cache: `cache_spk_cond`, `cache_emo_cond`
   - Reused across generations

2. **Streaming Generator**
   - Uses `infer_generator()` with `stream_return=True`
   - Yields audio chunks as they're generated
   - Yields progress updates for UI

3. **Live Progress**
   - Calculates metrics per chunk
   - Updates log in real-time
   - Estimates remaining time

### Caching Mechanism

The system caches:
- Speaker embeddings (`cache_spk_cond`)
- Emotion embeddings (`cache_emo_cond`)
- S2Mel style (`cache_s2mel_style`)
- S2Mel prompt (`cache_s2mel_prompt`)

Cache is invalidated when:
- Voice reference audio changes
- Emotion reference audio changes

## 🎓 Best Practices

1. **Always preload for long texts** - Saves significant time
2. **Use streaming for 5+ minute audio** - Better UX and prevents timeouts
3. **Monitor RTF in live log** - Indicates generation efficiency
4. **Start with default chunk size (120)** - Adjust only if needed
5. **Enable acceleration flags** - Significant speedup on compatible hardware
6. **Save intermediate results** - For very long generations

## 🔮 Future Enhancements

Potential improvements:
- [ ] Pause/Resume generation
- [ ] Save intermediate checkpoints
- [ ] Batch processing multiple texts
- [ ] Voice library management
- [ ] Custom emotion presets
- [ ] Export generation settings

## 📄 License

Same as original Index-TTS project.

## 🙏 Credits

Based on the original IndexTTS2 project with streaming enhancements for long-form audio generation.
