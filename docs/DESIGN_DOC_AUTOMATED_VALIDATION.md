# Design Document: Automated Audio Validation & Regeneration (Future Feature)

## 1. Overview
This feature aims to introduce an Automated Quality Assurance (QA) loop into the TTS generation pipeline. By utilizing a Speech-to-Text (STT/ASR) model, the system will transcribe each generated audio chunk and compare it against the original input text to detect skips, hallucinations, or pronunciation errors.

## 2. Core Architecture
To maximize hardware efficiency, the system will use a heterogeneous computing strategy:
*   **Text-to-Speech (TTS):** continues to run on **GPU(s)** (utilizing the `MultiGPUManager`).
*   **Validation (STT):** runs exclusively on **CPU**.

This ensures that the expensive GPU resources are dedicated solely to generation, while the validation (which is computationally lighter for modern models like Paraformer or Whisper Tiny) runs in parallel on the CPU without blocking the generation queue.

## 3. Technology Stack
*   **STT Model:** `funasr` (Paraformer) is recommended.
    *   **Reason:** Extremely fast, lightweight, and native support via `modelscope` (already a dependency). Excellent for mixed Chinese/English.
    *   **Alternative:** `openai-whisper` (Tiny/Small models).
*   **Comparison Metric:** Levenshtein Distance (Word Error Rate - WER / Character Error Rate - CER).
*   **Library:** `python-Levenshtein` or `fuzzywuzzy`.

## 4. Workflow Detail

1.  **Generation:** The `gen_single_streaming` function generates an audio chunk `Audio_N`.
2.  **Async Validation:** `Audio_N` is pushed to a `ValidationQueue`. A background worker (CPU thread) picks it up.
3.  **Transcription:** The STT model transcribes `Audio_N` -> `Transcript_N`.
4.  **Scoring:**
    *   Compare `Transcript_N` vs `Reference_Text_N`.
    *   Calculate `Similarity_Score` (0.0 to 1.0).
    *   Flag if `Similarity_Score` < `Threshold` (e.g., 0.85).
5.  **UI Feedback:** The frontend receives the Score and Flag status in real-time.

## 5. UI / UX Changes
The current "Single Player" output will be replaced or augmented by a **"Chunk Review Interface"**:

### Interface Mockup
| Chunk | Status | Score | Play | Text Segment | Actions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | ✅ OK | 98% | ▶️ | "Power is not about..." | [Regen] |
| #2 | ⚠️ Low | **65%** | ▶️ | "...finding yourself." | **[Regenerate]** |
| #3 | ✅ OK | 95% | ▶️ | "It's about destroying..."| [Regen] |

*   **Regenerate Button:** When clicked, sends a specific request to re-generate *only* Chunk #2 with a different random seed or slight parameter tweak.
*   **Merge & Download:** A global button to concatenate all "Accepted" chunks into the final WAV file.

## 6. Implementation Roadmap

### Phase 1: Minimum Viable Product (The "Review Panel")
**Goal:** Manual review and regeneration capability. No AI validation yet.
1.  **Backend:** Split `gen_single_streaming` to allow generating specific chunk indices (e.g., `gen_chunk(index=5)`).
2.  **Frontend:** Build the **Master-Detail UI**.
    *   List View: Table of all chunks.
    *   Detail View: Player + Text Editor + "Regenerate This Chunk" button.
    *   Merge: "Combine All" button.
*   **Result:** You can manually fix bad chunks, but you have to find them yourself by listening.

### Phase 2: Automated Intelligence (STT Integration)
**Goal:** The system tells *you* which chunks are bad.
1.  **Dependencies:** Install `funasr` and `python-Levenshtein`.
2.  **Backend:** Implement `AudioValidator` (CPU-based).
3.  **Integration:**
    *   Run Validation after each chunk generation.
    *   Feed the `Score` into the Phase 1 List View.
    *   **Auto-Sort:** Put the lowest underscores (red flags) at the top of the list.
*   **Result:** You only review the 5% of chunks that the AI flagged as "Suspicious". Massive time saver.
