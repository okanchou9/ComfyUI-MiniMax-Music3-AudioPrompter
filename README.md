# ComfyUI-MiniMax-Music3-AudioPrompter 🎵✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-Music-01](https://img.shields.io/badge/Model-MiniMax--Music--01-green.svg)](https://github.com/comfyanonymous/ComfyUI)

An intelligent, production-grade **Audio Analysis, Multi-Engine Cloud Lyrics Fetcher, RVQ Melody Cache & Reuse Engine, and Dynamic Prompt Generator** custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and **MiniMax Music 01 / Music-3**.

---

## 🌟 Key Features (核心特色)

1. ⚡ **0-Second RVQ Cache & Reuse Engine (0秒瞬發 RVQ 語意特徵快取系統)**:
   - Save expensive 169M RVQ neural codebook predictions to tiny `~280 KB` `.pt` cache files (`SaveMiniMaxMusic3RVQCache`).
   - Reload cached melody contours instantly (`LoadMiniMaxMusic3RVQCache`) without touching audio files, DAV, or 169M models!
   - **Releases 2GB+ VRAM** and enables rapid iterative prompt remixing.

2. ⚡ **Acoustic Physical DSP Analysis (物理聲學訊號分析)**:
   - **Octave-Error Robust BPM Detection**: Uses onset strength envelopes and cyclic Tempograms to eliminate BPM halving/doubling errors.
   - **Key & Scale Estimation**: Chromagram (CQT) tonal distribution analysis.
   - **Acoustic Energy & Dynamic Profile**: Analyzes RMS, spectral rolloff, and spectral centroids to categorize audio brightness and drive.

3. ☁️ **Dual-Engine 100% Cloud Lyrics Fetcher (雙引擎官方精確歌詞檢索)**:
   - Queries **NetEase Music (網易雲音樂)** and **LRCLib (全球開放歌詞庫)** concurrently.
   - 100% immune to local Whisper ASR homophone confusion, hallucinations, and background music bleed.

4. 🎤 **Anti-Monotone & Soaring Climax Engine (寬廣音域與副歌高亢爆發)**:
   - Injects melodic pitch elevation cues (`Wide Pitch Range & Soaring Climax`) to eliminate flat, robotic, monotonic chanting.

5. 🫁 **Organic Human Phrasing & Natural Breathing (真人呼吸頓挫時值)**:
   - Introduces natural rubato phrasing and breath breaks (`Organic Natural Breathing`) to break mechanical, metronome-like lyric intervals.

6. ⏳ **Modular Structural Controls (模組化大歌章節開關)**:
   - **5-Second Fast Intro**: Avoids 30-40s tedious instrumental build-ups.
   - **Instrumental Solo Gap**: Automatically inserts structured guitar/piano solos.
   - **Bridge & Double Climax Chorus**: Allows generating full-length 4~5 minute epic songs (~287s).
   - **Outro Fade Style**: Supports slow fade-outs, vocal ad-libs, or direct stops.

7. 🛡️ **Zero VAE OOM & Tiled Audio Decoding (無損分塊音訊解碼)**:
   - Employs `VAEDecodeAudioTiled` (`tile_size=1024`, `overlap=64`), eliminating memory spikes and decoding 5-minute tracks in ~26 seconds.

---

## 🚀 Installation (安裝方式)

### 1. Install AudioPrompter
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/okanchou9/ComfyUI-MiniMax-Music3-AudioPrompter.git
pip install -r ComfyUI-MiniMax-Music3-AudioPrompter/requirements.txt
```

### 2. (Optional Advanced) Install Open-RVQ-Encoder for Melody Extraction
If you want to extract melody from reference tracks:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/johndpope/open-rvq-encoder-minimax-music3.git
```
**Model Weights Setup:**
- Download `dav.pth` from [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) $\rightarrow$ place in `ComfyUI/models/vae/dav.pth`.
- Download `minimax_music3_rvq_encoder_v4_169m_autoregressive_depth_recommended.safetensors` from [SimpleTuner](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3) $\rightarrow$ place in `ComfyUI/models/minimax_music3_rvq_encoders/`.

---

## 🎹 Ready-to-Use Workflows (範例工作流程)

We provide three verified, production-grade workflows in `workflows/`:

### 1. Fast Remix with Cached RVQ (`workflows/minimax_music3_fast_remix_with_cache.json`) ⚡
*Instant 0-second melody reuse. Swap lyrics, prompts, and seeds without re-running audio encoders.*
```
[LoadMiniMaxMusic3RVQCache] (sample_c0.pt) ──────┐
                                                  │ (semantic_candidates)
[MiniMaxMusic3AudioAutoPrompter] (caption+lyrics) │
       │                    │                     │
       ▼                    ▼                     ▼
[MiniMaxMusic3TextEncodeWithCachedReference] (reference_interval=1, cfg=1.2)
       │                    │
 (conditioning)          (seconds)
       │                    │
       ▼                    ▼
   [KSampler] <── [EmptyMiniMaxMusic3LatentAudio]
       │
       ▼
 [VAEDecodeAudioTiled] (tile_size=1024) ──> [SaveAudioMP3] (320kbps)
```

### 2. Standard Full-Song Remix Workflow (`workflows/minimax_music3_remix_fullsong.json`)
*Pure Prompt-Driven 287-second Epic Song Generation.*

### 3. Dual-Track Live RVQ Workflow (`workflows/minimax_music3_audioprompter_plus_rvq_v4.json`)
*End-to-End Live Audio Conditioning + AudioPrompter Structuring.*

---

## 💡 Best Practice & Recommendations (推薦最佳設定)

- **Strict Remix (緊咬原曲旋律)**: Set `reference_interval = 1` in `MiniMaxMusic3TextEncodeWithCachedReference`.
- **Creative Remix (部分引導/自由發揮)**: Set `reference_interval = 5`.
- **`cfg_scale`**: Set to **`1.2`** to avoid autoregressive tempo drift on 4+ minute audio.
- **`tile_size`**: Use **`1024`** with `overlap=64` on 16GB/24GB/32GB GPUs for optimal decoding speed and zero OOM.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
