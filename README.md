# ComfyUI-MiniMax-Music3-AudioPrompter 🎵✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-Music-01](https://img.shields.io/badge/Model-MiniMax--Music--01-green.svg)](https://github.com/comfyanonymous/ComfyUI)

An intelligent, production-grade **Audio Analysis, Multi-Engine Cloud Lyrics Fetcher, and Dynamic Prompt Generator** custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and **MiniMax Music 01 / Music-3**.

---

## 🌟 Key Features (核心特色)

1. ⚡ **Acoustic Physical DSP Analysis (物理聲學訊號分析)**:
   - **Octave-Error Robust BPM Detection**: Uses onset strength envelopes and cyclic Tempograms to eliminate BPM halving/doubling errors.
   - **Key & Scale Estimation**: Chromagram (CQT) tonal distribution analysis.
   - **Acoustic Energy & Dynamic Profile**: Analyzes RMS, spectral rolloff, and spectral centroids to categorize audio brightness and drive.

2. ☁️ **Dual-Engine 100% Cloud Lyrics Fetcher (雙引擎官方精確歌詞檢索)**:
   - Queries **NetEase Music (網易雲音樂)** and **LRCLib (全球開放歌詞庫)** concurrently.
   - 100% immune to local Whisper ASR homophone confusion, hallucinations, and background music bleed.
   - Supports manual search keywords (`song_title_or_hint`) or automatic Whisper query derivation.

3. 🎤 **Anti-Monotone & Soaring Climax Engine (寬廣音域與副歌高亢爆發)**:
   - Injects melodic pitch elevation cues (`Wide Pitch Range & Soaring Climax`) to eliminate flat, robotic, monotonic chanting.
   - Supports nuanced vocal styles (Soulful Vibrato, Intimate Breathy, Emotional Belting).

4. 🫁 **Organic Human Phrasing & Natural Breathing (真人呼吸頓挫時值)**:
   - Introduces natural rubato phrasing and breath breaks (`Organic Natural Breathing`) to break mechanical, metronome-like lyric intervals.

5. ⏳ **Modular Structural Controls (模組化大歌章節開關)**:
   - **5-Second Fast Intro**: Avoids 30-40s tedious instrumental build-ups.
   - **Instrumental Solo Gap**: Automatically inserts structured guitar/piano solos.
   - **Bridge & Double Climax Chorus**: Allows generating full-length 4~5 minute epic songs (~287s).
   - **Outro Fade Style**: Supports slow fade-outs, vocal ad-libs, or direct stops.

6. 🛡️ **Zero Noise & Automatic Latent Alignment (零噪聲與動態時長對齊)**:
   - Outputs dynamic `seconds` to connect to `EmptyMiniMaxMusic3LatentAudio`, preventing noise bursts when generation terminates early.

---

## 🚀 Installation (安裝方式)

### 1. Install AudioPrompter
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/okanchou9/ComfyUI-MiniMax-Music3-AudioPrompter.git
pip install -r ComfyUI-MiniMax-Music3-AudioPrompter/requirements.txt
```

### 2. (Optional Advanced) Install Open-RVQ-Encoder for Melody Conditioning
If you want bottom-up melody and rhythm conditioning from reference tracks:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/johndpope/open-rvq-encoder-minimax-music3.git
```
**Model Weights Setup:**
- Download `dav.pth` from [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) $\rightarrow$ place in `ComfyUI/models/vae/dav.pth`.
- Download `minimax_music3_rvq_encoder_v4_169m_autoregressive_depth_recommended.safetensors` from [SimpleTuner](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3) $\rightarrow$ place in `ComfyUI/models/minimax_music3_rvq_encoders/`.

---

## 🎹 Ready-to-Use Workflows (範例工作流程)

We provide two verified, production-grade workflows in `workflows/`:

### 1. Standard Full-Song Remix Workflow (`workflows/minimax_music3_remix_fullsong.json`)
*Pure Prompt-Driven 287-second Epic Song Generation.*
```
[LoadAudio] ──> [MiniMaxMusic3AudioAutoPrompter] 
                     │              │
              (caption)            (lyrics)
                     │              │
                     ▼              ▼
         [MiniMaxMusic3TextEncode] (cfg_scale=1.2, max_duration=287.0)
                     │              │
             (conditioning)      (seconds)
                     │              │
                     ▼              ▼
                 [KSampler] <── [EmptyMiniMaxMusic3LatentAudio]
                     │
                     ▼
             [VAEDecodeAudio] ──> [SaveAudioMP3] (320kbps)
```

### 2. Dual-Track Advanced Remix Workflow (`workflows/minimax_music3_audioprompter_plus_rvq_v4.json`)
*Combines Global AudioPrompter Structuring with Bottom-Up RVQ-v4 Melody Guidance.*
```
[LoadAudio]
   │
   ├──> [MiniMaxMusic3AudioAutoPrompter] (caption + lyrics + structure)
   │           │               │
   └──> [MiniMaxMusic3RVQReferenceEncoderLoader] (v4 169M)
               │               │
               ▼               ▼
   [MiniMaxMusic3ReferenceAudioEncode] (reference_interval=5, cfg_scale=1.2)
               │               │
        (conditioning)      (seconds)
               │               │
               ▼               ▼
           [KSampler] <── [EmptyMiniMaxMusic3LatentAudio]
               │
               ▼
       [VAEDecodeAudio] ──> [SaveAudioMP3] (320kbps)
```

---

## ⚙️ Parameters Guide (參數詳細說明)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | `AUDIO` | Required | Input audio tensor from `LoadAudio`. |
| `auto_detect_genre` | `BOOLEAN` | `True` | AI physical audio detection vs manual override. |
| `language` | `COMBO` | `zh (Chinese)` | Language for cloud search and Whisper ASR. |
| `use_cloud_lyrics` | `BOOLEAN` | `True` | Fetches 100% accurate cloud lyrics from NetEase/LRCLib. |
| `song_title_or_hint`| `STRING` | `""` | Optional song name / artist query (e.g. `雨一直下 - 张宇`). |
| `rhythm_tag` | `COMBO` | `Steady Pop Drums` | Percussion style placed in prominent prompt position. |
| `instruments_tag` | `COMBO` | `Grand Piano + Acoustic Guitar` | Instrumentation layers. |
| `vocal_tag` | `COMBO` | `Emotional Male Vocal` | Lead vocal timbre. |
| `vocal_expression` | `COMBO` | `Wide Pitch Range & Soaring Climax` | **Recommended**: High pitch dynamics & emotional lifts. |
| `human_phrasing` | `COMBO` | `Organic Natural Breathing` | **Recommended**: Natural pauses & human breathing. |
| `intro_tag` | `COMBO` | `Short Intro ~5s` | **Recommended**: Enters verse within 5s. |
| `production_tag` | `COMBO` | `Clean Studio Mix` | Mix and mastering environment. |
| `repeat_final_chorus`| `BOOLEAN`| `True` | Repeats final chorus for full-track building climax. |
| `insert_instrumental_solo`| `BOOLEAN`| `True` | Inserts structured instrumental solo gap. |
| `include_bridge` | `BOOLEAN` | `True` | Adds emotional transition bridge before final climax. |
| `outro_style` | `COMBO` | `Slow Fade Out` | Outro style (Fade out / Ad-lib / Direct stop). |
| `dynamic_progression`| `COMBO`| `Building Climax` | Dynamic progression curve. |

---

## 💡 Best Practice & Recommendations (推薦最佳設定)

- **`cfg_scale`**: Set to **`1.2`** in `MiniMaxMusic3TextEncode` / `MiniMaxMusic3ReferenceAudioEncode`. Anything above `1.3` causes cumulative autoregressive speed-up drift on 4+ minute audio.
- **`reference_interval`**: Set to **`5`** in RVQ mode. Setting to `1` is too rigid and can conflict with lyrics phrasing.
- **`max_duration`**: Set to **`287.0`** (or your original song's exact seconds).
- **Vocal Elevation**: Keep `vocal_expression = Wide Pitch Range & Soaring Climax` to ensure energetic chorus delivery.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ❤️ Acknowledgements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [MiniMax AI](https://www.minimaxi.com/)
- [johndpope & SimpleTuner Team](https://github.com/johndpope/open-rvq-encoder-minimax-music3) (for Open RVQ Encoders)
- [NetEase Cloud Music & LRCLib](https://lrclib.net/)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) & [librosa](https://github.com/librosa/librosa)
