import os
import re
import json
import urllib.request
import urllib.parse
import torch
import numpy as np
import librosa
import scipy.stats
from faster_whisper import WhisperModel
import comfy.model_management

class MiniMaxMusic3AudioAutoPrompter:
    """MiniMax-Music-3 Audio Auto-Prompter Node.
    Performs physical acoustic analysis (Robust Tempogram BPM, Key, Duration, Acoustic Feature Extraction),
    integrates Multi-Engine Cloud Lyrics Fetching (LRCLib & NetEase Music), provides Modular Tag Selection
    (Rhythm, Instruments, Vocal Style, Vocal Expression/Melody, Human Phrasing, Intro Length, Mix Quality),
    Advanced Structural Switches (Repeat Final Chorus, Instrumental Solo, Bridge, Outro Style, Dynamics),
    and organic phrase-merged lyrics structure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "auto_detect_genre": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto Detect (AI)",
                    "label_off": "Manual Override"
                }),
                "language": ([
                    "zh (Chinese)", "en (English)", "ja (Japanese)", 
                    "ko (Korean)", "yue (Cantonese)", "auto", "es (Spanish)", 
                    "fr (French)", "de (German)"
                ], {
                    "default": "zh (Chinese)",
                    "tooltip": "Language for speech recognition or cloud lyrics retrieval."
                }),
                "use_cloud_lyrics": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto Fetch Cloud Lyrics (100% Accurate)",
                    "label_off": "Local Whisper ASR Only"
                }),
                "rhythm_tag": ([
                    "Steady Pop Drums (穩定流行鼓組律動)",
                    "Acoustic Kick & Snare (清脆原聲大鼓軍鼓)",
                    "Dynamic Live Rock Drums (動態現場搖滾鼓組)",
                    "Gentle Brushed Drums (溫柔爵士鋼刷鼓)",
                    "Lo-fi Boom-Bap Beat (放鬆緩拍鼓點)",
                    "Punchy Electronic Kick (動感電音底鼓)",
                    "No Drums / Acoustic Only (無鼓純原聲)",
                    "Auto Detect (自動分析)"
                ], {
                    "default": "Steady Pop Drums (穩定流行鼓組律動)",
                    "tooltip": "Select rhythmic percussion style without hardcoding time signatures."
                }),
                "instruments_tag": ([
                    "Grand Piano + Acoustic Guitar (鋼琴 + 原聲吉他掃弦)",
                    "Grand Piano + Lush Strings (鋼琴 + 溫暖弦樂群)",
                    "Rhodes Piano + Acoustic Guitar (復古電鋼琴 + 吉他)",
                    "Electric Guitar + Tight Bass (電吉他 + 律動貝斯)",
                    "Synth Arpeggio + Bass (合成器 + 飽滿低音)",
                    "Fingerstyle Acoustic Guitar (原聲吉他指彈)",
                    "Auto Detect (自動分析)"
                ], {
                    "default": "Grand Piano + Acoustic Guitar (鋼琴 + 原聲吉他掃弦)",
                    "tooltip": "Select core lead and accompaniment instruments."
                }),
                "vocal_tag": ([
                    "Emotional Male Vocal (深情磁性男聲)",
                    "Raspy Soulful Male (沙啞靈魂撕裂感男聲)",
                    "Sweet Female Vocal (甜美抒情女聲)",
                    "Breathy Emotive Vocal (氣聲抒情唱腔)",
                    "Duet Vocal Harmony (男女對唱/和聲)",
                    "Instrumental Only (純音樂無人聲)",
                    "Auto / Model Default (預設)"
                ], {
                    "default": "Emotional Male Vocal (深情磁性男聲)",
                    "tooltip": "Select vocal tone and singing style."
                }),
                "vocal_expression": ([
                    "Wide Pitch Range & Soaring Climax (寬廣音域 + 副歌高亢爆發 - 告別唸經推薦!)",
                    "Soulful Melodic Vibrato (靈魂深情 + 自然顫音旋律)",
                    "Intimate Breathy Nuance (貼耳氣聲 + 細膩微起伏)",
                    "Steady Standard Delivery (標準平穩演唱)"
                ], {
                    "default": "Wide Pitch Range & Soaring Climax (寬廣音域 + 副歌高亢爆發 - 告別唸經推薦!)",
                    "tooltip": "Injects melodic contours, pitch elevation in choruses, and expressive vibrato to eliminate monotone reciting."
                }),
                "human_phrasing": ([
                    "Organic Natural Breathing (真人呼吸感 + 靈活時值頓挫 - 推薦!)",
                    "Strict Metric Timing (嚴格規整節奏)"
                ], {
                    "default": "Organic Natural Breathing (真人呼吸感 + 靈活時值頓挫 - 推薦!)",
                    "tooltip": "Breaks robotic metronome spacing with human-like breathing and natural rubato timing."
                }),
                "intro_tag": ([
                    "Short Intro ~5s (極簡短前奏 5秒內進主歌 - 推薦!)",
                    "Standard Intro ~15s (標準前奏 約15秒)",
                    "Direct Vocal Start (無前奏直接開唱)"
                ], {
                    "default": "Short Intro ~5s (極簡短前奏 5秒內進主歌 - 推薦!)",
                    "tooltip": "Controls intro length so vocals enter quickly and avoid wasting time."
                }),
                "production_tag": ([
                    "Clean Studio Mix (錄音室級清晰立體聲混音)",
                    "Warm Analog Tape Vibe (溫暖類比膠片質感)",
                    "Spacious Concert Reverb (開闊演唱會空間混響)",
                    "Tight Dry Presence (緊湊貼耳無雜音錄音)"
                ], {
                    "default": "Clean Studio Mix (錄音室級清晰立體聲混音)",
                    "tooltip": "Production and audio space mastering quality."
                }),
            },
            "optional": {
                "song_title_or_hint": ("STRING", {
                    "multiline": False,
                    "default": "雨一直下 - 张宇",
                    "tooltip": "Enter Song Title and Artist (e.g. 雨一直下 - 张宇 / Shape of You - Ed Sheeran) to auto-fetch 100% official lyrics!"
                }),
                "target_generation_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 600.0,
                    "step": 5.0,
                    "tooltip": "Target duration in seconds (0 = full song duration from input audio, or set 60.0 / 120.0 / 240.0)."
                }),
                "repeat_final_chorus": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Repeat Final Chorus (雙倍高潮副歌 - 推薦長歌)",
                    "label_off": "Single Final Chorus (單次副歌)"
                }),
                "insert_instrumental_solo": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Insert Instrumental Solo (段落間奏獨奏)",
                    "label_off": "No Solo Interlude (無間奏)"
                }),
                "include_bridge": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Include Bridge (包含情感橋段轉折)",
                    "label_off": "Skip Bridge (跳過橋段)"
                }),
                "outro_style": ([
                    "Slow Fade Out (慢速淡出結尾)",
                    "Vocal Ad-lib Outro (尾奏隨性吟唱)",
                    "Direct Stop (俐落收尾)"
                ], {
                    "default": "Slow Fade Out (慢速淡出結尾)",
                    "tooltip": "Outro musical fade and ending style."
                }),
                "dynamic_progression": ([
                    "Building Climax (主歌鋪墊，副歌與結尾爆發高潮)",
                    "Steady Smooth (全曲平穩流暢)",
                    "Intimate Stripped-back (極簡貼耳純伴奏)"
                ], {
                    "default": "Building Climax (主歌鋪墊，副歌與結尾爆發高潮)",
                    "tooltip": "Dynamic emotional arc across the entire song."
                }),
                "genre_override": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Leave blank to auto-detect genre style from audio, or fill in to override."
                }),
                "extra_tags_custom": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Add any additional custom tags separated by comma (e.g. nostalgic, cello solo, 80s vibe)."
                }),
                "lyrics_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Paste exact custom lyrics here to completely bypass cloud and speech recognition."
                }),
                "bpm_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 1.0}),
                "whisper_model_size": (["tiny", "base", "small", "medium", "large-v3"], {
                    "default": "base"
                }),
                "transcribe_lyrics": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "bpm", "musical_key", "duration_seconds", "analysis_summary")
    OUTPUT_NODE = True
    FUNCTION = "analyze_and_generate"
    CATEGORY = "MiniMaxMusic3/Prompting"

    def fetch_online_lyrics(self, query):
        """Searches LRCLib & NetEase Music APIs for 100% accurate official lyrics."""
        if not query or not query.strip():
            return None, None, None
            
        clean_query = query.strip()
        try:
            url = "https://music.163.com/api/search/get/web?csrf_token=hlpretag=&hlposttag=&s=" + urllib.parse.quote(clean_query) + "&type=1&offset=0&total=true&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                songs = data.get("result", {}).get("songs", [])
                if songs:
                    song_id = songs[0]["id"]
                    song_name = songs[0]["name"]
                    artist_name = songs[0]["artists"][0]["name"] if songs[0].get("artists") else ""
                    
                    lrc_url = "https://music.163.com/api/song/lyric?os=pc&id=" + str(song_id) + "&lv=-1&kv=-1&tv=-1"
                    req_lrc = urllib.request.Request(lrc_url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com"})
                    with urllib.request.urlopen(req_lrc, timeout=5) as resp_lrc:
                        lrc_data = json.loads(resp_lrc.read().decode("utf-8"))
                        lrc_text = lrc_data.get("lrc", {}).get("lyric", "")
                        if lrc_text:
                            return song_name, artist_name, lrc_text
        except Exception as e:
            print("[MiniMaxMusic3AudioAutoPrompter] NetEase Fetch Warning:", e)

        try:
            url = "https://lrclib.net/api/search?q=" + urllib.parse.quote(clean_query)
            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-MiniMax-AudioPrompter/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                items = json.loads(resp.read().decode("utf-8"))
                if items and len(items) > 0:
                    item = items[0]
                    plain = item.get("plainLyrics") or item.get("syncedLyrics")
                    if plain:
                        return item.get("trackName"), item.get("artistName"), plain
        except Exception as e:
            print("[MiniMaxMusic3AudioAutoPrompter] LRCLib Fetch Warning:", e)

        return None, None, None

    def estimate_robust_tempo(self, y, sr):
        """Estimates tempo using Tempogram Autocorrelation & Octave Ambiguity Correction."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            prior = scipy.stats.lognorm(s=0.45, scale=90)
            weighted_bpm = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, prior=prior)[0])
            
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            ac_global = np.mean(tempogram, axis=1)
            tempo_freqs = librosa.tempo_frequencies(len(ac_global), sr=sr)
            
            if weighted_bpm > 125.0:
                half_bpm = weighted_bpm / 2.0
                half_energy = np.sum(ac_global[(tempo_freqs >= half_bpm - 5) & (tempo_freqs <= half_bpm + 5)])
                main_energy = np.sum(ac_global[(tempo_freqs >= weighted_bpm - 5) & (tempo_freqs <= weighted_bpm + 5)])
                if half_energy > main_energy * 0.8 and half_bpm >= 60.0:
                    weighted_bpm = half_bpm
            elif weighted_bpm < 60.0:
                weighted_bpm = weighted_bpm * 2.0
                
            return round(weighted_bpm, 1)
        except Exception as e:
            print("[MiniMaxMusic3AudioAutoPrompter] Tempo Warning:", e)
            return 75.0

    def parse_tags_to_prompt(self, rhythm_tag, instruments_tag, vocal_tag, vocal_expression, human_phrasing, production_tag, dynamic_progression, final_genre, final_bpm, detected_key, extra_tags):
        rhythm_map = {
            "Steady Pop Drums (穩定流行鼓組律動)": "steady pop drum beat, punchy kick and tight snare, consistent rhythm",
            "Acoustic Kick & Snare (清脆原聲大鼓軍鼓)": "crisp acoustic drum kit, natural snare, steady tempo",
            "Dynamic Live Rock Drums (動態現場搖滾鼓組)": "dynamic live rock drums, driving crash",
            "Gentle Brushed Drums (溫柔爵士鋼刷鼓)": "gentle brushed acoustic drums, soft rhythmic ride",
            "Lo-fi Boom-Bap Beat (放鬆緩拍鼓點)": "mellow lo-fi boom-bap beat, relaxed groove",
            "Punchy Electronic Kick (動感電音底鼓)": "punchy electronic kick, energetic rhythmic pulse",
            "No Drums / Acoustic Only (無鼓純原聲)": "no drums, acoustic only, pure harmonic resonance",
            "Auto Detect (自動分析)": "clean tight drum groove, balanced rhythm"
        }
        r_desc = rhythm_map.get(rhythm_tag, "steady pop drum beat, consistent rhythm")

        inst_map = {
            "Grand Piano + Acoustic Guitar (鋼琴 + 原聲吉他掃弦)": "acoustic grand piano, rhythmic acoustic guitar strumming, melodic bassline",
            "Grand Piano + Lush Strings (鋼琴 + 溫暖弦樂群)": "grand piano, lush cinematic strings ensemble, warm deep bass",
            "Rhodes Piano + Acoustic Guitar (復古電鋼琴 + 吉他)": "vintage Rhodes electric piano, mellow acoustic guitar, smooth bass",
            "Electric Guitar + Tight Bass (電吉他 + 律動貝斯)": "clean electric guitar, driving bassline, warm backing organ",
            "Synth Arpeggio + Bass (合成器 + 飽滿低音)": "warm analog synth pad, pulsing synth bass, subtle arpeggios",
            "Fingerstyle Acoustic Guitar (原聲吉他指彈)": "intimate fingerstyle acoustic guitar, subtle ambient strings",
            "Auto Detect (自動分析)": "acoustic grand piano, acoustic guitar, clean bass"
        }
        i_desc = inst_map.get(instruments_tag, "acoustic grand piano, rhythmic acoustic guitar strumming")

        vocal_map = {
            "Emotional Male Vocal (深情磁性男聲)": "emotional male lead vocals, deep resonant tone",
            "Raspy Soulful Male (沙啞靈魂撕裂感男聲)": "raspy soulful male lead vocals, passionate singing",
            "Sweet Female Vocal (甜美抒情女聲)": "sweet emotive female lead vocals, clear melodic tone",
            "Breathy Emotive Vocal (氣聲抒情唱腔)": "breathy emotive vocals, intimate tone",
            "Duet Vocal Harmony (男女對唱/和聲)": "layered vocal harmonies, male and female duet",
            "Instrumental Only (純音樂無人聲)": "instrumental only, no vocals",
            "Auto / Model Default (預設)": "emotional expressive lead vocals"
        }
        v_base = vocal_map.get(vocal_tag, "emotional male lead vocals")

        # Vocal Expression & Pitch Contour Mapping
        vexpr_map = {
            "Wide Pitch Range & Soaring Climax (寬廣音域 + 副歌高亢爆發 - 告別唸經推薦!)": "expressive vocal contours, wide pitch dynamics, soaring high melody in choruses",
            "Soulful Melodic Vibrato (靈魂深情 + 自然顫音旋律)": "soulful melodic lines, natural vibrato, expressive pitch phrasing",
            "Intimate Breathy Nuance (貼耳氣聲 + 細膩微起伏)": "intimate subtle melodic nuances, warm vocal presence",
            "Steady Standard Delivery (標準平穩演唱)": "melodic singing tone"
        }
        v_expr = vexpr_map.get(vocal_expression, "expressive vocal contours, wide pitch dynamics, soaring high melody in choruses")

        # Human Phrasing Mapping
        phrasing_map = {
            "Organic Natural Breathing (真人呼吸感 + 靈活時值頓挫 - 推薦!)": "human-like vocal breathing, organic expressive timing, natural rubato phrasing",
            "Strict Metric Timing (嚴格規整節奏)": "steady rhythmic delivery"
        }
        v_phrase = phrasing_map.get(human_phrasing, "human-like vocal breathing, organic expressive timing")

        v_full = f"{v_base}, {v_expr}, {v_phrase}" if "no vocals" not in v_base else "instrumental only"

        prod_map = {
            "Clean Studio Mix (錄音室級清晰立體聲混音)": "crystal clear studio recording, balanced stereo mix",
            "Warm Analog Tape Vibe (溫暖類比膠片質感)": "warm vintage analog tape sound, cozy atmosphere",
            "Spacious Concert Reverb (開闊演唱會空間混響)": "spacious concert hall reverb, wide open soundstage",
            "Tight Dry Presence (緊湊貼耳無雜音錄音)": "tight dry studio production, intimate presence"
        }
        p_desc = prod_map.get(production_tag, "crystal clear studio recording, balanced stereo mix")

        dyn_map = {
            "Building Climax (主歌鋪墊，副歌與結尾爆發高潮)": "dynamic arrangement, building emotional climax in choruses",
            "Steady Smooth (全曲平穩流暢)": "smooth continuous dynamic flow",
            "Intimate Stripped-back (極簡貼耳純伴奏)": "intimate acoustic arrangement, stripped-back texture"
        }
        d_desc = dyn_map.get(dynamic_progression, "dynamic arrangement, building emotional climax in choruses")

        # Stable Attention Ordering: Genre -> BPM -> Key -> Vocal -> Instruments -> Rhythm -> Dynamics -> Production
        elements = [f"{final_genre}", f"{final_bpm:.0f} BPM", f"{detected_key}"]
        if v_full:
            elements.append(v_full)

        elements.extend([i_desc, r_desc, d_desc, p_desc])

        if extra_tags.strip():
            elements.append(extra_tags.strip())

        return ", ".join([e for e in elements if e]) + "."

    def structure_lyrics_advanced(self, raw_text_or_lines, target_duration, intro_tag, repeat_final_chorus=True, insert_instrumental_solo=True, include_bridge=True, outro_style="Slow Fade Out (慢速淡出結尾)", is_full_song=False):
        """Structures lyrics with modular feature switches and expressive section cues."""
        if isinstance(raw_text_or_lines, str):
            lines = raw_text_or_lines.splitlines()
        else:
            lines = raw_text_or_lines

        filter_keywords = [
            "作词", "作曲", "编曲", "制作人", "混音", "录音", "母带", "吉他", "贝斯", "鼓", 
            "字幕", "志愿者", "歌词提供", "produced by", "lyrics by", "arranged by", "acoustic guitar",
            "electric guitar", "guitar", "bass", "drums", "engineer", "keyboards", "programmed by",
            "synth", "music by", "mixed by", "recorded by", "soloist", "producer"
        ]
        
        cleaned = []
        for line in lines:
            txt = line.strip()
            txt = re.sub(r"\[\d{2}:\d{2}(?:\.\d+)?\]", "", txt).strip()
            if not txt:
                continue
            if any(k in txt.lower() for k in filter_keywords):
                continue
            if "–" in txt or "—" in txt or " - " in txt:
                if any(role in txt.lower() for role in ["guitar", "bass", "drums", "by", "vocal", "synth", "mix"]):
                    continue
            txt = re.sub(r"^\[.*?\]", "", txt).strip()
            if txt and len(txt) >= 2:
                cleaned.append(txt)
                
        if not cleaned:
            return "[verse 1]\n(vocal melody)\n\n[outro]\n(fades out softly)"

        merged_phrases = []
        i = 0
        while i < len(cleaned):
            curr = cleaned[i]
            if len(curr) <= 7 and i + 1 < len(cleaned):
                merged_phrases.append(f"{curr}，{cleaned[i+1]}")
                i += 2
            else:
                merged_phrases.append(curr)
                i += 1

        if "Direct Vocal Start" in intro_tag:
            intro_header = ""
        else:
            intro_header = "[intro]\n"

        # Outro instruction string
        if "Direct Stop" in outro_style:
            outro_cue = "(direct clean stop)"
        elif "Vocal Ad-lib" in outro_style:
            outro_cue = "(vocal ad-lib fading slowly)"
        else:
            outro_cue = "(fades out slowly)"

        # Full Song Mode (>= 220s or target_duration == 0)
        if is_full_song or target_duration >= 220.0:
            total_phrases = len(merged_phrases)
            chunk = max(3, total_phrases // 4)
            v1 = "\n".join(merged_phrases[:chunk])
            c1 = "\n".join(merged_phrases[chunk:chunk*2])
            v2 = "\n".join(merged_phrases[chunk*2:chunk*3])
            c2 = "\n".join(merged_phrases[chunk*3:])
            
            solo_section = "\n\n[instrumental solo]" if insert_instrumental_solo else ""
            bridge_section = (
                "\n\n[bridge]\n"
                "不要再为了他挣扎，不要再为他左牵右挂\n"
                "今后不管他爱不爱谁，快乐吗 都随他"
            ) if include_bridge else ""
            
            final_chorus_repeat = f"\n\n[chorus]\n(soaring high pitch, belt)\n{c2}" if repeat_final_chorus else ""
            
            res = (
                f"{intro_header}[verse 1]\n{v1}\n\n"
                f"[chorus]\n(expressive melodic lift)\n{c1}"
                f"{solo_section}\n\n"
                f"[verse 2]\n{v2}"
                f"{bridge_section}\n\n"
                f"[chorus]\n(powerful high notes)\n{c2}"
                f"{final_chorus_repeat}\n\n"
                f"[outro]\n"
                f"那是从来，都没有后路的悬崖\n"
                f"碎了心 也要放得下\n"
                f"{outro_cue}"
            )
            return res.strip()
        elif target_duration >= 90.0:
            # 90s - 180s (e.g. 120s): Budget is ~16-20 complete phrases (Verse 1 + Chorus 1 + Verse 2 + Chorus 2)
            max_phrases = min(len(merged_phrases), max(12, int(target_duration / 6.0)))
            selected = merged_phrases[:max_phrases]
            q = max(3, len(selected) // 4)
            v1 = "\n".join(selected[:q])
            c1 = "\n".join(selected[q:q*2])
            v2 = "\n".join(selected[q*2:q*3])
            c2 = "\n".join(selected[q*3:])
            
            final_chorus_repeat = f"\n\n[chorus]\n(soaring high pitch)\n{c2}" if repeat_final_chorus else ""
            
            res = (
                f"{intro_header}[verse 1]\n{v1}\n\n"
                f"[chorus]\n(expressive melodic lift)\n{c1}\n\n"
                f"[verse 2]\n{v2}\n\n"
                f"[chorus]\n{c2}"
                f"{final_chorus_repeat}\n\n"
                f"[outro]\n{outro_cue}"
            )
            return res.strip()
        else:
            # 30s - 80s (e.g. 60s): Budget is ~6-8 complete phrases (Verse 1 + Chorus 1)
            max_phrases = min(len(merged_phrases), max(4, int(target_duration / 7.0)))
            selected = merged_phrases[:max_phrases]
            mid = max(2, len(selected) // 2)
            v1 = "\n".join(selected[:mid])
            c1 = "\n".join(selected[mid:])
            return f"{intro_header}[verse 1]\n{v1}\n\n[chorus]\n(expressive melodic lift)\n{c1}\n\n[outro]\n{outro_cue}".strip()

    def analyze_and_generate(self, audio, auto_detect_genre=True, language="zh (Chinese)", use_cloud_lyrics=True, rhythm_tag="Steady Pop Drums (穩定流行鼓組律動)", instruments_tag="Grand Piano + Acoustic Guitar (鋼琴 + 原聲吉他掃弦)", vocal_tag="Emotional Male Vocal (深情磁性男聲)", vocal_expression="Wide Pitch Range & Soaring Climax (寬廣音域 + 副歌高亢爆發 - 告別唸經推薦!)", human_phrasing="Organic Natural Breathing (真人呼吸感 + 靈活時值頓挫 - 推薦!)", intro_tag="Short Intro ~5s (極簡短前奏 5秒內進主歌 - 推薦!)", production_tag="Clean Studio Mix (錄音室級清晰立體聲混音)", song_title_or_hint="雨一直下 - 张宇", target_generation_duration=0.0, repeat_final_chorus=True, insert_instrumental_solo=True, include_bridge=True, outro_style="Slow Fade Out (慢速淡出結尾)", dynamic_progression="Building Climax (主歌鋪墊，副歌與結尾爆發高潮)", genre_override="", extra_tags_custom="", lyrics_override="", bpm_override=0.0, whisper_model_size="base", transcribe_lyrics=True):
        waveform = audio["waveform"]
        sr = audio.get("sample_rate", 44100)

        if waveform.ndim == 3:
            if waveform.shape[1] <= 8:
                wav_mono = waveform[0].mean(dim=0).cpu().float().numpy()
            else:
                wav_mono = waveform[0].mean(dim=-1).cpu().float().numpy()
        elif waveform.ndim == 2:
            if waveform.shape[0] <= 8:
                wav_mono = waveform.mean(dim=0).cpu().float().numpy()
            else:
                wav_mono = waveform.mean(dim=-1).cpu().float().numpy()
        else:
            wav_mono = waveform.cpu().float().numpy()

        audio_duration = float(len(wav_mono)) / float(sr)
        final_target_duration = target_generation_duration if target_generation_duration > 0.0 else audio_duration
        is_full_song = (target_generation_duration == 0.0) or (final_target_duration >= 220.0)

        try:
            if sr != 22050:
                y_dsp = librosa.resample(wav_mono, orig_sr=sr, target_sr=22050)
                dsp_sr = 22050
            else:
                y_dsp = wav_mono
                dsp_sr = sr

            detected_bpm = self.estimate_robust_tempo(y_dsp, dsp_sr)

            chroma = librosa.feature.chroma_cqt(y=y_dsp, sr=dsp_sr)
            key_idx = int(np.argmax(np.mean(chroma, axis=1)))
            pitch_classes = ["C", "D flat", "D", "E flat", "E", "F", "F sharp", "G", "A flat", "A", "B flat", "B"]
            detected_key = f"{pitch_classes[key_idx]} major"
            
            auto_genre = "Mandopop Ballad, Contemporary Pop" if "zh" in language else "Contemporary Pop, Melodic Ballad"
        except Exception as e:
            print("[MiniMaxMusic3AudioAutoPrompter] DSP Analysis Warning:", e)
            detected_bpm = 75.0
            detected_key = "E major"
            auto_genre = "Mandopop Ballad, Contemporary Pop"

        final_bpm = bpm_override if bpm_override > 0.0 else round(detected_bpm, 1)

        if not auto_detect_genre and genre_override.strip():
            final_genre = genre_override.strip()
        elif genre_override.strip():
            final_genre = genre_override.strip()
        else:
            final_genre = auto_genre

        # Assemble Full MiniMax-Music-3 Caption from Tags
        full_caption = self.parse_tags_to_prompt(
            rhythm_tag, instruments_tag, vocal_tag, vocal_expression, human_phrasing, production_tag, dynamic_progression,
            final_genre, final_bpm, detected_key, extra_tags_custom
        )

        # Resolve Lyrics
        lyrics_source = "Local Whisper ASR"
        cloud_lrc = None
        if lyrics_override.strip():
            final_lyrics = lyrics_override.strip()
            lyrics_source = "Manual Override"
        elif use_cloud_lyrics and song_title_or_hint.strip():
            print("[MiniMaxMusic3AudioAutoPrompter] Fetching official cloud lyrics for:", song_title_or_hint.strip())
            t_name, a_name, cloud_lrc = self.fetch_online_lyrics(song_title_or_hint.strip())
            if cloud_lrc:
                final_lyrics = self.structure_lyrics_advanced(
                    cloud_lrc, final_target_duration, intro_tag,
                    repeat_final_chorus=repeat_final_chorus,
                    insert_instrumental_solo=insert_instrumental_solo,
                    include_bridge=include_bridge,
                    outro_style=outro_style,
                    is_full_song=is_full_song
                )
                lyrics_source = f"Cloud Official Database ({t_name} - {a_name})"

        if not lyrics_override.strip() and not cloud_lrc and transcribe_lyrics and len(wav_mono) > 0:
            try:
                lang_code = language.split()[0] if language != "auto" else None
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                whisper = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
                
                transcribe_dur = len(wav_mono) if is_full_song else min(len(wav_mono), int((final_target_duration + 30) * sr))
                wav_to_transcribe = wav_mono[:transcribe_dur]

                if sr != 16000:
                    y_whisper = librosa.resample(wav_to_transcribe, orig_sr=sr, target_sr=16000)
                else:
                    y_whisper = wav_to_transcribe

                transcribe_kwargs = {
                    "beam_size": 5,
                    "no_speech_threshold": 0.6,
                    "log_prob_threshold": -1.0,
                    "condition_on_previous_text": False
                }
                if lang_code:
                    transcribe_kwargs["language"] = lang_code
                if song_title_or_hint.strip():
                    transcribe_kwargs["initial_prompt"] = f"歌曲名稱是《{song_title_or_hint.strip()}》，標準歌詞："

                segments, info = whisper.transcribe(y_whisper, **transcribe_kwargs)
                raw_lines = [seg.text.strip() for seg in segments if seg.text.strip()]
                final_lyrics = self.structure_lyrics_advanced(
                    raw_lines, final_target_duration, intro_tag,
                    repeat_final_chorus=repeat_final_chorus,
                    insert_instrumental_solo=insert_instrumental_solo,
                    include_bridge=include_bridge,
                    outro_style=outro_style,
                    is_full_song=is_full_song
                )
                lyrics_source = f"Whisper {whisper_model_size} ASR"
                
                del whisper
                comfy.model_management.soft_empty_cache()
            except Exception as e:
                print("[MiniMaxMusic3AudioAutoPrompter] Transcription Warning:", e)
                final_lyrics = "[verse 1]\n(melodic rhythm)\n\n[outro]\n(fades out softly)"
        elif not lyrics_override.strip() and not cloud_lrc:
            final_lyrics = "[verse 1]\n(melodic rhythm)\n\n[outro]\n(fades out softly)"

        mode_name = f"整首完整歌曲 (Full Track, {final_target_duration:.0f}s)" if is_full_song else f"指定時長 ({final_target_duration:.0f}s)"

        summary_text = (
            f"🎵 [MiniMax Music 3 Tag-Based Analysis Report]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏷️ 歌曲/歌手: {song_title_or_hint.strip()}\n"
            f"📜 歌詞來源: {lyrics_source}\n"
            f"⏱️ 產出模式: {mode_name} (原音訊長度: {audio_duration:.2f}s)\n"
            f"🥁 節奏骨架: {rhythm_tag}\n"
            f"🎹 核心配器: {instruments_tag}\n"
            f"🎤 人聲唱腔: {vocal_tag}\n"
            f"📈 音域表現: {vocal_expression}\n"
            f"🫁 呼吸時值: {human_phrasing}\n"
            f"⏳ 前奏控制: {intro_tag}\n"
            f"🎛️ 混音質感: {production_tag}\n"
            f"🔥 結構特徵: 結尾副歌重唱={repeat_final_chorus} | 間奏獨奏={insert_instrumental_solo} | 橋段={include_bridge}\n"
            f"📝 生成標籤:\n{full_caption}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        print("=== MiniMaxMusic3AudioAutoPrompter Result ===")
        print(summary_text)

        return {
            "ui": {"text": [summary_text]},
            "result": (full_caption, final_lyrics, float(final_bpm), detected_key, float(final_target_duration), summary_text)
        }


class SaveMiniMaxMusic3RVQCache:
    """Saves RVQ semantic candidates to a compact PyTorch file (~280KB) for instant reuse."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "semantic_candidates": ("SEMANTIC_CANDIDATES",),
                "filename_prefix": ("STRING", {"default": "rvq_cache/reference_c0"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_cache"
    CATEGORY = "audio/minimax"

    def save_cache(self, semantic_candidates, filename_prefix="rvq_cache/reference_c0"):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        os.makedirs(full_output_folder, exist_ok=True)
        file = f"{filename}_{counter:05}_.pt"
        full_path = os.path.join(full_output_folder, file)

        if not isinstance(semantic_candidates, torch.Tensor):
            raise ValueError("semantic_candidates must be a PyTorch Tensor")

        # Save tensor on CPU
        payload = {
            "semantic_candidates": semantic_candidates.detach().cpu().long(),
            "frames": semantic_candidates.shape[0],
            "duration_seconds": float(semantic_candidates.shape[0]) / 25.0,
            "version": "v4"
        }
        torch.save(payload, full_path)
        print(f"[SaveMiniMaxMusic3RVQCache] Saved {semantic_candidates.shape[0]} frames ({payload['duration_seconds']:.2f}s) to: {full_path}")
        return {"ui": {"text": [f"Saved RVQ cache: {file} ({payload['duration_seconds']:.2f}s)"]}}


class LoadMiniMaxMusic3RVQCache:
    """Loads cached RVQ semantic candidates with 0-second latency, bypassing DAV & 169M encoder."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        cache_dir = os.path.join(output_dir, "rvq_cache")
        os.makedirs(cache_dir, exist_ok=True)
        files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")] if os.path.exists(cache_dir) else []
        if not files:
            files = ["none"]

        return {
            "required": {
                "cache_file": (sorted(files), {"tooltip": "Select cached .pt file from output/rvq_cache/"}),
            }
        }

    RETURN_TYPES = ("SEMANTIC_CANDIDATES", "FLOAT")
    RETURN_NAMES = ("semantic_candidates", "duration_seconds")
    FUNCTION = "load_cache"
    CATEGORY = "audio/minimax"

    def load_cache(self, cache_file):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        cache_path = os.path.join(output_dir, "rvq_cache", cache_file)

        if not os.path.exists(cache_path) or cache_file == "none":
            raise FileNotFoundError(f"RVQ cache file not found: {cache_path}. Please generate and save cache first.")

        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "semantic_candidates" in data:
            candidates = data["semantic_candidates"]
            duration = float(data.get("duration_seconds", candidates.shape[0] / 25.0))
        elif isinstance(data, torch.Tensor):
            candidates = data
            duration = float(candidates.shape[0] / 25.0)
        else:
            raise ValueError(f"Invalid RVQ cache format in {cache_path}")

        print(f"[LoadMiniMaxMusic3RVQCache] Loaded {candidates.shape[0]} frames ({duration:.2f}s) instantly from {cache_file}!")
        return (candidates, duration)


class MiniMaxMusic3TextEncodeWithCachedReference:
    """Directly encodes CLIP text with cached RVQ semantic candidates without loading DAV or 169M model."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "semantic_candidates": ("SEMANTIC_CANDIDATES",),
                "caption": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "lyrics": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 18446744073709551615}),
                "reference_interval": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "1 = Strict melody adherence (recommended for strong remix); 5 = Looser styling."}),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 100.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("conditioning", "seconds")
    FUNCTION = "encode"
    CATEGORY = "conditioning/minimax music"

    def encode(
        self,
        clip,
        semantic_candidates,
        caption,
        lyrics,
        seed,
        reference_interval=1,
        cfg_scale=1.2,
        top_k=50,
    ):
        frame_count = semantic_candidates.shape[0]
        tokens = clip.tokenize(
            caption,
            lyrics=lyrics,
            seed=seed,
            max_audio_frames=frame_count,
            cfg_scale=cfg_scale,
            top_k=top_k,
        )
        tokens["minimax_reference_semantic_candidates"] = semantic_candidates
        tokens["minimax_reference_interval"] = reference_interval
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        for cond in conditioning:
            hidden = cond[0]
            cond[1]["conditioning_scale"] = torch.ones(
                (hidden.shape[0], 1, 1),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        duration_seconds = float(frame_count) / 25.0
        return (conditioning, duration_seconds)


NODE_CLASS_MAPPINGS = {
    "MiniMaxMusic3AudioAutoPrompter": MiniMaxMusic3AudioAutoPrompter,
    "SaveMiniMaxMusic3RVQCache": SaveMiniMaxMusic3RVQCache,
    "LoadMiniMaxMusic3RVQCache": LoadMiniMaxMusic3RVQCache,
    "MiniMaxMusic3TextEncodeWithCachedReference": MiniMaxMusic3TextEncodeWithCachedReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxMusic3AudioAutoPrompter": "MiniMax Music 3 Audio Auto-Prompter 🎵",
    "SaveMiniMaxMusic3RVQCache": "Save MiniMax Music3 RVQ Cache 💾",
    "LoadMiniMaxMusic3RVQCache": "Load MiniMax Music3 RVQ Cache ⚡",
    "MiniMaxMusic3TextEncodeWithCachedReference": "MiniMax Music3 Text Encode (Cached RVQ) ⚡",
}

