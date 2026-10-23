# TRAINING.md — MeshTalk 语音 SOS 模型训练说明

本文件说明：用了哪些数据、每个参数是什么值以及为什么这样选、如何从零复现、训练结果图表和运行环境。
**所有参数都在 `configs/train.yaml` 里，代码只从这个文件读取**（`kws/cfg.py`），代码中没有其他"魔法数字"。

> **当前状态（2026-10-23）**：Step 2 未达标（NO-GO），已暂停等待决定，详见 `REPORT.md`。
> 已有结果图：`docs/kw_training_curves.png`（训练曲线）、`docs/kw_confusion_val.png`（混淆矩阵）、
> `docs/kw_hit_vs_fa_{val,test}.png`（命中率–误报率，设备原逻辑）、`docs/kw_hit_vs_fa_{val,test}_window_gate.png`（建议的门限修改）、
> `models/prelim_spk/spk_eer_val_*.png`（声纹 EER，初步）。复现命令、运行环境与耗时将在 Step 4 完成后补全。

---

## 1. 整体流程

```
01_download.py   公开数据集 -> data/raw            （流式下载、校验 MD5/SHA-256、只保留需要的片段）
03_tts.py        Kokoro 合成 help / help me / SOS 及合成负样本 -> data/cache/tts
02_prepare.py    解码、重采样到 16 kHz、找语音段、打标签、按说话人划分 -> data/cache
10_train_kw.py   关键词头（help / help me / sos / _other）-> models/kw.pt
11_train_spk.py  声纹头（只用真人）-> models/best.pt
20_eval.py       用 kws.detect.Detector（与设备逻辑一致）测命中率、每小时误报；按设备录入方式测声纹 EER
30_export.py     阈值 -> kws.export -> out/kws_model.c + out/kws_model_test.c -> kws.check_c
```

代码模块：

| 文件 | 内容 |
|---|---|
| `kws/features.py` | 固件前端的 Python 版（**未改动**），所有特征都由它计算 |
| `kws/model.py` | `KwsNet`（**未改动**，只通过参数改尺寸） |
| `kws/detect.py` | 设备判决逻辑（**未改动**），评估直接调用 |
| `kws/export.py` / `kws/check_c.py` | 导出与 C 对比（**未改动**） |
| `kws/cfg.py` | 读配置、固定随机种子、按说话人哈希划分 |
| `kws/audio.py` | 重采样、电平、语音段检测 |
| `kws/augment.py` | 数据增强（增益、噪声、混响、变速、变调、喊叫、高通） |
| `kws/dataset.py` | 生成 T 帧训练窗口（特征只用 `kws.features.logmel`） |
| `kws/evaluate.py` | 流式评估（Detector）、声纹录入/比对、EER |

## 2. 数据

| 类别 | 来源 | 许可 |
|---|---|---|
| help（真人） | MSWC 英文 `help` 全部 4433 条；LibriSpeech（带 MFA 词对齐）中所有 "help" | CC-BY 4.0 |
| help me（真人） | LibriSpeech 中真实的 "help me"（98 条）；同一说话人的 "help" + "me" 拼接（间隔 0–80 ms） | CC-BY 4.0 |
| SOS | **只有合成**：Kokoro-82M，22 个训练音色 + 音色混合，语速 0.8–1.3 | Apache-2.0 |
| help / help me 合成 | Kokoro，同上 | Apache-2.0 |
| _other | MSWC 近音词（hello, helped, held, health, self, hell, yelp, boss, loss, sauce …）与随机词；Speech Commands v2；LibriSpeech 单词、"me"、1.5 s 连续语音；Kokoro 合成负样本（"Hello"、"Hold me"、"S. O."、"Help desk" …）；纯噪声；静音 | CC-BY 4.0 / Apache-2.0 |
| 噪声 | MUSAN noise + music（30% 的音乐文件），ESC-50，Speech Commands 背景噪声 | CC-BY 4.0 / CC-BY-NC 3.0 |
| 误报测试 | LibriSpeech dev（10.5 h，用于选阈值）/ test（10.7 h，只用于报告）+ 测试噪声 3.7 h | CC-BY 4.0 |

为什么这样选：

- **MSWC 没有 `me` 和 `sos`**（MSWC 只收 ≥3 个字母的词），所以 "help me" 的真人数据来自 LibriSpeech：它带词级时间对齐，可以切出真实的 "help me"，也可以把同一个人的 "help" 和 "me" 拼起来（同一说话人，声音一致）。
- **SOS 没有公开真人录音**，只能合成。Kokoro 会把 "SOS" 读成 "sauce"，所以只用拼读写法（"S. O. S."、"S-O-S!"、"Ess, Oh, Ess!" …）。
- 合成语音**只用于关键词头，从不用于声纹头**；同时加入同样数量级的合成负样本，防止网络学成"合成声音 = 关键词"。
- 近音词（hello、helped、held、self、boss、sauce …）作为 _other，专门压低误报。
- 划分**按说话人**（说话人 ID 的 MD5 哈希 80/10/10），同一个人不会同时出现在训练和测试里；Kokoro 留出 2 个验证音色、4 个测试音色。

## 3. 参数说明（`configs/train.yaml`）

### 3.1 全局

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `seed` | 20261022 | 全局种子。每个训练窗口的随机数 = `Generator([seed, 标签, epoch, 序号])`，与 DataLoader 进程数无关，重跑得到相同数据 |
| `labels` | help, help me, sos, _other | 输出顺序 = 导出时的 `--labels`；`_` 开头表示"非关键词" |

### 3.2 数据选择（`sources`、`prepare`、`split`）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `mswc.keyword_words` | [help] | 全部保留 |
| `mswc.lookalike_words` | 31 个近音/易混词 | 每词最多 `lookalike_max_per_word`=1500 条（按哈希取），避免 "also"(3 万条) 这类词淹没其他 |
| `mswc.random_other_clips` | 30000 | 其他随机词，提供广泛的 _other |
| `mswc.spk_speakers / spk_min_clips / spk_clips_per_speaker` | 4000 / 25 / 25 | 声纹头需要"人多、每人多句"；优先选说过 help 的人 |
| `librispeech.train_subsets` | train-clean-100 + 360 | 约 460 h、1172 人；流式处理后删除原文件（磁盘只有约 30 GB） |
| `librispeech.context_s` | 0.30 | 切词时前后各留 0.3 s，放入窗口时更自然 |
| `librispeech.me_keep_frac` 等 | 0.25 / 0.5 / 0.025 / 0.15 | 按"话语+词位置"的哈希抽取一定比例，结果可复现，数量够用且不占太多磁盘 |
| `speech_commands.other_clips` | 15000 | _other 的另一个风格（短命令词、多种麦克风） |
| `esc50.test_fold` | 5 | 第 5 折只用于误报测试 |
| `musan.music_keep_frac / test_pct` | 0.30 / 25 | 磁盘限制只留 30% 音乐；25% 文件留作测试 |
| `prepare.min_rms_db` | -60 | 去掉几乎无声的坏片段 |
| `prepare.span_rel_db` | -35 | "语音段" = 比最响帧低 35 dB 以内的部分，用于保证关键词完整落在窗口内 |
| `prepare.spk_min_clips` | 10 | 声纹训练只用至少 10 句的真人说话人 |
| `split.val_pct / test_pct` | 10 / 10 | 按说话人哈希划分 |

### 3.3 TTS（`tts`）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `voice_prefixes` | af_, am_, bf_, bm_ | Kokoro v1.0 的全部 28 个英语音色（美/英，男/女） |
| `test_voices / val_voices` | 4 / 2 个 | 留出音色，测试 SOS 时音色从未见过 |
| `blend_prob` | 0.35 | 35% 的训练样本用两个同口音音色按 0.3–0.7 混合，产生"新"音色 |
| `speed` | 0.8–1.3 | 慢速到快速说话 |
| `clips_per_class` | help 2500, help me 2500, sos 3000, _other 5000 | SOS 只有合成数据，所以多一些；合成负样本最多，防止"合成 = 关键词" |
| `texts` | 见配置 | 平静和急促写法（"Help!"、"HELP!"、"S. O. S."）；去掉了超过窗口长度的长句 |

### 3.4 窗口（`window`）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `T` | **81**（1.62 s） | 默认 61（1.22 s）放不下慢速拼读的 "S. O. S."（合成数据中位 0.95 s，最长 1.8 s）。CONTRACT 允许 T ≤ 101。代价：每次推理约 6.9 M 乘加（默认 5.2 M，+33%），T=81 时仍只有 2.4% 的 SOS 样本过长被跳过 |
| `margin_frames` | 2 | 关键词距窗口两端至少 2 帧（40 ms），保证"完整在窗口内" |

### 3.5 数据增强（`augment`，`kws/augment.py`）

处理顺序与真实声学链一致：说话人（变速/变调）→ 音量 → 喊叫 → 放入窗口（+ 背景人声）→ 房间混响 → 环境噪声 + 麦克风底噪 → 麦克风高通 → int16 量化（含削波）→ `logmel`。

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `ref_level_dbfs`, `gain_db` | -25, [-20, +6] | 语音 RMS 在 -45…-19 dBFS 之间：覆盖远处小声到近处大声；INMP441 无 AGC，所以电平范围要宽 |
| `p_noise`, `snr_db` | 0.85, [0, 20] dB | 85% 的窗口加噪声，信噪比 0–20 dB（0 dB 很嘈杂，求救场景常见） |
| `noise_floor_dbfs` | [-75, -55] | 每个窗口都加麦克风白噪声底噪 |
| `p_reverb`, `rt60_s`, `rir_len_s`, `reverb_wet` | 0.4, 0.15–0.8 s, 0.5 s, 0.3–1.0 | 合成房间冲激响应（直达声 + 指数衰减噪声），模拟室内/走廊 |
| `p_speed`, `speed` | 0.5, 0.9–1.1 | 重采样变速（语速与音高一起变） |
| `p_pitch`, `pitch_semitones` | 0.3, ±2 半音 | 相位声码器变调（语速不变），模拟不同嗓音 |
| `p_shout`, `shout_gain_db` | 0.15, 6–18 dB | 喊叫：增益后 tanh 软削波，模拟大喊和麦克风过载 |
| `p_highpass`, `highpass_hz` | 0.7, 60–120 Hz | 二阶巴特沃斯高通，模拟麦克风和外壳低频衰减 |
| `p_context`, `context_level_db` | 0.25, -12…0 dB | 关键词前或后有别人的说话声（不重叠），让模型在连续说话中也能找到关键词 |

### 3.6 网络（`net`）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `c1, c2, c3, hid` | 16, 32, 48, 64 | 默认尺寸。层类型固定（CONTRACT 2），T 已经增大了 33% 的计算量，通道数不再加大 |
| `spk_dim` | 32 | 声纹向量维度（≤64） |

### 3.7 关键词训练（`train_kw`）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `epochs`, `windows_per_epoch`, `batch` | 40, 48000, 256 | 每轮的窗口都是新随机生成的（增强不同），40 轮共 192 万个窗口；CPU 上约 140 s/轮 |
| `class_mix` | 0.2 / 0.2 / 0.2 / 0.4 | 采样器按类别平衡；_other 多一些，因为它的种类最多、误报最要紧 |
| `other_mix` | 见配置 | _other 内部各来源的比例：近音词 17%、连续语音 18%、合成负样本 12%、噪声 10% … |
| `real_vs_tts` | help 0.75, help me 0.55, sos 0 | 有真人数据就以真人为主；SOS 只能 100% 合成 |
| `help_me_sources` | 真实 0.4 / 拼接 0.6 | 真实 "help me" 只有 79 条训练样本，拼接样本补充多样性 |
| `optimizer`, `lr`, `weight_decay` | AdamW, 0.003, 0.001 | 小网络常用设置 |
| `schedule` | OneCycle：5% 线性预热，余弦下降到 lr/100 | 前期大步长快速收敛，后期小步长让 BatchNorm 统计稳定 |
| `loss`, `class_weights`, `label_smoothing` | 交叉熵, [1,1,1,1], 0.05 | 采样已平衡，所以权重相等；标签平滑让输出概率不过度自信，便于设阈值 |
| `val_windows_per_class` | 1500 | 验证集每类 1500 个固定窗口（验证说话人/音色） |
| `norm_windows` | 4000 | 用 4000 个训练窗口计算 `in_mean / in_std`（CONTRACT 要求从训练特征设置） |
| 模型选择 | 验证集平衡准确率最高的一轮 | |

### 3.8 声纹训练（`train_spk`）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `mode` | frozen_trunk | 冻结主干和关键词头，只训练 `spk` 线性层，关键词性能保持不变 |
| `loss`, `aam_margin`, `aam_scale` | AAM-softmax, 0.2, 30 | 说话人识别常用的角度间隔损失，m=0.2、s=30 是常见取值 |
| `epochs`, `windows_per_epoch`, `batch`, `lr` | 30, 40000, 256, 0.003 | |
| `augment` | 噪声 0.6（5–25 dB）、混响 0.3、高通 0.7、增益 -15…+6 dB | **不做变速/变调/喊叫**：这些会改变"是谁的声音" |
| `enrol_takes`, `enrol_classes`, `take_s` | 5, help/help me/sos/help/help me, 2 s | 与设备录入完全一致 |
| `trial_speakers`, `test_clips_*`, `impostors_per_test` | 300, 3–8, 30 | EER 试验规模 |

### 3.9 评估（`eval`）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `context_s` | 2.0 | 每个测试关键词前后各 2 s 背景，整段逐帧送入 Detector |
| `test_snr_db`, `test_level_dbfs` | 5–20 dB, -38…-15 dBFS | "noisy" 条件；"clean" 只有 -70 dBFS 底噪 |
| `hit_after_ms` | 1500 | 关键词结束后 1.5 s 内的命中都算（窗口 1.62 s + 平滑 3 次运行） |
| `keyword_guard_s` | 1.5 | LibriSpeech 里真实说出的 "help" 附近的命中不算误报 |
| `fa_noise_mix_prob`, `fa_noise_snr_db` | 0.5, 5–20 dB | 一半的连续语音再叠加噪声 |
| `thr_grid` | 0.30…0.99 | 扫描阈值画命中率–误报曲线 |

### 3.10 导出（`export`）

<!-- EXPORT -->
