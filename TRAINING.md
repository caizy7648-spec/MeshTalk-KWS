# TRAINING.md — MeshTalk 语音 SOS 模型训练说明（第二轮，固件 v1.4.1）

本文件说明：数据来源与数量、**每个参数的值和选择理由**、从零复现的完整命令、运行环境和每一步耗时、结果图。
所有参数都在两个配置文件里，代码只从这里读（`kws/cfg.py`、`spk/data.py`）：

- `configs/train.yaml`：关键词模型（阶段 A）——数据、合成语音、窗口、增强、网络、三轮训练、难负样本挖掘、评估、阈值、导出；
- `configs/spk.yaml`：独立声纹模型（阶段 B）。

所有随机性都由种子决定：每个训练窗口的随机数 = `numpy.random.default_rng([seed, 标签, epoch, 序号])`，与 DataLoader
进程数无关；数据选择（抽样、划分、挖掘）全部按字符串的 MD5 哈希排序/取模，在任何机器上都相同。

结果数字见 `REPORT.md`（这里只列关键数字）。

---

## 0. 结果一览

| | 指标（test，与设备一致的 int8 模型 + `kws.detect.Detector`） |
|---|---|
| 关键词模型 v1（中档） | 喊 6 次触发率：真人 help/help me 混合 **92.5%**，Kokoro 混合 99.5%，只喊 SOS 97%，真人只喊 help 88%；单次误报 6.9 次/h；误触发 SOS 1.2 次/24 h（dev+test 合计 0.99 次/24 h） |
| 关键词模型大小 | 6.92 M 乘加/次，权重 22.9 KB，工作内存 88 KB |
| 声纹模型 | 单句 EER（10 段录入）干净 6.8% / 噪声 9.8% / 喊叫 7.2%；系统：本人喊 6 次 75.5% / 58.5% / 78%，别人 2.5% / 1.5% / 2%（**未达标**，建议保持"谁喊都触发"） |
| 声纹模型大小 | 12.8 M 乘加/次，权重 229 KB，工作内存 177 KB |

图（`docs/`）：

| 图 | 内容 |
|---|---|
| `kw_r1_training_curves.png`、`kw_r2_…`、`kw_r3_…` | 三轮关键词训练的 loss / 准确率曲线 |
| `kw_r1_confusion_val.png`、`kw_r2_…`、`kw_r3_…` | 验证集混淆矩阵（单个窗口） |
| `sys_trigger_vs_fa_val_sweep_r2.png`、`…_r3.png` | dev 上扫描 thr × cls_off：喊 6 次触发率 vs 每小时误报 / 每 24 h 误触发 SOS |
| `sys_trigger_vs_fa_val.png`、`sys_trigger_vs_fa_test.png` | 导出的三档灵敏度（int8 模型）在 dev / test 上的触发率–误报 |
| `spk_training_curves.png` | 声纹模型训练曲线（loss、验证 EER） |
| `spk_eer_val.png`、`spk_eer_test.png` | 声纹 EER 曲线（拒识/误识 vs 阈值）和系统级"本人 vs 别人"触发率曲线 |
| `round1/` | 第一轮（固件 v1.4）的命中率–误报曲线，仅存档 |

---

## 1. 流程和代码

```
scripts/01_download.py   公开数据 -> data/raw          流式下载（断线续传）、校验 MD5/SHA-256、只留需要的片段
scripts/03_tts.py        Kokoro 合成 help / help me / SOS 和合成负样本 -> data/cache/tts
scripts/02_prepare.py    解码、16 kHz、找语音段、打标签、按说话人划分 -> data/cache（clips / 噪声库 / 误报语音+词对齐）
scripts/10_train_kw.py   关键词模型：--stage r1（从零）/ r2 / r3（加难负样本微调）
scripts/12_mine.py       难负样本挖掘（用上一轮模型扫训练说话人的连续语音和噪声）；同时补充留出说话人的误报测试语音
scripts/20_eval.py       系统级评估（Detector：喊 4/6/8 次的触发率、每小时误报、每 24 h 误触发 SOS、误报来源词）
scripts/30_export.py     选定的一轮 -> models/best.pt -> kws.export（thr、cls_off）-> kws.check_c；生成 int8 等价模型
scripts/40_train_spk.py  独立声纹模型（只用真人）
scripts/41_eval_spk.py   声纹：设备录入协议的单句 EER + 与关键词模型组合的系统级指标；在 val 上选阈值
scripts/42_export_spk.py 声纹模型 -> out/spk_model.c 等 -> spk.check_c（C == Python）
```

| 模块 | 内容 |
|---|---|
| `kws/features.py`、`kws/model.py`、`kws/detect.py`、`kws/export.py`、`kws/check_c.py`、`c_ref/` | 固件对应部分，**未改动**（`detect.py`/`export.py`/`c_ref/` 为 v1.4.1 原样） |
| `kws/cfg.py` | 读配置、种子、按说话人哈希划分 train/val/test |
| `kws/audio.py` | 重采样、电平、语音段检测 |
| `kws/augment.py` | 增强：增益、噪声、麦克风底噪、合成混响、变速、变调（相位声码器）、喊叫（tanh 软削波）、高通 |
| `kws/dataset.py` | 训练窗口：按类别/来源抽样 → 增强 → 放入 T 帧窗口 → `kws.features.logmel` |
| `kws/evaluate.py` | 流式评估：批量算 CNN 输出，交给原版 `Detector` 逐帧判定（`check=True` 校验窗口一致） |
| `spk/model.py`、`spk/export.py`、`spk/c/`、`spk/check_c.py` | 声纹网络、int8 导出 + numpy 参考实现、C 引擎和 C 对比测试 |
| `spk/data.py`、`spk/evaluate.py` | 声纹训练数据、设备录入协议评估 |

---

## 2. 数据（来源、许可、数量）

| 数据 | 用途 | 许可 | 数量 |
|---|---|---|---|
| MSWC 英文（GCS，35 GB 流式，只留选中的 14.4 万段） | 真人 help；近音词（31 个）；随机词；声纹 | CC-BY 4.0 | help 4433 段 / 3026 人 |
| LibriSpeech + MFA 词对齐（HF `gilkeyio/librispeech-alignments`） | 真人 help、真实 "help me"、同一人 help+me 拼接、"me"、近音词、连续语音块；误报测试语音；声纹 | CC-BY 4.0 | train-clean-100/360、train-other-500 流式切取；dev/test 整体保留 |
| Speech Commands v2（GCS 镜像） | _other 命令词、背景噪声 | CC-BY 4.0 | 15 000 段 |
| ESC-50（HF 镜像 `ashraq/esc50`） | 噪声（1–4 折训练，第 5 折测试） | CC-BY-NC 3.0 | 2000 段 |
| MUSAN noise + 30% music | 噪声（按哈希 75/25 训练/测试） | CC-BY 4.0（逐文件） | — |
| Kokoro-82M（GitHub release） | 合成 help / help me / SOS 和合成负样本（**只用于关键词模型**） | Apache-2.0 | 16 000 段 |

划分后的数量（`data/cache/counts.json`）：

| 类别 | train | val | test |
|---|---|---|---|
| help | 8263（MSWC 3501、LibriSpeech 2262、Kokoro 2500） | 1080 | 1012 |
| help me | 2701（LibriSpeech 真实 201 + 同一人拼接，Kokoro 2500） | 328 | 334 |
| sos | 3000（**只有合成**） | 300 | 300 |
| _other | 220 310 + 难负样本 r2 23 206、r3 8932 | 28 377 | 27 469 |

- 噪声：训练 17.3 h，测试 3.7 h（按文件哈希一半给 dev、一半给 test）。
- 误报测试语音（只来自从未训练过的说话人）：dev = LibriSpeech dev 10.5 h + 留出说话人 20.0 h（196 人）；test = LibriSpeech test 10.7 h + 留出说话人 27.9 h（207 人）；一半叠加噪声。
- 声纹训练：5224 个真人说话人（每人最多 20 段），val 662 人，test 655 人。

为什么这样选：
- **MSWC 没有 "me" 和 "sos"**（只收 ≥3 个字母的词），所以 "help me" 的真人数据来自 LibriSpeech（带词对齐）；**SOS 没有公开的真人录音**，只能合成（Kokoro 会把 "SOS" 读成 "sauce"，所以只用 "S. O. S." 等拼读写法）。
- 合成语音同时生成合成负样本，防止网络学成"合成音色 = 关键词"。
- VoxCeleb（需要申请凭据）、Common Voice（HF 需要登录）、VCTK（11 GB 单个压缩包，云端磁盘放不下）都试过，不可用。

---

## 3. 参数说明 —— `configs/train.yaml`（关键词模型）

### 3.1 全局、数据选择、划分

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `seed` | 20261022 | 全局种子 |
| `labels` | help, help me, sos, _other | 输出顺序 = 导出的 `--labels` |
| `sources.mswc.lookalike_words` / `lookalike_max_per_word` | 31 个 / 1500 | 近音、易混词（hello, helped, held, health, self, hell, yelp, boss, sauce…）做 _other，每词最多 1500 段，防止 "also"（3 万段）淹没其他 |
| `sources.mswc.random_other_clips` | 30 000 | 其他随机词 |
| `sources.mswc.spk_speakers` / `spk_min_clips` / `spk_clips_per_speaker` | 4000 / 25 / 25 | 声纹需要"人多、每人多句"，优先选说过 help 的人 |
| `sources.librispeech.train_subsets` | clean-100、clean-360、other-500 | 约 960 h、2338 人，流式处理后删除原文件 |
| `…context_s` | 0.30 | 切词时前后各留 0.3 s |
| `…me_keep_frac` / `lookalike_keep_frac` / `spk_word_keep_frac` / `other_chunks_per_utt` | 0.25 / 0.5 / 0.025 / 0.15 | 按"话语+词位置"的哈希取一定比例：数量够用、可复现、磁盘放得下 |
| `…subset_overrides.train_other_500` | 语音块 0.25、2.0 s | 第一轮 v2 加入：主要为 _other 提供更多连续语音（误报来源） |
| `sources.speech_commands.other_clips` | 15 000 | 另一种风格的短词 |
| `sources.esc50.test_fold` | 5 | 第 5 折只用于测试 |
| `sources.musan.music_keep_frac` / `test_pct` | 0.30 / 25 | 磁盘限制只留 30% 音乐；25% 文件留作测试 |
| `prepare.min_rms_db` / `span_rel_db` / `spk_min_clips` | −60 / −35 / 10 | 去掉无声片段；"语音段" = 比最响帧低 35 dB 以内；声纹只用 ≥10 段的真人 |
| `prepare.fa_subsets` | dev: dev_clean+dev_other；test: test_clean+test_other | 误报测试语音 |
| `split.val_pct` / `test_pct` | 10 / 10 | 按说话人 MD5 哈希划分（Kokoro：22 个训练音色，2 个验证，4 个测试） |

### 3.2 合成语音 `tts`

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `voice_prefixes` | af_, am_, bf_, bm_ | Kokoro v1.0 全部 28 个英语音色 |
| `test_voices` / `val_voices` | 4 / 2 个 | 留出音色，测试用的音色从未训练过 |
| `blend_prob` | 0.35 | 35% 的训练样本用两个同口音音色按 0.3–0.7 混合，得到"新"音色 |
| `speed` | 0.8–1.3 | 慢到快 |
| `clips_per_class` | help 2500、help me 2500、sos 3000、_other 5000 | SOS 只有合成数据所以多一些；合成负样本最多 |
| `texts` | 见配置 | 平静和急促写法；SOS 只用拼读写法；去掉超过窗口的长句 |

### 3.3 窗口 `window`

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `T` | **81**（1.62 s） | 默认 61（1.22 s）放不下慢速拼读的 "S. O. S."（合成中位 0.95 s、最长 1.8 s）。CONTRACT 允许 ≤101；代价 6.9 M 乘加（默认 5.2 M），在 15 M 预算内 |
| `margin_frames` | 2 | 关键词距窗口两端至少 40 ms |
| `p_right_edge` / `right_edge_s` | 0.35 / 0.30 s | 35% 的关键词窗口让词落在窗口最后 0.3 s（第一轮为旧门限加入；新门限下也保留，帮助模型在词刚说完时就响应） |

### 3.4 难负样本挖掘 `mine` / `mine_r3`（`12_mine.py`）

| 参数 | 值（r2 / r3） | 说明 / 理由 |
|---|---|---|
| `model` | kw_r1.pt / kw_r2.pt | 用上一轮模型找它自己的误报 |
| `shards` / `skip_shards` | clean-360 前 24 + other-500 前 16 / 之后的 20 + 20 | 约 300 h / 另外 300 h（r3 换新分片：新的人和文本）；只用训练划分的说话人（约 242 h） |
| `threshold` | 0.5 / 0.4 | 窗口的最大关键词概率超过它就算误报；r3 更低，因为最终阈值在 0.5 附近 |
| `min_gap_s` / `clip_s` / `clip_after_s` | 1.0 / 2.0 / 0.2 | 每秒最多一个；保存 2 s，结束于触发窗口之后 0.2 s |
| `per_shard_max` | 500 | 每片取最自信的 500 个（多样性、磁盘） |
| `noise_per_file_max` | 50 / 200 | 训练噪声库也挖；第二轮发现噪声误报偏多，r3 提高到 200 |
| `fa_extra_hours` | test 30、dev 20 / 0 | 同一次流式处理中，把**测试/验证划分说话人**的完整语音存为扩展误报测试集（与模型无关） |
| 结果 | r2：2 万 + 噪声 3206；r3：7946 + 噪声 986 | r2 模型的误报明显变少 |

### 3.5 数据增强 `augment`

处理顺序与真实声学链一致：说话人（变速/变调）→ 音量 → 喊叫 → 放入窗口（+ 背景人声）→ 混响 → 环境噪声 + 底噪 → 高通 → int16 量化（含削波）→ `logmel`。

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `ref_level_dbfs` / `gain_db` | −25 / −20…+6 dB | 语音 RMS −45…−19 dBFS：远处小声到近处大声（INMP441 无 AGC） |
| `p_noise` / `snr_db` | 0.85 / 0–20 dB | 求救场景常有噪声，0 dB 很嘈杂 |
| `noise_floor_dbfs` | −75…−55 | 每个窗口都加麦克风底噪 |
| `p_reverb` / `rt60_s` / `rir_len_s` / `reverb_wet` | 0.4 / 0.15–0.8 s / 0.5 s / 0.3–1.0 | 合成房间冲激响应（直达声 + 指数衰减噪声） |
| `p_speed` / `speed` | 0.5 / 0.9–1.1 | 重采样（语速和音高一起变） |
| `p_pitch` / `pitch_semitones` | 0.3 / ±2 | 相位声码器变调，语速不变 |
| `p_shout` / `shout_gain_db` | 0.15 / 6–18 dB | 喊叫：增益后 tanh 软削波 |
| `p_highpass` / `highpass_hz` | 0.7 / 60–120 Hz | 二阶巴特沃斯高通（麦克风、外壳） |
| `p_context` / `context_level_db` | 0.25 / −12…0 dB | 关键词前后有别人的说话声（不重叠） |

### 3.6 网络 `net`

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `c1, c2, c3, hid` | 16, 32, 48, 64 | 默认尺寸（你要求暂不加大通道）；层类型固定 |
| `spk_dim` | 0 | v1.4.1：声纹改为独立模型（第一轮训练时有一个未使用的 32 维头，导出时去掉，不影响关键词权重） |

### 3.7 三轮训练 `train_kw`、`train_kw_r2`、`train_kw_r3`

| 参数 | r1（`train_kw`） | r2 | r3 | 说明 / 理由 |
|---|---|---|---|---|
| `init` | — | kw_r1.pt | kw_r2.pt | r2/r3 在上一轮基础上微调 |
| `epochs` × `windows_per_epoch` | 50 × 48 000 | 25 × 48 000 | 20 × 48 000 | 每轮窗口都是新随机生成的 |
| `batch` | 256 | 256 | 256 | |
| `lr`（OneCycle 峰值） | 0.002 | 0.001 | 0.0007 | 微调用更小步长；5% 线性预热，余弦降到 lr/100（`div_factor` 25、`final_div_factor` 4） |
| `optimizer` / `weight_decay` | AdamW / 0.001 | 同 | 同 | |
| `loss` / `class_weights` / `label_smoothing` | 交叉熵 / 1,1,1,1 / 0.05 | 同 | 同 | 采样已平衡；标签平滑让概率不过度自信，便于设阈值 |
| `class_mix` | 0.17 / 0.16 / 0.17 / 0.5 | 同 | 同 | _other 占一半：误报最要紧 |
| `other_mix` 主要项 | 连续语音 0.30、近音词 0.14、噪声 0.13、合成负样本 0.12… | 难负样本 0.25 + 噪声难负样本 0.05，连续语音 0.17… | r3 难负样本 0.20 + 0.07，r2 难负样本 0.10 + 0.03… | 见配置 |
| `max_other_per_group` | — | 20 000 | 15 000 | 每个 _other 来源按哈希固定取子集：缓存超过内存会导致磁盘抖动、训练慢 10 倍 |
| `real_vs_tts` | help 0.75、help me 0.55、sos 0 | 同 | 同 | 有真人就以真人为主 |
| `help_me_sources` / `join_gap_ms` | 真实 0.4 / 拼接 0.6；0–80 ms | 同 | 同 | 真实 "help me" 只有 201 段 |
| `select` | val_bal_acc | last | last | 单窗口准确率不反映误报，微调轮取最后一轮 |
| `norm_windows` | 4000 | （沿用 r1 的 in_mean/in_std） | 同 | CONTRACT 要求从训练特征设置归一化 |
| `workers` / `torch_threads` | 4 / 2 | | | 只影响速度，不影响结果 |

### 3.8 系统级评估 `eval`、`eval_sys`

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `eval_sys.sets` | real_help、real_mixed、tts_mixed、tts_help、tts_sos | 一个人喊：真人只喊 help（MSWC+LibriSpeech 测试说话人）、真人 help/help me 混合（LibriSpeech）、Kokoro 三词混合、只喊 help、只喊 SOS |
| `calls` / `window_s` / `lead_s` / `tail_s` | 4, 6, 8 / 20 s / 0.5 s / 2 s | 20 s 内喊 N 次，每次放在自己的时间槽里随机位置 |
| `scenarios` | 100 | 每个（集合 × 次数 × 干净/噪声）100 个场景 |
| `call_gain_db` / `p_speed` / `speed` | ±3 dB / 0.5 / 0.95–1.05 | 同一个人每次喊的自然变化 |
| `eval.test_level_dbfs` / `test_snr_db` / `noise_floor_dbfs` | −38…−15 / 5–20 dB / −70 | "噪声"条件的 SNR；"干净"只有底噪 |
| `eval.fa_noise_mix_prob` / `fa_noise_snr_db` / `noise_level_dbfs` | 0.5 / 5–20 dB / −45…−15 | 一半的连续语音叠加噪声；纯噪声流的电平 |
| `eval.keyword_guard_s` / `hit_after_ms` | 1.5 s / 1500 ms | 误报语音里真实说出的 "help" 附近的命中不算误报；关键词结束后 1.5 s 内的命中算这次喊叫 |
| `blame_s` / `blame_words` / `blame_top` | 1.0 / 4 / 30 | 记录误报前 1 s 说的词 |
| `sweep_thr` × `sweep_off` | 10 × (4×2×2) | dev 上扫 160 组 thr × cls_off |
| `target_trigger_6` / `target_false_sos_24h` | 0.90 / 1.0 | 你的目标 |

### 3.9 导出 `export`

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `model` | kw_r3.pt | 第三轮（误报最少） |
| `thr` | **0.65 / 0.55 / 0.50**（低/中/高） | 在 dev 上选：中档 = 误触发 SOS 为 0 时喊 6 次触发率最高；低档单次误报约 2/h；高档更灵敏（dev 上约 2–3 次误触发 SOS/24 h） |
| `cls_off` | help −0.10、help me 0、sos +0.05 | help 是短词、最难，阈值低一些；SOS 只有合成数据，偏保守 |
| `hits` / `hits_ms` | 3 / 20 000 | 设备规则 |
| `name` | kws-20261023-r3 | |

**int8 等价模型**：`30_export.py` 同时生成 `models/best_int8sim.pt`（BN 折叠、权重 = int8 × scale），与 `kws.export.reference`（= C 运算）最大差 8.5e-7。REPORT 里的最终数字都用它测，所以与设备上的 int8 推理一致（浮点模型的概率与 int8 平均差 0.016）。

---

## 4. 参数说明 —— `configs/spk.yaml`（独立声纹模型）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `seed` | 20261023 | |
| `data.max_clips_per_speaker` | 20 | 训练时每人按哈希取 20 段：说话人多比每人段数多更重要，也让数据留在内存里 |
| `data.exclude_groups_eval` | libri_chunk | 录入/测试用单个词（像设备上的一次喊叫），不用连续语音块 |
| `net.frames` | 81 | = 关键词窗口 T：固件把触发时的窗口交给 `verify()` |
| `net.channels` / `kernels` / `dilations` | 128,128,128,256 / 5,3,3,1 / 1,2,3,1 | 小型 TDNN，感受野 15 帧（0.3 s）；均值+标准差统计池化保留音色信息 |
| `net.dim` | 128 | 声纹向量长度（≤256） |
| `augment` | 噪声 0.7（0–25 dB）、混响 0.4、增益 −20…+6、**喊叫 0.3（6–18 dB）**、高通 0.7；**无变速/变调** | 变速变调会改变"是谁的声音"；紧急时会喊，所以喊叫增强比关键词模型更多 |
| `train.epochs` × `windows_per_epoch` × `batch` | 40 × 60 000 × 256 | |
| `train.lr` / `weight_decay` / 调度 | 0.002 / 0.0005 / OneCycle（同关键词） | |
| `train.loss` / `aam_margin` / `aam_scale` | AAM-softmax / 0.2 / 30 | 说话人识别常用的角度间隔损失和常见取值 |
| `train.val_speakers` | 150 | 每轮用 150 个验证说话人（5 段录入、干净）算 EER，取最好的一轮 |
| `eval.enrol_takes` | 5, 10 | 设备录入 3–10 段；你计划最多 10 段 |
| `eval.enrol_classes` | help, help me, sos | 第 i 段录入对应第 i%3 类（设备轮流提示） |
| `eval.enrol_level_dbfs` | −35…−20 | 录入时安静、正常音量 |
| `eval.test_snr_db` / `shout_gain_db` | 5–20 dB / 6–18 dB | 噪声和喊叫条件 |
| `eval.test_takes_max` / `impostors` / `speakers` | 10 / 30 / 400 | 试验规模 |
| `eval.system_*` | 200 场景、喊 6 次、10 段录入 | 系统级：本人/别人各喊 6 次 |
| `eval.thr_grid` / `target_owner` / `target_other` | 0–0.9 步长 0.025 / 0.90 / 0.03 | 在 val 上扫阈值，选"别人 ≤3% 时本人最高" |
| `export.threshold` / `enrol_takes` / `name` | 0.60 / 10 / spk-20261023 | |

---

## 5. 从零复现

需要约 40 GB 磁盘（下载时流式处理，最终保留约 12 GB 原始片段 + 约 25 GB 缓存）、约 16 GB 内存、4 核 CPU 即可（无需 GPU）。

```bash
git clone https://github.com/caizy7648-spec/MeshTalk-KWS.git && cd MeshTalk-KWS
python3 -m venv venv && . venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m kws.check_c                                        # 19 checks, 0 failed

# 阶段 A：关键词模型
python scripts/01_download.py                                # 公开数据（约 75 GB 流式，校验和见 data/raw/MANIFEST.json）
python scripts/03_tts.py --procs 4                           # Kokoro 合成
python scripts/02_prepare.py                                 # -> data/cache
python scripts/10_train_kw.py --stage r1                     # 第一轮 -> models/kw_r1.pt
python scripts/12_mine.py                                    # 用 r1 挖难负样本 + 扩展误报测试语音
python scripts/10_train_kw.py --stage r2                     # -> models/kw_r2.pt
python scripts/12_mine.py --stage r3                         # 用 r2 再挖
python scripts/10_train_kw.py --stage r3                     # -> models/kw_r3.pt
python scripts/20_eval.py --model models/kw_r3.pt --split val --sweep     # 在 dev 上选 thr / cls_off（写入 configs/train.yaml export:）
python scripts/30_export.py                                  # -> models/best.pt, models/best_int8sim.pt, out/kws_model*.c, check_c
python scripts/20_eval.py --model models/best_int8sim.pt --split val      # 三档复核
python scripts/20_eval.py --model models/best_int8sim.pt --split test     # 最终数字 -> models/eval_sys_test.json

# 阶段 B：独立声纹模型
python scripts/40_train_spk.py                               # -> models/spk_best.pt
python scripts/41_eval_spk.py --split val  --kw-model models/best_int8sim.pt   # 选阈值（写入 configs/spk.yaml export.threshold）
python scripts/41_eval_spk.py --split test --kw-model models/best_int8sim.pt   # 最终数字
python scripts/42_export_spk.py                              # -> out/spk_*.c/.h, spk.check_c
```

说明：
- 同一台机器上重跑得到相同的数据和模型；不同 CPU 的浮点运算细节可能略有差别（权重有极小差异，指标应在 ±1% 内）。
- `spk_id()` 每次导出都会变（固件据此丢弃旧的录入），这是 CONTRACT_SPK 要求的。
- 云端环境磁盘只有约 30 GB，所以我在缓存生成后删除了部分原始片段；在你的电脑上不需要这样做。

---

## 6. 运行环境与耗时

云端容器：4 vCPU Intel Xeon @ 2.10 GHz，15 GB 内存，无 GPU；Python 3.13.16，torch 2.14.1+cpu，numpy 2.5.4，scipy 1.18.1，
onnxruntime 1.31.0，kokoro-onnx 0.6.1，gcc 13.3。

| 步骤 | 耗时 | 备注 |
|---|---|---|
| 01_download | 约 1.5 h | MSWC 35 GB 约 10 min（70 MB/s），LibriSpeech 960 h 约 50 min，MUSAN 约 18 min |
| 03_tts | 约 1 h | 16 000 段，3 进程 |
| 02_prepare | 约 6 min | |
| 10_train_kw r1 | 132 min | 50 轮，约 140 s/轮 |
| 12_mine（r2） | 133 min | 40 个分片（242 h 训练说话人语音）+ 噪声 |
| 10_train_kw r2 | 56 min | 25 轮 |
| 12_mine --stage r3 | 约 2 h（与声纹训练并行时 224 min） | |
| 10_train_kw r3 | 50 min | 20 轮 |
| 20_eval --sweep（dev） | 50 min | 160 组设置；CNN 输出有缓存，重复扫描很快 |
| 20_eval（dev / test，3 档） | 22 / 25 min | |
| 30_export | 2 min | |
| 40_train_spk | 130 min | 40 轮（与挖掘并行） |
| 41_eval_spk（val / test） | 19 / 6 min | |
| 42_export_spk | <1 min | |

---

## 7. 阶段 B2：声纹模型热启动微调 + 蒸馏（2026-11，结论：未交付）

目的：在 spk-20261023（`models/spk_best_v1.pt`，与 `models/spk_best.pt` 相同）的基础上继续训练，让本人跨天、跨距离的分数更稳定，
同时不降低防冒充能力。网络结构、输入（40 维 log-mel × 81 帧）、128 维输出、C 引擎、导出格式都不变。
结果：三次尝试都没有"明显更好"（测试集平均 EER 9.25% → 最好 8.90%，标准是至少好 15%），所以没有导出新的 `spk_model.c`，详见 `REPORT.md`。

### 7.1 参数（`configs/spk.yaml` 的 `finetune:` 段）

| 参数 | 值 | 说明 / 理由 |
|---|---|---|
| `init` / `out` | spk_best_v1.pt / spk_ft.pt | 热启动，不从头训练 |
| `extra.vctk` | 每人 40 句、≤4 s | 110 个英式口音说话人；从 10.7 GB 的 zip 里用 HTTP Range 只取需要的文件 |
| `extra.peoples_speech` | 7 个分片 × 1500 段 × 3 s | 公开演讲、听证、访谈，说话人多样；没有说话人标签，只做蒸馏 |
| `teacher.model` | speechbrain/spkrec-ecapa-voxceleb | ECAPA-TDNN（VoxCeleb 1+2 训练，Apache-2.0），192 维 |
| `teacher.per_speaker` / `max_s` | 3 段 / 3 s | 老师看**干净**的、最长 3 s 的音频；学生看同一段经过设备链的 1.62 s 窗口 |
| `device.*` | 见配置 | INMP441 高通 60–150 Hz + 低/高频搁架 ±2–4 dB；距离 0.3–1 m（临界距离 0.4–0.8 m → DRR +8…−8 dB）；RT60 0.2–0.8 s；电平 −45…−3 dBFS（满幅硬削波）；喊叫 30%；噪声 60%（0–25 dB）；底噪 −80…−62 dBFS |
| `train.freeze_layers` | 1 | 冻结第一层卷积和它的 BN（你的要求） |
| `train.freeze_bn_stats` | true（尝试 C） | 所有 BN 层沿用旧模型的运行统计量，避免强增强数据一开始就把归一化推偏（A/B 开头会先掉点） |
| `train.lr` / `lr_heads` | 1e-4（A/B：3e-4）/ 3e-3 | 网络用小学习率；新加的投影层和 AAM 权重用大一些 |
| `train.epochs` × `steps_per_epoch` × 批 | 16 × 150 × (128 AAM + 128 蒸馏) | |
| `train.loss_aam` / `aam_margin` / `aam_scale` | 0.5 / 0.2 / 30 | 有标签数据（5302 人，含 VCTK）保留 AAM；AAM 权重从旧模型的说话人中心初始化 |
| `train.loss_distill` | 0.5（A：1.0） | 1 − cos(投影(学生), 老师)，投影层 128→192 只在训练时用，导出时去掉 |
| `train.loss_relational` | 10（A：0） | batch 内两两余弦矩阵与老师的均方差（与维度无关） |
| `train.loss_consistency` | 1.0（A：0） | 同一段音频的"设备视图"和"录入视图"向量一致 |
| `train.max_clips_per_speaker` | 10 | AAM 流每人取 10 段（工作集能放进内存） |
| 性别平衡 | 两个数据流都 50/50 | 数据里男声约为女声的 2.5 倍 |
| `eval.*` | 10 段录入、6 段测试、同/跨性别各 15 个冒充者、0.52 | 设备录入协议；条件 clean / far / shout / noisy 见配置 |
| `deliver_if` | 平均 EER 至少好 15%，且各条件不变差 > 0.5 点 | 交付标准 |

### 7.2 复现命令

```bash
python scripts/43_spk_extra.py                                # VCTK + People's Speech + LibriSpeech 性别表 -> data/cache/spk_extra.*
python3 -m venv venv_teacher && venv_teacher/bin/pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
venv_teacher/bin/pip install -r requirements_teacher.txt
venv_teacher/bin/python scripts/44_teacher.py                 # 老师向量 + 性别 -> data/cache/distill_*, spk_gender.csv
python scripts/45_finetune_spk.py                             # -> models/spk_ft.pt（尝试 C 的设置；A/B 的值写在配置注释里）
python scripts/46_compare_spk.py --split test                 # 新旧对比 -> models/spk_compare_test.json, docs/spk_compare_test.png
# 只有 46 给出 "deliver": true 时才导出：把 spk_ft.pt 复制为 models/spk_best.pt，再运行 scripts/42_export_spk.py
```

### 7.3 耗时（4 vCPU）

| 步骤 | 耗时 |
|---|---|
| 43_spk_extra | 约 12 min |
| 44_teacher（33 145 段） | 78 min（另加约 13 min 冷缓存随机读） |
| 45_finetune_spk | A 24 min，B 31 min，C 31 min |
| 46_compare_spk | 3 min |

图：`docs/spk_ft_a_training_curves.png`、`docs/spk_ft_b_training_curves.png`、`docs/spk_ft_training_curves.png`（C）、`docs/spk_compare_test.png`（新旧分数分布）。

---

## 8. 阶段 C：关键词模型 r4（减少咳嗽和说话误触发，2026-11）

**为什么**：固件 v1.6.7 + r3 在设备上，单词 "help" 常被咳嗽（用力咳一声 help 概率可到 0.65）和旁边的人说话误触发。原因：
① `_other` 里从来没有单独的咳嗽、清嗓、喷嚏、呼吸、笑声（ESC-50 只作为语音底下的背景噪声），模型没见过，softmax 把它分给最短、气声最重的 help；
② 难负样本只来自 LibriSpeech 有声书朗读，没有日常对话，也没有中文；
③ 从没用过设备麦克风的真实录音。

**做法**：从 r3 热启动微调（网络、窗口、特征都不变，固件只需换 `kws_model.c`），加入：主人用 DataRec 在设备上录的 help / help me / SOS（正样本）；VocalSound + ESC-50 的人声非语音（负样本）；AMI 英文会议对话、AISHELL-1 中文（负样本）；r3 在这些新数据上误报的窗口（难负样本 r4）。**所有评估都用固件 v1.6.7 的检测逻辑本身**（C 代码 `c_ref_v167/` 编译后由 Python 调用，`kws/fwdet.py`），不是 Python 近似。

### 8.1 新增代码
| 文件 | 作用 |
|---|---|
| `scripts/04_import_own.py` | 导入 DataRec 录音（文件夹或 zip）：检查格式（16 kHz/16 bit/单声道）、文件夹→标签、固定划分（每人 ≥30 条时按文件名哈希 20 % 测试 / 10 % 验证；不足 30 条的人整体作测试 = 没见过的人）→ `data/own/manifest.csv`。`data/own` 不进仓库（公开仓库，真人声音） |
| `tools/check_own.py` | 用固件逻辑让 r3（或 r4）听每一条录音：每个词、每种风格的命中率；列出疑似录错的条目 |
| `scripts/05_download_r4.py` | 下载 VocalSound（1.7 GB）、AMI 会议（约 4 GB）+ 逐词转写、AISHELL-1（15 GB 流式，只留选中的人） |
| `scripts/06_prepare_r4.py` | 生成缓存 `own`、`vocal`（每个声音事件一段，连咳保持在一起）、`chat`（AMI 2 s 片段、AISHELL 整句）、误报测试流 `fa4_val/test`（AMI 整场会议、AISHELL 每人连续说、人声非语音按人连在一起） |
| `scripts/13_mine_r4.py` | r3 在新训练音频上触发的窗口 → `hardneg_r4`；统计每类声音被 r3 误触发的比例（"优化前"的证据） |
| `kws/dataset.py` | 读新缓存；关键词窗口按 `own_share` 取主人录音，用较轻的 `augment_own` |
| `kws/fwdet.py`、`c_ref_v167/harness.c` | 固件 v1.6.7 检测器（C）：20 ms 一块 → `kws_dbfs` → C 前端 → `kws_det_frame`（声音分段、连说两次、10 s 内 2 次、help +0.05 不放宽）；概率来自 PyTorch 模型，或 `--native` 用导出的 int8 C 引擎 |
| `scripts/21_eval_r4.py` | r3 / r4 系统级评估、验证集阈值扫描、对比图 |
| `scripts/run_r4.sh` | 按步骤运行（0–10），每步日志 `logs/r4_step<N>.log` |
| `tools/smoke_r4.py` | 用合成音频把整条流程跑一遍（几分钟），检查环境和代码 |

检验：`python -m kws.fwdet --check`：C 前端与 `kws.features.logmel` 最大差 9e-6；脚本化场景与固件单元测试一致（help 0.70+0.45：v1.6.7 之前 SOS、之后不 SOS；SOS 0.70+0.45 仍 SOS）。合成数据上 PyTorch 模型与 int8 C 引擎的评估结果一致（误报 14 vs 13，int8 舍入）。

### 8.2 参数（`configs/train.yaml`）
| 参数 | 值 | 理由 |
|---|---|---|
| `own.min_clips_per_speaker` / `test_pct` / `val_pct` | 30 / 20 / 10 | 主人（A）的录音按文件分出测试、验证；朋友录得少就整体作测试，检验没见过的人 |
| `r4_sources.vocalsound` | 官方说话人划分 tr/val/te | 3 365 人的咳嗽、清嗓、喷嚏、吸鼻、叹气、笑；按人划分，测试的人训练时没见过 |
| `r4_sources.esc50_vocal` | coughing, sneezing, breathing, laughing, snoring；fold 5 测试、4 验证 | 原来只当背景噪声，现在也作为独立的 `_other` 片段；fold 5 与以前一样只用于测试 |
| `r4_sources.ami` | train 36 场 / val 12 场 / test 16 场（full-corpus-asr 划分） | 自然对话（抢话、笑、语气词），和有声书很不一样；逐词转写用来排除真的说了 "help" 的地方 |
| `r4_sources.aishell` | 每个划分取 120 / 20 / 20 人，每人 60 / 100 / 120 句 | 中文里大量 h 开头的短音节（喝、嘿、好、还、哈）；按 AISHELL 的说话人划分 |
| `r4_prepare.event_*` | 比最响帧低 30 dB 以内且高于背景 10 dB；间隔 < 0.6 s 合并；前后各留 0.25 s；最长 4 s | 一个片段 = 一次声音事件，连咳保持在一起（设备上最容易凑成两次） |
| `r4_prepare.ami_chunk_s` / `ami_chunk_every_s` | 2.0 / 6.0 | 每 6 s 取一段 2 s，覆盖整场会议又不过多；离 "help" 1 s 以内的不用 |
| `r4_prepare.fa_hours` | val：AMI 6 h + AISHELL 1.5 h；test：AMI 9 h + AISHELL 2.5 h | 误报测试流的上限 |
| `mine_r4.threshold` | 0.35 | r3 中档 help 阈值 0.45（设备上 +0.05 = 0.50）；比它低一点，抓住"差一点就误报"的窗口 |
| `mine_r4.max_per_source` | 对话英 6000、中 4000、人声 6000 | 每类最有把握的那些，避免某一类占满 |
| `augment_own` | 增益 −10…+6 dB，噪声概率 0.6、SNR 5–25 dB，混响 0.15，变速 0.3，变调 0.15，喊叫 0.05，高通 0.3 | 主人录音已经是设备麦克风和真实房间，增强轻一些，避免失真 |
| `train_kw_r4.init` / `epochs` / `lr` | r3 / 20 / 0.0005 | 微调：学习率比 r3（0.0007）再低，保留 r3 已学会的东西 |
| `train_kw_r4.own_share` | help 0.25、help me 0.25、sos 0.35 | 关键词窗口里主人录音的比例；SOS 以前只有合成语音，所以比例更高 |
| `train_kw_r4.other_mix` | 人声 0.14、r4 难负样本 0.12、英文对话 0.07、中文 0.08，其余为 r1–r3 的来源（按比例缩小） | 新问题的数据占 41 %，旧数据仍占多数，防止遗忘（LibriSpeech 误报不能变差） |
| `eval_r4.device` | 中档、10 s 内 2 次（连说模式）、help +0.05 | 固件 v1.6.7 的默认设置 |
| `eval_r4.sweep_med_thr` × `sweep_help_extra` | 0.45–0.70 × {0, 0.05, 0.10} | 只在验证集上选：对话误触发 SOS ≤ 1 次/24 h 的前提下单次命中率最高；low = 中档 +0.10，high = 中档 −0.05（与 r3 相同的间距） |

### 8.3 运行（在 WSL Ubuntu 的笔记本上）
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # 版本装不上时用最接近的，并记录
pip install -r requirements.txt
sudo apt install -y build-essential          # gcc（固件检测器要编译）
bash scripts/run_r4.sh 0                     # 环境 + 冒烟测试
bash scripts/run_r4.sh 1 /mnt/c/Users/Caizy/Desktop/MeshTalk_KWS_Data/DATASET
bash scripts/run_r4.sh 2                     # … 一直到 10（见 PLAN_R4.md 的表）
```

### 8.4 结果（由运行后填写，数字来自 `models/eval_r4_*.json`）
| 指标（测试集，中档，固件 v1.6.7 逻辑） | r3 | r4 | r4 int8（C 引擎） |
|---|---|---|---|
| 主人 help / help me / SOS 单次命中率 | 100 / 100 / 55 % | **100 / 100 / 65 %** | **100 / 100 / 65 %** |
| 公开 real_help / real_help_me / tts_sos 单次命中率 | 90.0 / 94.1 / 94.3 % | **82.0 / 79.4 / 89.0 %** | **85.5 / 79.4 / 89.3 %** |
| 咳嗽每 100 声误触发（VocalSound / ESC-50） | 38.5 / 33.3 | **0.33 / 0** | **0.65 / 0** |
| 其他人声每 100 声误触发（清嗓、喷嚏、吸鼻、叹气、笑） | 9.6 | **0.33** | **0.57** |
| 对话每小时误触发（AMI / AISHELL / LibriSpeech） | 5.85 / 9.30 / 6.17 | **1.88 / 1.49 / 4.25** | **2.65 / 1.49 / 4.08** |
| 对话每 24 h 误触发 SOS | 22.2（21.2 / 26.8 / 22.0） | **9.1（2.6 / 0 / 16.0）** | **12.1（10.6 / 0 / 16.0）** |
| 选定阈值（低/中/高）、HELP_EXTRA | 0.65/0.55/0.50、0.05 | **0.80/0.70/0.65、0.05** | 同 r4 |

结论：咳嗽和其他人声的误触发降低约 30–100 倍（目标 ≥5 倍，**达到**）；主人录音命中率不低于 r3（SOS 55→65 %，**达到**）；但对话误报 SOS 仍为 9.1 次/24 h（目标 ≤1，**未达到**），公开数据命中率低 4.7–14.7 个点（目标 ≤2，**未达到**）。验证集扫描中没有一组阈值能同时满足"对话 ≤1 次/24 h"和"公开命中率不下降"；上面的阈值是在验证集上按"误报最低优先"选的（8.9 次/24 h），**没有用测试集调参**。脚本自动给出的 `recommended`（0.55/0.45/0.40、help_extra 0）是"没有任何一组满足约束"时退回"命中率最高"的结果，其对话误报为 459.9 次/24 h，未采用。
