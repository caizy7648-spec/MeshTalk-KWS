# 第三阶段（r4）：在 WSL Ubuntu 里用 Claude Code 终端版**监督训练**

训练代码已经全部写好并测试过（`tools/smoke_r4.py` 用合成音频跑通全部 14 步）。
终端里的 Claude Code **不需要写训练代码**，它的工作是：按步骤运行、盯进度、检查每一步的数字是否合理、遇到问题停下来问你、把结果写进报告并推送。

## 一、开始前（你自己做，约 10 分钟）
1. **录音**：用 MeshTalk DataRec v1.0 录 help / help me / SOS，拔卡，把卡里整个 `DATASET` 文件夹拷到
   `C:\Users\Caizy\Desktop\MeshTalk_KWS_Data\DATASET`
   （在 Ubuntu 里路径是 `/mnt/c/Users/Caizy/Desktop/MeshTalk_KWS_Data/DATASET`）
2. **用电脑上已有的仓库** `~/mt/meshtalk-kws`（11 月 2–3 日本地跑声纹时留下的，里面已有 29 GB 数据）。在 Ubuntu 终端里一行一行运行：
   ```bash
   cd ~/mt/meshtalk-kws
   git add models/teacher_gender.json && git commit -m "teacher_gender.json from the local GPU run"
   git push origin gpu-train                      # 本地 GPU 声纹训练的改动存到 GitHub 的 gpu-train 分支
   git remote set-url origin https://github.com/caizy7648-spec/MeshTalk-KWS.git
   git checkout main
   git pull
   claude
   ```
3. **磁盘**：还需要约 30 GB（新数据约 10 GB + 难负样本挖掘时临时下载的 LibriSpeech 分片）。
4. 把下面分割线以下的整段粘贴进 Claude Code。
5. 之后它每完成一步会停下来汇报，你看一眼回"继续"就行；有需要你决定的事（比如疑似录错的录音）它会问你。
   第 3 步要临时下载 LibriSpeech 分片挖难负样本，可能要几个小时，可以挂着过夜。

---

我的录音在：`/mnt/c/Users/Caizy/Desktop/MeshTalk_KWS_Data/DATASET`

这个仓库是我毕业设计 MeshTalk（ESP32-S3 LoRa 对讲机）"语音 SOS"的关键词模型训练包。现在做第三阶段 **r4**：设备上单词 "help" 常被咳嗽和旁边的人说话误触发，r4 要减少误报，同时不降低真实呼救的识别率。

**训练代码已经全部写好、测试过，你不要写新的训练代码，也不要改网络或训练逻辑。** 你的任务是在我这台电脑上按步骤运行、监督、检查结果、汇报和打包。

请先完整读这三个文件：`PLAN_R4.md`（步骤表、每步要检查什么、验收目标、规则）、`TRAINING.md` 第 8 节（每个参数和理由）、`CONTRACT.md`（固件对模型的硬性要求）。

运行环境：我的笔记本，Windows + WSL Ubuntu，Intel Core Ultra 9 275HX（24 核）、96 GB 内存、RTX 5070。仓库在 `~/mt/meshtalk-kws`（main 分支）。r4 全部用 CPU 跑（模型很小，结果可复现），不要改成 GPU。

**这个文件夹以前用过**（11 月 2–3 日本地训练声纹，分支 `gpu-train`，那个分支和 `spk_big*` 文件、`venv_teacher/`、`out_big120/` 都不要动）：
- 已有 Python 环境 `.venv/`：每次运行前 `source .venv/bin/activate`；第 0 步会检查缺不缺包，缺的按 `requirements.txt` 补装（先告诉我命令）。
- **第 2 步已经做过**：`data/raw`（12 GB：MSWC、LibriSpeech、MUSAN、ESC-50、Kokoro、Speech Commands）和 `data/cache`（17 GB：clips、noise、fa_dev/fa_test、tts）都在。**不要运行 `run_r4.sh 2`**，只检查 `data/cache/counts.json` 和 `data/raw/MANIFEST.json` 是否完整，结果写进 REPORT.md。
- **第 3 步要做**：`hardneg.npy`、`hardneg_r3.npy` 还没有。
需要 `sudo apt install build-essential`（gcc）时先问我。

**然后按顺序一步一步运行** `bash scripts/run_r4.sh <步骤号>`（第 1 步：`bash scripts/run_r4.sh 1 <上面的录音路径>`），步骤 0、1、3、4 … 10（跳过 2），每步的日志在 `logs/r4_step<N>.log`。

每一步做完后：
1. 对照 `PLAN_R4.md` 步骤表的"check before going on"那一列，检查输出文件和数字是否合理；
2. 用中文给我一个简短汇报：这一步做了什么、关键数字（表格）、有没有异常；
3. 在 `REPORT.md` 最后的 Phase C 表里填上这一步的结果（数字在前，简洁），然后 `git add` 代码、配置、`models/*.json`、`models/kw_r4.pt`、`docs/*.png`、`out/`，commit 并 push；
4. **等我回复"继续"再做下一步**。第 3 步很长，可以在后台运行（比如 `nohup ... &` 或 tmux），隔一段时间看日志汇报进度。

特别注意：
- **第 1 步**：把 `tools/check_own.py` 列出的"may be wrong takes"给我看（文件名、听成了什么、分数）。我确认哪些是录错的，你再把那些文件从 `data/own` 和 `data/own/manifest.csv` 里去掉（不要删我桌面上的原始文件），然后重新跑 `python scripts/04_import_own.py --src <路径>` 之前先问我。"missed"（r3 没听到）是正常的，正是 r4 要学的，不用删。
- **第 8 步**：把 `models/eval_r4_sweep_val.json` 的 "recommended"（thr 和 help_extra）给我看，经我同意后填进 `configs/train.yaml` 的 `export_r4:`（`thr` 和 `help_extra` 两行），再做第 9 步。
- **第 9、10 步**：给我 r3 和 r4 的对比表（主人录音命中率、公开数据命中率、咳嗽和其他人声每 100 声误触发、对话每小时误触发和每 24 h 误触发 SOS），对照 `PLAN_R4.md` 的 Targets 逐条说达到没有。没达到就如实说，**不要在测试集上调参数**。
- 最后：补全 `REPORT.md` 的 Phase C 部分和 `TRAINING.md` 8.4 节的结果表（数字来自 `models/eval_r4_*.json`），说明推荐的阈值和 HELP_EXTRA，推送。告诉我 `out/kws_model.c` 已经生成（我会把它交给固件那边）。

规则：
- **`data/` 目录永远不要提交**（仓库是公开的，`data/own` 是我本人的声音）；`build/`、`logs/` 也不提交。
- 某一步报错：把错误信息和你认为最小的修改方案给我看，**我同意后**再改，并在 REPORT.md 里记下改了什么。不要自己绕过问题或换方法。
- 删除文件、安装系统软件、用 sudo 之前都先问我。
- 用中文和我交流。
