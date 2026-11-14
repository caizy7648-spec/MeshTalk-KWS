# r4b：在 WSL Ubuntu 里用 Claude Code 终端版**监督训练**（"hel" 不该算 help）

训练代码已经写好并测试过（`tools/smoke_r4.py` 用合成音频跑通，含 r4b 的步骤）。
终端里的 Claude Code **不写训练代码**，只负责：按步骤运行、盯进度、检查数字、汇报、推送。

## 一、开始前（你自己做，2 分钟）
在 Ubuntu 终端里一行一行运行：
```bash
cd ~/mt/meshtalk-kws
git status                 # 应该是干净的（data/ build/ logs/ 不会显示）
git checkout main
git pull
source .venv/bin/activate
claude
```
不需要新录音、不需要下载新数据（r4 的数据都还在 `data/`）。磁盘只需要几 GB。
然后把下面分割线以下的整段粘贴进 Claude Code。它每做完一步会停下来汇报，你回"继续"就行。
全程大约 2–3 小时（第 1、3 步各约半小时到一小时，CPU 跑）。

---

这个仓库是我毕业设计 MeshTalk（ESP32-S3 LoRa 对讲机）"语音 SOS"的关键词模型训练包。r4 已经训练完成并用在固件 v1.6.8 上：咳嗽不再误触发了，但出现新问题——**我说 "hel"（不发尾音 /p/）也会被识别成 help**，r3 不会这样。现在做 **r4b**：在 r4 基础上加入"切掉词尾的 help"作为负样本，同时减少我本人录音的比例、恢复相似词（hell、held、hello…）的比例。

**训练代码已经全部写好、测试过，你不要写新的训练代码，也不要改网络或训练逻辑。** 你的任务是按步骤运行、监督、检查结果、汇报和推送。

请先完整读：`PLAN_R4B.md`（原因、改动、步骤表、每步检查什么、验收目标），再看 `PLAN_R4.md` 的 Rules 部分和 `REPORT.md` 的 Phase C（r4 的结果，r4b 要和它比）。

运行环境：我的笔记本，Windows + WSL Ubuntu，24 核 CPU、96 GB 内存。仓库 `~/mt/meshtalk-kws`（main 分支），Python 环境 `.venv/`（每次运行前 `source .venv/bin/activate`）。全部用 CPU 跑，不要改成 GPU。`data/` 里 r4 用过的数据都在，**不要重新下载、不要运行 `run_r4.sh` 的任何步骤**。之前 11 月初本地蓝屏过一次（内存），长步骤请在 tmux 或 `nohup` 后台跑，并留意内存。

**按顺序一步一步运行** `bash scripts/run_r4b.sh <步骤号>`，步骤 0 到 5，日志在 `logs/r4b_step<N>.log`。

每一步做完后：
1. 对照 `PLAN_R4B.md` 步骤表的 "check before going on" 一列检查输出文件和数字；
2. 用中文给我简短汇报：做了什么、关键数字（表格）、有没有异常；
3. 在 `REPORT.md` 最后新加一节 **"Phase C2 (r4b)"**，填这一步的结果（数字在前，简洁），然后 `git add` 代码、配置、`models/*.json`、`models/kw_r4b.pt`、`models/kw_r4b_log.csv`、`docs/*.png`、`out/`，commit 并 push；
4. **等我回复"继续"再做下一步**。

特别注意：
- **第 1 步**：给我看 r4 的 `cut_own_help` 和 `cut_real_help`（heard、right、`@pair` 的 sos_rate）以及各组的 `score_median`（我本人 vs 公开数据的分数差），这就是问题本身的数字。
- **第 3 步**：给我看 sweep 里 medium = 0.65 / 0.70 / 0.75、help_extra = 0.05 三行：每组呼救命中率、cut 组被听成 help 的比例、对话每 24 h 误触发 SOS。`export_r4b` 先保持和 r4 一样（0.80/0.70/0.65、help_extra 0.05），**我决定之后**才改。
- **第 4、5 步**：给我 r4 和 r4b 的对比表（int8 和 PyTorch 都要）：我本人 help / help me / SOS 命中率、公开数据命中率、cut_own_help 和 cut_real_help 被听成 help 的比例和 "hel…hel" 误触发 SOS 的比例、咳嗽和其他人声每 100 声误触发、对话每 24 h 误触发 SOS。对照 `PLAN_R4B.md` 的 Targets 逐条说达到没有，没达到如实说，**不要在测试集上调参数**。
- 最后：补全 REPORT.md 的 Phase C2 一节（为什么做 r4b、改了什么、对比表、结论），推送，告诉我 `out/kws_model.c` 已经生成且名字是 `kws-r4b`（我会交给固件那边）。

规则：
- **`data/` 永远不要提交**（仓库是公开的，`data/own` 是我本人的声音）；`build/`、`logs/` 也不提交。
- 某一步报错：把错误信息和你认为最小的修改方案给我看，**我同意后**再改，并在 REPORT.md 里记下。不要自己绕过问题或换方法。
- 删除文件、安装系统软件、用 sudo 之前都先问我。
- 用中文和我交流。
