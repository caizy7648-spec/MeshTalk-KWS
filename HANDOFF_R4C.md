# r4c：在 WSL Ubuntu 里用 Claude Code 终端版**监督训练**（r4b 的修正版）

代码已写好并测试过（`tools/smoke_r4.py` 合成音频全部 PASS，含 r4c）。终端里的 Claude Code 只负责运行、检查、汇报、推送。

## 一、开始前（你自己做）
```bash
cd ~/mt/meshtalk-kws
git status                 # 应该是干净的
git checkout main
git pull
source .venv/bin/activate
claude                     # 开新对话即可；也可以用 claude --continue 接着上次的对话
```
然后把下面分割线以下的整段粘贴进去。全程约 1.5–2 小时，4 步（0–4），不用录音、不用下载。

---

这个仓库是我毕业设计 MeshTalk（ESP32-S3 LoRa 对讲机）"语音 SOS"的关键词模型训练包。现在做 **r4c**，是 r4b 的修正版。

背景（详见 `REPORT.md` 的 Phase C2）：r4 会把 "hel"（help 不发尾音 /p/）听成 help。r4b 加了"切掉词尾的 help"作为负样本，其中一半来自我本人的录音——结果我本人的 help 和 "hel" 在每个阈值下命中率都完全一样（我录音里的 /p/ 基本听不见），模型分不开，只能把我的 help 整体压低（int8 test：我的 help 100 % → 38 %）。公开数据上则拉开了差距（同样误报水平下 help 与 "hel" 的差从 4 个点到 21 个点）。

**r4c 的改动（只改了配置，`configs/train.yaml` 的 `train_kw_r4c` / `export_r4c`）**：从 **r4**（不是 r4b）开始训 8 轮；切尾负样本**只用公开数据**（`cut_help: {own_p: 0.0}`）；比重 0.08 → 0.05；我本人录音的比例恢复到 r4 的 0.25 / 0.25 / 0.35。阈值先保持和 r4 一样（0.80/0.70/0.65、help_extra 0.05）。

**训练代码已经全部写好、测试过，你不要写新的训练代码，也不要改网络或训练逻辑。**

先读 `PLAN_R4B.md`（cut 组是什么、每个数字的含义）和 `scripts/run_r4c.sh` 开头的注释。对照基准（r4 含 cut 组的数字）已经在 `models/eval_r4_r4_val.json`、`eval_r4_r4_test.json`、`eval_r4_r4_int8_test.json` 里，**不用重跑**。

运行环境：笔记本 WSL Ubuntu，24 核 CPU、96 GB 内存，仓库 `~/mt/meshtalk-kws`（main），每次运行前 `source .venv/bin/activate`。全部 CPU，不要改 GPU。不要运行 `run_r4.sh`、`run_r4b.sh` 的任何步骤，不要重新下载数据。长步骤用 tmux 或 nohup 后台跑。

**按顺序一步一步运行** `bash scripts/run_r4c.sh <步骤号>`，步骤 0 到 4，日志在 `logs/r4c_step<N>.log`。每一步做完后：
1. 检查输出和数字是否合理；
2. 用中文简短汇报（表格）；
3. 在 `REPORT.md` 最后新加一节 **"Phase C3 (r4c)"** 填结果，`git add` 代码、配置、`models/*.json`、`models/kw_r4c.pt`、`models/kw_r4c_log.csv`、`docs/*.png`、`out/`，commit 并 push；
4. **等我回复"继续"再做下一步**。

每步要特别给我看的：
- **第 1 步（训练）**：`models/kw_r4c_log.csv` 每轮的 val_acc，和 r4 最后一轮（0.776）、r4b（0.741）比；以及 `kw_r4c.pt` 里的 `val_cm` 第一行（help 这一类分对多少、错成 _other 多少），和 r4 的 `[928, 91, 0, 481]` 比。**如果 help 分对的数量低于 850，先停下来告诉我。**
- **第 2 步（val sweep）**：medium = 0.65 / 0.70 / 0.75、help_extra 0.05 三行，列：own_help、cut_own_help、real_help、cut_real_help（都用 heard/right）、real_help@pair 和 cut_real_help@pair 的 sos_rate、real_help_me、tts_sos、对话每 24 h 误触发 SOS。同时列出 r4 在 0.70 的同一行（来自 `eval_r4_r4_val.json`）。
- **第 3、4 步**：r4 / r4b / r4c 三列对比表（int8 和 PyTorch 都要）：我本人 help / help me / SOS 命中率、公开 help / help me / SOS 命中率、cut_own_help 和 cut_real_help 被听成 help 的比例、"hel…hel" 误触发 SOS 比例、咳嗽和其他人声每 100 声误触发、对话每 24 h 误触发 SOS。

验收（test，int8，0.70）：我本人 help ≥ 90 %；我本人 help me、SOS 不低于 r4；公开 help 比 r4 低不超过 5 个点；cut_real_help 被听成 help 明显低于 r4（r4：81.75 %）；对话误报、咳嗽不比 r4 差。cut_own_help 预计改善有限（我的录音里 /p/ 听不见），如实报告即可。**不要在测试集上调参数。**

最后：补全 REPORT.md 的 Phase C3（为什么做、改了什么、三列对比表、结论），推送，告诉我 `out/kws_model.c` 的名字是 `kws-r4c`。

规则：
- **`data/` 永远不要提交**（公开仓库，`data/own` 是我本人的声音）；`build/`、`logs/` 也不提交。
- 报错时：给我看错误和最小修改方案，**我同意后**再改，并记进 REPORT.md。
- 删除文件、安装系统软件、sudo 之前先问我。
- 用中文和我交流。
