# 声纹 Phase D：在 WSL Ubuntu 里用 Claude Code 终端版**监督训练**（防模仿）

训练代码已写好并测试过（`tools/smoke_spk_d.py` 合成声音全部 6 步 PASS）。终端里的 Claude Code 只负责运行、检查、汇报、推送。

## 一、开始前（你自己做）
```bash
cd ~/mt/meshtalk-kws
git status                 # 应该是干净的（models/spk_big120.pt 这类未跟踪文件没关系，不要删）
git checkout main
git pull
source .venv/bin/activate
claude                     # 开新对话
```
然后把下面分割线以下的整段粘贴进去。全程预计 4–6 小时（主要是三次训练，可以挂着过夜）。不用录音、不用下载。

---

这个仓库是我毕业设计 MeshTalk（ESP32-S3 LoRa 对讲机）的语音模型训练包。现在做**声纹模型 Phase D**：设备上的 "Only My Voice" 声纹（spk-20261023）被一位女性压低声音模仿我，试十几次能骗过一次。Phase D 用同样的网络重新训练三个对比版本（d0 / d1 / d2），核心想法是让模型"忽略音高、只认声道（共振峰）"——人能随意改音高，改不了声道长度。

**训练代码已经全部写好、测试过，你不要写新的训练代码，也不要改网络或训练逻辑。**

先完整读：`PLAN_SPK_D.md`（原因、三个版本、每个数字的含义、步骤表、验收目标），再看 `REPORT.md` 的 Phase B、B2、B3（旧声纹模型的历史结果）。

运行环境：笔记本 WSL Ubuntu，Intel Core Ultra 9 275HX（24 核）、96 GB 内存、RTX 5070；仓库 `~/mt/meshtalk-kws`（main），每次运行前 `source .venv/bin/activate`。训练会自动用 GPU（没有 GPU 就用 CPU，也能跑，只是慢一些——如果第 0 步显示 `cuda False`，告诉我再继续）。不要运行 `run_r4*.sh` 的任何步骤，不要重新下载数据，不要动 `gpu-train` 分支和 `spk_big*`、`venv_teacher/`、`out_big120/`。之前本地蓝屏过一次（内存），训练用 tmux 或 nohup 后台跑，留意内存；如果内存紧张，可以把 `configs/spk.yaml` 的 `phase_d.train.workers` 从 16 改小（先问我）。

**按顺序运行** `bash scripts/run_spk_d.sh <步骤号>`，日志在 `logs/spk_d_step<N>.log`。

- **第 0 步做完停下来汇报**：GPU、缺不缺文件、我本人（own-A）和模仿者（own-B）各有多少条录音（按 style 分）、smoke test 结果。**我回复"继续"后**，第 1–6 步**自动连续跑完，不用每步等我**，每步照常检查、写进 REPORT.md、commit、push。
- **只有这几种情况停下来问我**：某一步报错；第 0 步有文件缺失；训练 loss 不降或 val_select 一直不降；内存不够。
- **第 7 步（导出）必须等我决定**用哪个版本、哪个阈值之后才做。

每一步做完后：
1. 对照 `PLAN_SPK_D.md` 步骤表 "check before going on" 一列检查；
2. 在 `REPORT.md` 最后新加一节 **"Phase D (spk, anti-imitation)"**，填这一步的结果（数字在前，简洁）；
3. `git add` 代码、配置、`models/*.json`、`models/spk_d*.pt`、`models/spk_d*_log.csv`、`docs/*.png`，commit 并 push。

最后（第 6 步之后）一次性给我汇报：
- **表 1（val 和 test，firmware 打分 `max`）**：old / big120（如有）/ d0 / d1 / d2 每列：clean、far、shout、noisy 的 EER，四个平均，`mimic_f2m` 和 `plain_f2m` 的 EER，`mimic_f2m` 的 `far_at_frr`（5 % 拒绝本人时模仿者的通过率）。
- **表 2（打分方式）**：每个模型 `max` / `asn` / `pair` / `pasn` 的 select_metric（四个条件 + 模仿的平均 EER）。
- **表 3（我和模仿者的真实录音，`own`）**：每个模型 `max` 和 `pair` 的 EER、我本人分数范围（owner_pct）、模仿者分数范围（imitator_pct）、我本人按 style 的通过率、模仿者 10 次和 20 次内触发 SOS 的概率。
- `best_variant_val`，以及对照 `PLAN_SPK_D.md` Targets 逐条说达到没有。没达到如实说，**不要在测试集上调参数**。
- 选中版本的 `firmware_thresholds`（旧阈值 0.46 / 0.52 / 0.60 对应的新阈值）。

规则：
- **`data/` 永远不要提交**（公开仓库，`data/own` 是我和模仿者的真实声音）；`build/`、`logs/` 也不提交。
- 报错时：给我看错误和最小修改方案，**我同意后**再改，并记进 REPORT.md。不要自己绕过问题或换方法。
- 删除文件、安装系统软件、sudo 之前先问我。
- 用中文和我交流。
