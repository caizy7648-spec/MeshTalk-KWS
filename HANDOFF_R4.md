# 第三阶段（r4）提示词：在电脑上的 Ubuntu（WSL）里用 Claude Code 终端版训练

## 开始前
1. 用 MeshTalk DataRec v1.0 录好 help / help me / SOS，拔卡，把卡里的 **`DATASET` 整个文件夹** 拷到
   `C:\Users\Caizy\Desktop\MeshTalk_KWS_Data\DATASET`
   在 Ubuntu 里它的路径是 `/mnt/c/Users/Caizy/Desktop/MeshTalk_KWS_Data/DATASET`（直接读，不用拷进仓库）。
2. 打开 Ubuntu 终端，把仓库克隆到 **Linux 自己的目录**（不要放在 /mnt/c 下，那里读写很慢）：
   ```bash
   cd ~
   git clone https://github.com/caizy7648-spec/MeshTalk-KWS.git    # 已经克隆过就 cd ~/MeshTalk-KWS && git pull
   cd MeshTalk-KWS
   claude
   ```
3. 把下面整段粘贴进 Claude Code（路径不同就改第一行）。

---

我的录音在：`/mnt/c/Users/Caizy/Desktop/MeshTalk_KWS_Data/DATASET`（Windows 桌面上的文件夹）

运行环境：我的笔记本（Windows + WSL Ubuntu），Intel Core Ultra 9 275HX（24 核）、96 GB 内存、RTX 5070 8 GB。按 `requirements.txt` 建 Python 虚拟环境；训练和以前一样用 CPU 版 torch 即可（模型很小，结果可复现）。如果你判断某一步用 GPU 明显更快，RTX 5070 需要 CUDA 12.8 以上的 torch 版本，先问我再装。公开数据下载到仓库的 `data/`（WSL 自己的磁盘），开始前先告诉我大概要多少空间。

这个仓库是我毕业设计 MeshTalk（ESP32-S3 LoRa 对讲机）"语音 SOS"的训练包。r1–r3 和声纹阶段都已完成（见 `REPORT.md`、`TRAINING.md`）。现在做第三阶段 **r4**：设备上单词 "help" 经常被**咳嗽**和旁边的人**说话**误触发，要减少误报，同时不降低真实呼救的识别率。

请先完整读 `PLAN_R4.md`（这一阶段的原因、数据、训练、评估、验收标准），以及 `CONTRACT.md`（固件对模型的硬性要求，不能改），然后按 PLAN_R4.md 做：

1. **导入我的录音**：运行 `python scripts/04_import_own.py --src <上面的路径>`，把统计表（每类、每种风格、训练/验证/测试条数）发给我。再用 r3 检查一遍，把可能录错的条目列给我确认。**我的录音（data/own）和任何我的音频都不要提交到仓库**（仓库是公开的）。
2. **公开负样本**：咳嗽、清嗓、喷嚏、呼吸、笑声、英文对话、中文语音，按 PLAN_R4.md 的候选数据集下载，核实许可证写进 REPORT.md；下载不了的把确切网址和大小写进 REPORT.md 后停下来告诉我。需要我在网页上登录/同意条款的数据集（比如 Common Voice）也先问我。
3. **先测基线**：按 PLAN_R4.md 用 `c_ref_v167/` 编译固件 v1.6.7 的检测器，r3 在所有新测试集上的数字先出来（这就是"优化前"）。
4. **训练 r4**：从 r3 热启动微调，加一轮难负样本挖掘；所有新参数写进 `configs/train.yaml`。
5. **对比和导出**：r3 vs r4 对比表和图（`docs/kw_r4_*.png`），选阈值，说明固件里 help 的 +0.05（HELP_EXTRA）还要不要；导出 `out/kws_model.c`，`python -m kws.check_c out` 0 failed。

要求（和前两个阶段一样）：
- 训练代码和参数是交付物（答辩要用）：所有代码放进仓库并 push，参数全在 `configs/train.yaml`，`TRAINING.md` 加中文的第三阶段说明（每个新参数的值和理由、完整复现命令、环境、耗时）。
- 每完成一步更新 `REPORT.md`（数字写在前面，简洁）并 commit + push。
- 达不到目标或数据拿不到就停下来汇报，不要硬凑，不要在测试集上调参。
- 用中文和我交流。
