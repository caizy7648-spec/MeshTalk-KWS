# 粘贴给 Claude Code 云端会话的提示词

（在 claude.ai/code 或桌面 App 的 Code 里，选这个 GitHub 仓库开新会话，把下面整段粘贴进去。）

---

这个仓库是我毕业设计 MeshTalk（ESP32-S3 LoRa 对讲机）里"语音 SOS"的训练包。设备固件这边已经全部写好（特征提取、CNN 推理、连续识别、3 次触发、声纹录入界面），只差一个训练好的模型文件。请你在这个云端环境里完成训练。

请先完整读 `CONTRACT.md`（固件对模型的硬性要求，不能改）和 `PLAN.md`（步骤、数据来源、验收标准），然后按 PLAN.md 的 Step 0 → 4 做：

1. Step 0：装环境，`python -m kws.check_c` 必须 0 failed；记录哪些下载站点能访问。
2. Step 1：只用公开数据（MSWC 的 help / me、Kokoro 合成语音、Speech Commands、LibriSpeech、ESC-50 等）。下载不了的，把确切网址和大小写进 REPORT.md 后停下来告诉我，我在电脑上下载再传上来。
3. Step 2：训练关键词（help / help me / sos / _other），用 `kws.detect.Detector`（和设备逻辑完全一样）评估命中率和每小时误报。
4. Step 3：声纹头只用真人数据训练，按设备的录入方式（5 句录入）测 EER，给出"Only My Voice 能不能用"的明确结论。
5. Step 4：选阈值、导出、`python -m kws.check_c out` 0 failed，交付 `out/kws_model.c`、`out/kws_model_test.c`、`REPORT.md`，以及下面要求的训练代码、参数文件和 TRAINING.md。

**训练代码和参数必须完整交给我**（这是交付物的一部分，和模型同等重要，答辩要用）：
- 所有 Python 代码都放进仓库并 push：数据下载 / 整理、数据增强、训练、评估、阈值选择、导出，不能只在临时目录里跑。按 `PLAN.md` 里 "Training code and parameters" 一节的目录结构放。
- 所有参数写进 `configs/train.yaml`（一个文件，代码从这里读）：数据集和每类样本数、窗口长度、网络尺寸、增强的每个参数和概率、优化器、学习率和调度、batch、epoch、损失函数、类别权重、随机种子、阈值。
- 写一份中文的 `TRAINING.md`：每个参数是什么、为什么这样选；从零复现的完整命令；训练曲线（loss / 准确率）、混淆矩阵、命中率–误报率曲线、声纹 EER 曲线的图（PNG 放进 `docs/`）；运行环境（Python、torch 版本、CPU、训练耗时）。
- 训练好的权重 `models/best.pt`（几百 KB，直接提交）。
- 我在自己电脑上按 TRAINING.md 的命令应该能复现出同样的结果。

要求：
- 特征必须用 `kws/features.py`，网络层类型不能改（只能改尺寸，限制见 CONTRACT.md）。
- 每完成一步就更新 REPORT.md 并 commit + push，数字写在前面，简洁。
- 遇到 go/no-go 不达标或数据拿不到，就停下来汇报，不要硬凑。
- 用中文和我交流。
