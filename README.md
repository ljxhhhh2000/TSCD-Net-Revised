# TSCD-Net Revised 有序窗口与逐帖统计模型

这个项目输入一名用户的微博文本与发帖时间，输出该用户的 SWDD 正类预测分数和分类结果。模型把逐帖症状分数送入两条分支。时序分支用 **最多 16 个连续窗口**学习窗口内的症状组合和窗口间的先后关系。统计分支从原始逐帖分数提取均值、标准差和相邻变化幅度。

主流程可以在**单独取得此项目和 SWDD 数据**后运行。步骤从生成特征开始，既可以本项目自行训练，也可以复用兼容的特征缓存。只有后面的 DSTS 配对脚本需要基线项目。 [架构、公式与逐轮实验结果](MODEL_DESIGN.md)

## 先选你要做的事

| 想做什么 | 从这里开始 |
|---|---|
| 确认环境可用 | [安装和测试](#安装和测试) |
| 用少量微博走通训练并生成一份权重 | [快速开始](#快速开始) |
| 跑完整 15 症状主实验 | [完整训练](#完整训练) |
| 加载训练好的模型或查看预测 | [输出和权重](#输出和权重) |
| 在同批用户上与 DSTS-InceptionTime 比较 | [可选实验](#可选实验) |

## 模型和数据流

```text
微博与时间戳 → 冻结 SBERT → 逐帖症状分数与对数时间间隔
  ├── 连续分窗 → 症状注意力 → 位置编码 → 跨窗口注意力 → 时序分类分数
  └── 逐帖均值、标准差、相邻绝对差 → MLP → 统计分类分数
两个 logit 相加 → sigmoid → 用户级分类
```

模型按帖子数分窗，再汇总症状分数与时间间隔。学习到的位置编码让模型能区分前期与后期。时序分支处理的序列较短，统计分支继续利用窗口内部的逐帖波动。主实验使用 **15 个症状通道**，DSTS 配对实验采用基线缓存的 **11 个症状通道**。加载权重时应使用该轮训练对应的通道顺序和模型配置。

## 安装和测试

在本项目根目录运行以下命令。推荐 Python 3.10。要复用已验证的固定 CUDA 12.1 Python 依赖，使用 `requirements.txt`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m tscd_net --help
python -m pytest -q
```

CPU 或已有 PyTorch 的环境可用 `python -m pip install -e ".[cjk,test]"`。测试使用构造输入，涵盖分窗、补齐掩码、顺序敏感性和阈值计算等行为，可以在准备语料前执行。 [包配置](pyproject.toml)

## 快速开始

### 1. 准备数据

从 SWDD 发布页获取语料，使用包含 `control.jsonl` 和 `depressed.jsonl` 的解压目录，或包括这两个成员的原始 ZIP。每行是一名用户的微博记录，模型读取用户标签、文本与发帖时间。主配置按用户组织帖子并保留原始类别比例。 [SWDD 数据发布页](https://github.com/ethan-nicholas-tsai/SWDD) · [读取与筛选规则](src/tscd_net/data.py)

### 2. 建一个自己的配置

```powershell
Copy-Item configs/chronological.json configs/chronological.local.json
```

在 `configs/chronological.local.json` 中修改以下字段。`data` 小节在当前模板中可以省略，想用自己的数据路径时，在 JSON 顶层添加 `"data": {"dataset_zip": "D:/data/SWDD"}`。其余配置可保留模板值。

| 字段 | 怎样填写 |
|---|---|
| `data.dataset_zip` | 解压目录或 ZIP 的绝对路径 |
| `features.embedding_model` | 本地句向量模型目录，或 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `features.embedding_device` | 特征提取设备，CPU 快速检查时设 `cpu` |
| `model.device` | 分类器设备，可设 `auto`、`cpu` 或 `cuda` |
| `experiment.artifact_dir` | 独立项目可以用 `artifacts/main` |
| `experiment.output_dir` | 如 `runs/chronological-local` |

模型标识首次使用时需要获取模型文件。已有本地模型目录可直接填绝对路径。`--device` 只覆盖分类器设备，特征提取设备由 `features.embedding_device` 控制。

### 3. 跑一轮小样本训练

```powershell
python -m tscd_net all --config configs/chronological.local.json --scan-limit-per-file 50 --max-users-per-class 40 --sequence-length 120 --runs 1 --epochs 1 --device cpu --artifact-dir artifacts/quickstart --output-dir runs/quickstart
Get-Content runs/quickstart/metrics_summary.csv
```

这条命令先从小样本预计算特征，再跑一轮轻量训练。它会生成 `artifacts/quickstart/features.npz`，以及 `runs/quickstart/` 下的指标和权重。若使用经过额外筛选的数据，确保样本中保留足够的两类用户供分层划分。小样本结果只用于验证训练流程。

## 完整训练

使用上面准备的本地配置，去掉小样本命令中的扫描、用户数、步数和 epoch 限制。先提取正式特征，再运行正常训练。

```powershell
python -m tscd_net prepare --config configs/chronological.local.json
python -m tscd_net train --config configs/chronological.local.json
```

`prepare` 保存逐帖症状分数、时间差、用户标签、长度和症状向量。`train` 使用同一配置读取缓存，默认按 **70%／10%／20%** 做用户级分层划分，重复 5 个种子，每轮最多 30 个 epoch。主实验按验证集 F1 选择权重和阈值，测试集计算 Accuracy、Precision、Recall 和 F1。当前工作区有兼容的 15 维特征缓存，主配置已指向 `../../Dataset/TSCD-Net-Features/main`，可以直接运行如下短训练。

```powershell
python -m tscd_net train --config configs/chronological.json --runs 1 --epochs 1 --output-dir runs/check-main-cache
```

同一配置可从已完成的逐用户特征或已完成轮次继续。准备不同数据集、改变症状集合或修改编码器时，配置对应的新缓存目录。特别是复用缓存时，`features.embedding_model` 应与缓存的记录保持一致。

## 输出和权重

| 文件 | 内容 |
|---|---|
| 特征目录下的 `features.npz` | 用户的症状分数、时间间隔、长度和标签等 |
| `runs/<实验名>/run_config.json` | 全部运行参数 |
| `runs/<实验名>/metrics_by_run.jsonl` | 每次划分的指标和测试混淆矩阵 |
| `runs/<实验名>/metrics_summary.csv` | 按方法汇总的均值和标准差 |
| `runs/<实验名>/predictions/run_*.pt` | 每轮验证集选出的最佳 PyTorch 权重 |
| `runs/<实验名>/predictions/run_*.npz` | 同轮验证与测试用户、标签、正类分数和阈值 |

`results/` 中保存当前工作区已经完成的实验。下面的读取示例针对上面的 **15 维快速验证**，使用当轮记录的配置、特征和阈值对缓存样本做推理。

```python
import numpy as np
import torch
from pathlib import Path

from tscd_net.config import Config
from tscd_net.features import load_features
from tscd_net.models.registry import build_classifier
from tscd_net.preprocess import crop_series

run_dir = Path("runs/quickstart")
config = Config.from_json(run_dir / "run_config.json")
features = load_features(config.artifact_dir / "features.npz")
features = crop_series(features, config.features.sequence_length)
name = config.experiment.models[0]
stem = run_dir / "predictions" / f"run_00_{name}"

config.model.device = "cpu"
classifier = build_classifier(
    name, config.model, config.training, config.training.master_seed,
    features.symptom_embeddings,
)
classifier.model.load_state_dict(
    torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
)
with np.load(stem.with_suffix(".npz"), allow_pickle=False) as saved:
    threshold = float(saved["threshold"])
    ids = saved["test_user_ids"][:4]
    lookup = {user_id: i for i, user_id in enumerate(features.user_ids)}
    rows = [lookup[user_id] for user_id in ids]
    scores = classifier.predict_proba(
        features.symptom_scores[rows], features.log_time_gaps[rows],
        features.lengths[rows],
    )
print(scores, classifier.classify(scores, threshold))
```

11 维配对实验另有 `dsts_alignment.npz`，推理时仍需要与之配套的 DSTS 原始分数。模型权重和输入通道绑定，每个实验分别使用自己的 `run_config.json`。

## 可选实验

### 与完整逐帖 DSTS-InceptionTime 配对

做同口径比较时，先确认 DSTS 主实验缓存和指标已就绪。配对脚本读取同级 DSTS 项目已有的 **7,000 名用户、11 维原创帖症状特征和十轮基线指标**，再训练本项目的配对模型。它用原种子 2023–2032 重建 60%／20%／20% 划分，新版按验证集 Accuracy 选权重，用固定 0.5 边界测试。每轮测试 1,400 人，正类和负类各 700 人。 [工作区内的 DSTS 项目说明](../depression%20detect/README.md)

```powershell
python scripts/compare_dsts_protocol.py all
```

需要同级 `depression detect/artifacts/main/features.npz`、`depression detect/runs/main/run_config.json` 和 `depression detect/runs/main/metrics_by_run.jsonl`。对齐时间还需要原始 SWDD，也可使用本项目已归档的对齐文件。脚本默认把新结果写入 `runs/dsts_protocol`，包括逐轮对比的 `paired_metrics.csv`、模型权重和预测。 [配对方案与指标解释](MODEL_DESIGN.md#8-swdd-7k-配对实验如何做) · [配对脚本](scripts/compare_dsts_protocol.py)

### 症状、时间和文字特征对照

| 入口 | 内容与依赖 |
|---|---|
| [compare_chronological.py](scripts/compare_chronological.py) | 用 `--population observed` 比较症状均值与统计模型。`balanced` 还需要 DSTS 缓存的用户清单，可对比 16 窗口 InceptionTime 和去时间间隔的新模型。 |
| [fuse_chronological.py](scripts/fuse_chronological.py) | 用同种子、同用户的时序与症状统计预测，在验证集选择融合权重。先生成两种预测。 |
| [compare_text_features.py](scripts/compare_text_features.py) | 从原始微博提取汉字片段及 TF-IDF，和已生成的症状统计预测比较或融合。 |

下面用本地 15 维缓存生成原比例下的症状统计对照。

```powershell
python scripts/compare_chronological.py --config configs/chronological.local.json --population observed --output runs/chronological_controls --models symptom_means symptom_statistics symptom_statistics_boosted --runs 5
```

## 已有结果怎么读

| 实验 | 口径 | 平均 F1 |
|---|---|---:|
| 新版主实验 | 原比例，15 维，5 轮，验证集选阈值 | 82.11% |
| 新版平衡对照 | 共同用户的平衡子集，15 维，3 轮 | 90.28% |
| 新版 SWDD-7K 配对 | 7,000 人，11 维，10 轮，固定 0.5 | **91.00%** |
| 本地 DSTS-InceptionTime | 上一行相同用户和划分，历史结果 | 90.51% |

按对应协议比较指标。配对结果呈现整套设置的差异，各模块贡献需要在相同协议下逐项消融。工作区完整配对结果位于 `results/dsts_protocol/`。 [完整结果说明](MODEL_DESIGN.md)

## 代码入口

| 路径 | 用途 |
|---|---|
| [configs/](configs) | 主模型、数据和训练参数 |
| [cli.py](src/tscd_net/cli.py) · [experiment.py](src/tscd_net/experiment.py) | 命令行、缓存和训练流程 |
| [features.py](src/tscd_net/features.py) | SBERT 症状分数、时间差和保存格式 |
| [tscdnet.py](src/tscd_net/models/tscdnet.py) | 两级注意力、位置编码和预测融合 |
| [chronological.py](src/tscd_net/models/chronological.py) | 按帖子分窗和逐帖统计 |
| [torch_trainer.py](src/tscd_net/models/torch_trainer.py) | 优化、早停、验证选权重及推理 |
| [tests/](tests) | 用小型输入检查数据、掩码和模型行为 |
