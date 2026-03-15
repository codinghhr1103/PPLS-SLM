# PPLS-SLM

`PPLS-SLM` 是一个面向论文复现实验的 Python 研究代码库，围绕 Probabilistic Partial Least Squares (PPLS) 的参数估计、预测和真实数据应用展开。当前仓库实现的核心方法包括：

- `SLM-Manifold`
- `BCD-SLM`
- `SLM (trust-constr)`
- `EM / ECM`

仓库重点不是“可安装的通用软件包”，而是从项目根目录直接运行的实验代码，覆盖：

- Monte Carlo 参数恢复实验
- 标量似然与矩阵似然速度对比
- 合成数据预测实验
- BRCA 多组学预测与校准
- CITE-seq 蛋白预测与校准
- 关联分析
- PCCA / PPCA 辅助实验
- 理论命题数值检查

- 论文产物同步与表格生成

## 环境准备

建议在项目根目录创建虚拟环境后安装依赖：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

补充说明：

- `SLM-Manifold` 与 `BCD-SLM` 依赖 `pymanopt`。
- 如果需要从 `.h5ad` 预处理 CITE-seq 数据，还需要额外安装 `anndata`；部分流程也可能需要 `scanpy` / `muon`。
- 如果需要编译 `paper/` 下的论文，还需要系统安装 `pdflatex`。

## 快速开始

以下命令默认都在项目根目录执行。

### 1. 一键运行主要实验

```bash
python scripts/run_all_experiments.py --config config.json
```

常用可选参数：

```bash
python scripts/run_all_experiments.py --config config.json --clean
python scripts/run_all_experiments.py --config config.json --skip-citeseq --skip-brca
python scripts/run_all_experiments.py --config config.json --clean-artifacts
```

说明：该脚本会临时将 `config.json` 中的 `output.force_*` 设为 `true` 以确保重跑，结束后再恢复原值。

### 2. 单独运行核心实验

#### Monte Carlo

```bash
python -m ppls_slm.cli.montecarlo --config config.json
```

#### Speed benchmark

```bash
python -m ppls_slm.benchmarks.speed_experiment
```

#### 合成数据预测

```bash
python -m ppls_slm.apps.prediction --config config.json
```

#### BRCA 预测 / 校准

```bash
python -m ppls_slm.apps.brca_prediction --config config.json
python -m ppls_slm.apps.brca_calibration --config config.json
```

快速调试：

```bash
python -m ppls_slm.apps.brca_prediction --config config.json --smoke
```

#### CITE-seq 预测 / 校准

```bash
python -m ppls_slm.apps.citeseq_prediction --config config.json
python -m ppls_slm.apps.citeseq_prediction --config config.json --smoke
```

便捷脚本：

```bash
python scripts/run_citeseq.py --config config.json
```

#### Association 分析

```bash
python -m ppls_slm.apps.association_analysis --brca_data application/brca_data_w_subtypes.csv.zip --output_dir results_association --plot
```

#### PCCA / PPCA

```bash
python -m ppls_slm.apps.pcca_simulation --config config.json
python -m ppls_slm.apps.ppca_verification --config config.json
```

便捷脚本：

```bash
python scripts/run_pcca_experiment.py --config config.json
python scripts/run_ppca_verification.py --config config.json
```

#### Smoke / Theory checks

```bash
python -m ppls_slm.cli.smoke
python theory_checks/run_all_checks.py
```


## 论文实验与 README 实验对照

如果你的目标是“按论文复现实验并刷新论文中的表格/图”，建议使用以下流程：

1. 先运行对应实验命令；
2. 再执行：

```bash
python scripts/sync_artifacts.py
```

这样会把结果同步到 `paper/artifacts/`，并刷新 `paper/generated/tables/` 下的论文表格片段。若还要重新编译论文，再执行：

```bash
python scripts/compile_paper_pdflatex.py
```

另外，如果希望一次性跑完论文中的主要实验，可直接使用：

```bash
python scripts/run_all_experiments.py --config config.json
```

它会依次运行 Monte Carlo、speed benchmark、association、synthetic prediction、BRCA prediction/calibration、CITE-seq、PCCA、PPCA，并在最后同步论文产物。

### 逐项对应关系

| 论文位置 | 论文中的实验/结果 | README 中对应实验 | 对应脚本 | 复现命令 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 第 8 节 `Simulation`；表 `tab:parameter_mse` | 主 Monte Carlo 参数恢复实验（低噪声 + 高噪声） | `Monte Carlo` | `ppls_slm/cli/montecarlo.py` | `python -m ppls_slm.cli.montecarlo --config config.json` | 主配置来自 `config.json`；同一次运行会产出正文主表所需结果。 |
| 附录 `Convergence statistics`；表 `tab:algorithm_convergence` | SLM-Manifold 与 BCD-SLM 收敛迭代数比较 | `Monte Carlo` | `ppls_slm/cli/montecarlo.py` | `python -m ppls_slm.cli.montecarlo --config config.json` | 不是独立实验，而是由 Monte Carlo 同次运行派生出的汇总表。 |
| 附录附加 loading 恢复图 | `W`/`C` loading 恢复图 | `Monte Carlo` | `ppls_slm/cli/montecarlo.py` | `python -m ppls_slm.cli.montecarlo --config config.json` | 同样来自 Monte Carlo 输出。 |
| 附录 `Speedup heatmap`；图 `fig:speed_comparison` | 标量似然 vs 矩阵似然速度热图 | `Speed benchmark` | `ppls_slm/benchmarks/speed_experiment.py` | `python -m ppls_slm.benchmarks.speed_experiment` | 当前脚本默认输出到 `output/figures/speed_comparison.png`。 |
| 第 8 节 `Synthetic Prediction and Calibration`；表 `tab:pred_synth_summary` | 合成数据预测与区间校准 | `合成数据预测` | `ppls_slm/apps/prediction.py` | `python -m ppls_slm.apps.prediction --config config.json` | 当前配置写入 `results_prediction_hd/`；而 `scripts/sync_artifacts.py` 仍默认读取 `results_prediction/`，同步论文表格前需注意该路径差异。 |
| 第 8 节 `TCGA-BRCA Multi-Omics` 与附录 `BRCA prediction and calibration`；表 `tab:pred_brca_summary` | BRCA 预测与校准 | `BRCA 预测 / 校准` | `ppls_slm/apps/brca_prediction.py`、`ppls_slm/apps/brca_calibration.py` | `python -m ppls_slm.apps.brca_prediction --config config.json`<br>`python -m ppls_slm.apps.brca_calibration --config config.json` | 校准步骤依赖前一步生成的 `results_prediction_brca/brca_prediction_summary.csv`。 |
| 第 8 节 `TCGA-BRCA Multi-Omics` 与附录 `Association screening on TCGA-BRCA`；表 `tab:Npairs` | BRCA 关联筛选 | `Association 分析` | `ppls_slm/apps/association_analysis.py` | `python -m ppls_slm.apps.association_analysis --brca_data application/brca_data_w_subtypes.csv.zip --output_dir results_association --plot` | 主表对应 `tab:Npairs`；`top 10` 对应生成文件 `paper/generated/tables/tab_top10_pairs.tex`。该脚本默认做子采样分析；若要严格对齐论文中的 `SLM-Manifold` 表述，建议先核对脚本内 `ScalarLikelihoodMethod` 的 `optimizer` 设置。 |
| 第 8 节 `Single-Cell CITE-seq Protein Imputation`；表 `tab:citeseq_pred_summary` | CITE-seq 蛋白预测主结果 | `CITE-seq 预测 / 校准` | `ppls_slm/apps/citeseq_prediction.py` | `python -m ppls_slm.apps.citeseq_prediction --config config.json` | 同一次运行会同时生成预测指标、校准汇总和 loading 解释结果。 |
| 第 8 节 `Single-Cell CITE-seq Protein Imputation` 正文覆盖率描述 | CITE-seq 预测区间校准 | `CITE-seq 预测 / 校准` | `ppls_slm/apps/citeseq_prediction.py` | `python -m ppls_slm.apps.citeseq_prediction --config config.json` | 覆盖率结果写入 `results_citeseq/citeseq_calibration_summary.csv`。 |
| 第 8 节 `Single-Cell CITE-seq Protein Imputation` 末段与附录 `Single-cell CITE-seq experiment details` | CITE-seq loading 可解释性分析 | `CITE-seq 预测 / 校准` | `ppls_slm/apps/citeseq_prediction.py` | `python -m ppls_slm.apps.citeseq_prediction --config config.json` | 对应输出包含 `citeseq_loadings_top.csv`。 |
| 第 8 节 Simulation 末段与附录 `PCCA specialization simulation`；表 `tab:pcca_parameter_mse` | PCCA 特化仿真 | `PCCA / PPCA` | `ppls_slm/apps/pcca_simulation.py` | `python -m ppls_slm.apps.pcca_simulation --config config.json` | 也可用便捷脚本 `python scripts/run_pcca_experiment.py --config config.json`。 |
| 第 8 节 Simulation 末段与附录 `PPCA noise variance estimation verification`；表 `tab:ppca_noise_verification` | PPCA 噪声方差验证 | `PCCA / PPCA` | `ppls_slm/apps/ppca_verification.py` | `python -m ppls_slm.apps.ppca_verification --config config.json` | 也可用便捷脚本 `python scripts/run_ppca_verification.py --config config.json`。 |



## 目录说明


### 核心代码

- `ppls_slm/ppls_model.py`：PPLS 模型、协方差结构、标量似然、噪声估计。
- `ppls_slm/algorithms.py`：`SLM`、`EM`、`ECM` 的统一实现入口。
- `ppls_slm/slm_manifold.py`：基于流形优化的 `SLM-Manifold`。
- `ppls_slm/bcd_slm.py`：`BCD-SLM`。
- `ppls_slm/data_generator.py`：模拟数据生成。
- `ppls_slm/experiment.py`：Monte Carlo 实验编排、并行与汇总。
- `ppls_slm/visualization.py`：结果绘图与导出。

### 应用与实验入口

- `ppls_slm/apps/`：预测、校准、关联分析、PCCA / PPCA 等应用实验。
- `ppls_slm/cli/`：Monte Carlo、smoke test 等命令行入口。

- `ppls_slm/benchmarks/`：速度基准实验。
- `scripts/`：一键运行、CITE-seq 数据准备、论文产物同步、表格生成、论文编译。
- `theory_checks/`：理论命题数值校验脚本。

### 数据与结果目录

- `application/`：仓库自带输入数据，如 `brca_data_w_subtypes.csv.zip`、`citeseq_rna.csv`、`citeseq_adt.csv`。
- `data/`：额外原始数据或缓存数据。
- `output/`：Monte Carlo、PCCA、PPCA、日志、图像等输出。
- `results_citeseq/`：CITE-seq 实验结果。
- `paper/`：论文源码、生成片段和同步后的产物。
- `notes/`：运行记录与备忘。

## 数据格式约定

### BRCA

默认 BRCA 数据文件为：

```text
application/brca_data_w_subtypes.csv.zip
```

代码假定：

- 基因表达列以前缀 `rs_` 开头
- 蛋白列以前缀 `pp_` 开头

### CITE-seq

CITE-seq 输入支持以下形式之一：

- 单个 `.h5ad`
- 包含 `citeseq_rna.csv` 和 `citeseq_adt.csv` 的目录
- 单个 `.csv` / `.csv.zip`，其中列名前缀为 `rna_*` 和 `adt_*`

数据准备脚本：

```bash
python scripts/prepare_citeseq.py --scvi-hao-pbmc
```

或使用本地 `.h5ad`：

```bash
python scripts/prepare_citeseq.py --input data/pbmc_seurat_v4.h5ad
```

## 结果输出位置

当前代码中的主要输出位置如下：

- Monte Carlo：`output/data/`、`output/data_high/`、`output/results/`、`output/results_high/`、`output/figures/`、`output/figures_high/`
- Speed benchmark：`output/figures/speed_comparison.png`
- 合成预测：默认来自 `config.json`，当前配置为 `results_prediction_hd/`
- BRCA 预测 / 校准：`results_prediction_brca/`
- Association：`results_association/`
- CITE-seq：`results_citeseq/`

- PCCA：`output/pcca_simulation/`
- PPCA：`output/ppca_verification/`
- 一键运行日志：`output/logs/one_click/`
- 论文产物：`paper/artifacts/` 与 `paper/generated/`

## 运行建议

- 大多数主实验都把 `config.json` 作为单一配置来源，命令行参数只覆盖局部设置。
- 该仓库没有 `pyproject.toml` / `setup.py`，推荐直接在仓库根目录使用 `python -m ...` 或 `python scripts/...` 运行。
- Windows 下已考虑多进程 `spawn` 场景，但长时间实验仍建议控制并行度，避免线程/进程过度竞争。
- 一些脚本会主动限制 BLAS / OMP 线程数，以减少卡顿和过度占用。

## 测试

目前仓库内显式的 pytest 测试较少，可运行：

```bash
pytest ppls_slm/tests/test_bcd_slm.py
```

除此之外，项目的主要验证方式仍是实验输出、理论检查脚本和论文产物。
