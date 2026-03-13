# Theory Verification Checklist

- [x] `check_lemma_rankn.py`：验证 Appendix 中 rank-\(n\) 引理（行列式与逆矩阵公式）
  - 结果：PASS（`trials=200`，`max_det_rel=3.08e-15`，`max_inv_rel=1.09e-15`）

- [x] `check_pcca_schur_pd.py`：验证 PCCA 情况下 Schur 补闭式与正定性
  - 结果：PASS（`trials=200`，`max_rel=5.70e-15`，`min_eig_schur=1.38e-02`）

- [x] `check_scalar_expansion.py`：验证 Theorem A（标量展开目标函数）与矩阵形式一致
  - 结果：PASS（`trials=120`，`max_abs_diff=5.15e-14`）

- [x] `check_bcd_propositions.py`：验证 Proposition 4/5（\(\theta_t^2\) 闭式更新与 \(b\) 三次方程/唯一正根）
  - 结果：PASS（`trials=300`，`max_theta_grad_fd=8.88e-10`，`max_cubic_residual=4.66e-15`，`positive_root_uniqueness_cases=300`）

- [x] `check_noise_theorem.py`：经验验证噪声谱估计理论界（高概率与期望阶）
  - 结果：PASS（`trials=300`，`mae=3.04e-03`，`expectation_bound_rhs=3.90e-03`，`bound_hit_rate=0.903`）

## Run command

```bash
python C:\Users\ASUS\Desktop\ECAI-PPLS-SLM\theory_checks\run_all_checks.py
```

## Output file

- 汇总结果：`theory_checks/results.json`
