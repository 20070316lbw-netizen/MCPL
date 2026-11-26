---
id: mcpl-leakage
title: Leakage Recovery
source_type: github
source_url: https://github.com/20070316lbw-netizen/MCPL
year: 2026
authors:
- MCPL
sector: equities
domain: cross_sectional
universe: 历史成分快照
horizon: daily
target_object: 单股次日开盘到收盘对数收益
model_family: other
estimation_method: 配对 MLP 与按日 HAC
data_required:
- prices
- universe
- calendar
feasibility: ready
one_liner: 检验暴露于标签的模型是否需要更多干净数据恢复
key_highlight: 固定测试输入与测试时间, 区分即时表现与干净补训效果
key_risk: 弱信号及单一市场年份限制可识别性
tags:
- mlp
- leakage
status: in_progress
scouted_at: '2026-11-26'
experimented: true
experiments:
- name: leakage_recovery
  config_file: configs/leakage.yaml
  report_path: outputs/leakage/report.md
  ran_at: '2026-11-26'
  outcome: inconclusive
  summary: null
---

# Leakage Recovery

## 1. 核心直觉
模型在预训练时获得明天的真实收益, 可能减少对合法历史信息的利用。移除这条捷径后, 我们观察它与全程干净模型之间的误差差距, 并检验有限干净数据能否缩小差距。

## 2. 研究对象的核心建模
目标 y = 100 × log(close/open), 取下一交易日的开盘与收盘。三个分支共享网络结构和补训批次, 在固定测试年比较逐股绝对误差, 再按日配对。

## 3. 数据映射与复现改造
按特征日匹配最新历史成分快照。先在完整交易日历构建特征和标签, 只在最后应用成分掩码, 保留下期仍有价格的退出股票。跨窗口标签与行情缺口分别剔除。

## 4. 亮点, 假设与失效边界
完整标签泄漏是机制实验的极端设定。误差差距消失可能源于干净模型本身缺少有效信号, 随机种子不能增加独立测试年份。保持测试全年相同以控制市场状态。

## 5. 实证结论
2025 年固定测试窗内, 12 个月补训后泄漏组比干净组差 4.13 bp, 比重新初始化差 2.90 bp。零补训差距无法与零区分, 曲线不单调, 单一测试年不足以确定恢复时间。详见 outputs/leakage/report.md。
