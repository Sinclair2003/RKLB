# RKLB Quant Research

一个基于 `RKLB` 日线数据的量化研究项目，包含：

- 数据抓取（任意 ticker）
- Qlib 风格研究框架（特征/模型/策略/回测）
- 明日价格预测（基础版 + 增强版 + walk-forward）

## 1. 项目结构

- `fetch_any_tickers.py`：抓取任意股票/指数/期货的日线 OHLCV
- `fetch_daily_data.py`：多资产抓取脚本（固定资产组合）
- `qlib_like_rklb_framework.py`：Qlib 风格轻量框架（回测导向）
- `qlib_like_rklb_framework.ipynb`：上面框架的 Notebook 版本
- `predict_next_close_rklb.py`：RKLB 明日价格预测（增强版，含外部因子 + walk-forward）
- `market_data/`：原始数据与外部因子缓存
- `outputs/`：模型输出结果

## 2. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas yfinance numpy
```

如果你使用本仓库现成虚拟环境：

```bash
source .venv/bin/activate
```

## 3. 数据抓取

### 3.1 抓任意代码

```bash
python fetch_any_tickers.py --symbols RKLB
python fetch_any_tickers.py --symbols RKLB,SPY,QQQ --start 2020-01-01
```

输出默认在：`market_data/`。

### 3.2 固定资产组合

```bash
python fetch_daily_data.py
```

## 4. Qlib 风格回测研究

```bash
python qlib_like_rklb_framework.py --csv market_data/rklb_daily.csv
```

输出目录：`outputs/rklb_qlib_like/`

- `metrics_summary.csv`
- `metrics.json`
- `backtest_detail.csv`

Notebook 版本：`qlib_like_rklb_framework.ipynb`

## 5. 明日价格预测（增强版）

```bash
python predict_next_close_rklb.py --refresh-context
```

该脚本会自动拉取并缓存外部市场因子（如 `SPY/QQQ/VIX/TNX/DXY/CL/GC`）到：

- `market_data/context_cache/`

输出目录：`outputs/rklb_price_forecast_v2/`

- `forecast.json`：下一交易日预测结果
- `metrics_summary.csv`：固定切分指标
- `predictions_detail.csv`：逐日预测明细
- `walk_forward_summary.json`：滚动窗口评估摘要
- `walk_forward_detail.csv`：滚动窗口逐日明细

## 6. 模型说明（predict_next_close_rklb.py）

- 预测目标：`next-day return`（而非直接回归价格）
- 模型：线性 Ridge + 非线性 RFF Ridge 集成
- 调参：在验证集上搜索 `alpha / gamma / blend`
- 评估：固定切分 + walk-forward（更接近实盘）

## 7. 常用命令

```bash
# 语法检查
python -m py_compile fetch_any_tickers.py
python -m py_compile qlib_like_rklb_framework.py
python -m py_compile predict_next_close_rklb.py

# 运行增强预测
python predict_next_close_rklb.py --refresh-context
```

## 8. GitHub 上传（本地）

```bash
git add -A
git commit -m "update README"
git push origin main
```

如使用 PAT 推送：

```bash
git push https://$GITHUB_TOKEN@github.com/Sinclair2003/RKLB.git main
```

