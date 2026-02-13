"""
Qlib-style lightweight quant research framework for RKLB.

Design (inspired by Qlib classic workflow):
1) Data Layer: DataHandlerCSV
2) Feature Layer: AlphaFeatureEngineer
3) Model Layer: RidgeReturnModel
4) Strategy Layer: ThresholdLongOnlyStrategy
5) Backtest + Evaluation Layer: Backtester, Evaluator, Recorder

Run:
  .venv/bin/python qlib_like_rklb_framework.py
  .venv/bin/python qlib_like_rklb_framework.py --csv market_data/rklb_daily.csv --horizon 1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
@dataclass
class FrameworkConfig:
    csv_path: str = "market_data/rklb_daily.csv"
    output_dir: str = "outputs/rklb_qlib_like"
    horizon: int = 1
    # Date split can be tuned by user.
    train_end: str = "2023-12-31"
    valid_end: str = "2024-12-31"
    test_end: str = "2099-12-31"
    transaction_cost_bps: float = 5.0
    annualization: int = 252
    ridge_alpha: float = 5.0
    # Threshold candidates for valid-set tuning.
    threshold_grid: Tuple[float, ...] = (-0.005, -0.002, 0.0, 0.002, 0.005)


# =========================
# Data Layer
# =========================
class DataHandlerCSV:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        needed = {"date", "open", "high", "low", "close", "volume"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")
        df = df.set_index("date")

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])  # close is mandatory
        return df


# =========================
# Feature Layer
# =========================
class AlphaFeatureEngineer:
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window).mean()

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
        out = df.copy()

        out["ret_1"] = out["close"].pct_change(1)
        out["ret_5"] = out["close"].pct_change(5)
        out["ret_10"] = out["close"].pct_change(10)

        out["vol_5"] = out["ret_1"].rolling(5).std()
        out["vol_20"] = out["ret_1"].rolling(20).std()

        for win in (5, 10, 20, 60):
            ma = out["close"].rolling(win).mean()
            out[f"ma_ratio_{win}"] = out["close"] / ma - 1

        out["rsi_14"] = self._rsi(out["close"], 14)

        atr_14 = self._atr(out, 14)
        out["atr_norm_14"] = atr_14 / out["close"].replace(0, np.nan)

        log_vol = np.log1p(out["volume"].clip(lower=0))
        out["vol_z_20"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()

        label_col = f"label_ret_fwd_{self.horizon}d"
        out[label_col] = out["close"].shift(-self.horizon) / out["close"] - 1.0

        feature_cols = [
            "ret_1",
            "ret_5",
            "ret_10",
            "vol_5",
            "vol_20",
            "ma_ratio_5",
            "ma_ratio_10",
            "ma_ratio_20",
            "ma_ratio_60",
            "rsi_14",
            "atr_norm_14",
            "vol_z_20",
        ]

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=feature_cols + [label_col])
        return out, feature_cols, label_col


# =========================
# Model Layer
# =========================
class RidgeReturnModel:
    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_aug = np.hstack([ones, X])

        reg = np.eye(X_aug.shape[1]) * self.alpha
        reg[0, 0] = 0.0  # no penalty on intercept

        xtx = X_aug.T @ X_aug
        xty = X_aug.T @ y
        self.weights = np.linalg.solve(xtx + reg, xty)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted")
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_aug = np.hstack([ones, X])
        return X_aug @ self.weights


# =========================
# Strategy Layer
# =========================
class ThresholdLongOnlyStrategy:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def generate_position(self, pred: pd.Series) -> pd.Series:
        # Signal at t based on prediction for t->t+1 return.
        return (pred > self.threshold).astype(float)


# =========================
# Backtest & Eval Layer
# =========================
class Backtester:
    def __init__(self, cost_bps: float = 5.0):
        self.cost_rate = cost_bps / 10000.0

    def run(self, df: pd.DataFrame, pred_col: str, label_col: str, threshold: float) -> pd.DataFrame:
        out = df.copy()
        strategy = ThresholdLongOnlyStrategy(threshold)
        out["position"] = strategy.generate_position(out[pred_col])

        # Position decided at close(t), applied to next-day return represented by label at t.
        out["gross_ret"] = out["position"] * out[label_col]

        turnover = (out["position"] - out["position"].shift(1).fillna(0)).abs()
        out["cost"] = turnover * self.cost_rate
        out["strategy_ret"] = out["gross_ret"] - out["cost"]
        out["bench_ret"] = out[label_col]

        out["strategy_nav"] = (1 + out["strategy_ret"]).cumprod()
        out["bench_nav"] = (1 + out["bench_ret"]).cumprod()
        return out


class Evaluator:
    def __init__(self, annualization: int = 252):
        self.annualization = annualization

    @staticmethod
    def _max_drawdown(nav: pd.Series) -> float:
        peak = nav.cummax()
        dd = nav / peak - 1.0
        return float(dd.min())

    def summarize(self, ret: pd.Series, nav: pd.Series) -> Dict[str, float]:
        ret = ret.dropna()
        if ret.empty:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_vol": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        total_return = float(nav.iloc[-1] - 1.0)
        ann_ret = float((1 + ret.mean()) ** self.annualization - 1)
        ann_vol = float(ret.std(ddof=0) * np.sqrt(self.annualization))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
        mdd = self._max_drawdown(nav)
        win_rate = float((ret > 0).mean())
        return {
            "total_return": total_return,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "win_rate": win_rate,
        }

    @staticmethod
    def info_coef(pred: pd.Series, label: pd.Series) -> Dict[str, float]:
        aligned = pd.concat([pred, label], axis=1).dropna()
        if aligned.empty:
            return {"ic_pearson": 0.0, "ic_spearman": 0.0}

        p = aligned.iloc[:, 0]
        y = aligned.iloc[:, 1]
        # Spearman via rank-correlation to avoid scipy dependency.
        p_rank = p.rank(method="average")
        y_rank = y.rank(method="average")
        return {
            "ic_pearson": float(p.corr(y, method="pearson")),
            "ic_spearman": float(p_rank.corr(y_rank, method="pearson")),
        }


class Recorder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_table(self, df: pd.DataFrame, name: str) -> str:
        path = os.path.join(self.output_dir, name)
        df.to_csv(path)
        return path

    def save_json(self, obj: Dict, name: str) -> str:
        path = os.path.join(self.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path


# =========================
# Orchestration (Qlib-style workflow)
# =========================
def split_by_time(df: pd.DataFrame, train_end: str, valid_end: str, test_end: str) -> Dict[str, pd.DataFrame]:
    train_end_ts = pd.to_datetime(train_end)
    valid_end_ts = pd.to_datetime(valid_end)
    test_end_ts = pd.to_datetime(test_end)

    train = df[df.index <= train_end_ts]
    valid = df[(df.index > train_end_ts) & (df.index <= valid_end_ts)]
    test = df[(df.index > valid_end_ts) & (df.index <= test_end_ts)]
    return {"train": train, "valid": valid, "test": test}


def tune_threshold(valid_df: pd.DataFrame, backtester: Backtester, evaluator: Evaluator, label_col: str, grid: Tuple[float, ...]) -> float:
    if valid_df.empty:
        return 0.0

    best_th = 0.0
    best_sharpe = -np.inf
    for th in grid:
        bt = backtester.run(valid_df, pred_col="pred", label_col=label_col, threshold=th)
        metrics = evaluator.summarize(bt["strategy_ret"], bt["strategy_nav"])
        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_th = th
    return float(best_th)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qlib-style RKLB quant research framework")
    parser.add_argument("--csv", default="market_data/rklb_daily.csv", help="Path to OHLCV CSV")
    parser.add_argument("--output-dir", default="outputs/rklb_qlib_like", help="Output directory")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in trading days")
    parser.add_argument("--train-end", default="2023-12-31", help="Train end date")
    parser.add_argument("--valid-end", default="2024-12-31", help="Validation end date")
    parser.add_argument("--test-end", default="2099-12-31", help="Test end date")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in bps per turnover")
    parser.add_argument("--ridge-alpha", type=float, default=5.0, help="Ridge regularization strength")
    return parser


def run_pipeline(cfg: FrameworkConfig) -> Dict[str, Dict[str, float]]:
    handler = DataHandlerCSV(cfg.csv_path)
    feat = AlphaFeatureEngineer(horizon=cfg.horizon)
    model = RidgeReturnModel(alpha=cfg.ridge_alpha)
    backtester = Backtester(cost_bps=cfg.transaction_cost_bps)
    evaluator = Evaluator(annualization=cfg.annualization)
    recorder = Recorder(cfg.output_dir)

    raw = handler.load()
    data, features, label_col = feat.transform(raw)
    splits = split_by_time(data, cfg.train_end, cfg.valid_end, cfg.test_end)

    if len(splits["train"]) < 50:
        raise ValueError("Training samples too few. Please widen train period.")

    X_train = splits["train"][features].to_numpy(dtype=float)
    y_train = splits["train"][label_col].to_numpy(dtype=float)
    model.fit(X_train, y_train)

    for k in splits:
        if splits[k].empty:
            continue
        X = splits[k][features].to_numpy(dtype=float)
        splits[k] = splits[k].copy()
        splits[k]["pred"] = model.predict(X)

    threshold = tune_threshold(
        splits["valid"],
        backtester=backtester,
        evaluator=evaluator,
        label_col=label_col,
        grid=cfg.threshold_grid,
    )

    metrics: Dict[str, Dict[str, float]] = {}
    merged_bt = []

    for split_name in ("train", "valid", "test"):
        d = splits[split_name]
        if d.empty:
            metrics[split_name] = {}
            continue

        bt = backtester.run(d, pred_col="pred", label_col=label_col, threshold=threshold)
        merged_bt.append(bt.assign(split=split_name))

        m = evaluator.summarize(bt["strategy_ret"], bt["strategy_nav"])
        m.update(evaluator.info_coef(bt["pred"], bt[label_col]))
        m["threshold"] = threshold
        metrics[split_name] = m

    all_bt = pd.concat(merged_bt, axis=0).sort_index() if merged_bt else pd.DataFrame()

    if not all_bt.empty:
        recorder.save_table(
            all_bt[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "pred",
                    label_col,
                    "position",
                    "cost",
                    "strategy_ret",
                    "bench_ret",
                    "strategy_nav",
                    "bench_nav",
                    "split",
                ]
            ],
            "backtest_detail.csv",
        )

    recorder.save_json(
        {
            "config": cfg.__dict__,
            "feature_count": len(features),
            "features": features,
            "metrics": metrics,
        },
        "metrics.json",
    )

    summary_rows = []
    for split_name, vals in metrics.items():
        if not vals:
            continue
        row = {"split": split_name}
        row.update(vals)
        summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        recorder.save_table(summary_df, "metrics_summary.csv")

    return metrics


def main() -> None:
    args = build_parser().parse_args()
    cfg = FrameworkConfig(
        csv_path=args.csv,
        output_dir=args.output_dir,
        horizon=args.horizon,
        train_end=args.train_end,
        valid_end=args.valid_end,
        test_end=args.test_end,
        transaction_cost_bps=args.cost_bps,
        ridge_alpha=args.ridge_alpha,
    )

    metrics = run_pipeline(cfg)

    print("=" * 72)
    print("Qlib-like research pipeline completed")
    print(f"Data: {cfg.csv_path}")
    print(f"Output: {os.path.abspath(cfg.output_dir)}")
    print("=" * 72)
    for split, m in metrics.items():
        if not m:
            print(f"[{split}] empty")
            continue
        print(
            f"[{split}] sharpe={m['sharpe']:.3f}, ann_ret={m['annual_return']:.2%}, "
            f"mdd={m['max_drawdown']:.2%}, ic={m['ic_pearson']:.3f}, th={m['threshold']:.4f}"
        )


if __name__ == "__main__":
    main()
