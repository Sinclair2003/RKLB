"""
Enhanced next-day RKLB forecasting.

Changes vs old version:
1) Target changed to next-day return (not direct next close level)
2) Model changed to ensemble:
   - linear ridge on standardized features
   - non-linear ridge on random Fourier features (RFF)
3) Auto-fetch additional market/context data via yfinance and join as features

Run:
  .venv/bin/python predict_next_close_rklb.py
  .venv/bin/python predict_next_close_rklb.py --refresh-context
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Config:
    csv_path: str = "market_data/rklb_daily.csv"
    output_dir: str = "outputs/rklb_price_forecast_v2"

    train_end: str = "2023-12-31"
    valid_end: str = "2024-12-31"
    test_end: str = "2099-12-31"

    # External market/context symbols used as features.
    context_symbols: Tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
        "^VIX",
        "^TNX",
        "DX-Y.NYB",
        "CL=F",
        "GC=F",
    )

    # Model hyper-parameter grids.
    alpha_grid: Tuple[float, ...] = (0.1, 1.0, 5.0, 20.0, 100.0)
    gamma_grid: Tuple[float, ...] = (0.1, 0.3, 0.7, 1.0)
    blend_grid: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    # RFF config
    rff_dim: int = 256
    rff_seed: int = 42

    min_train_rows: int = 120
    wf_train_days: int = 504
    wf_valid_days: int = 126
    wf_test_days: int = 21


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enhanced RKLB next-day close forecasting")
    p.add_argument("--csv", default="market_data/rklb_daily.csv")
    p.add_argument("--output-dir", default="outputs/rklb_price_forecast_v2")
    p.add_argument("--train-end", default="2023-12-31")
    p.add_argument("--valid-end", default="2024-12-31")
    p.add_argument("--test-end", default="2099-12-31")
    p.add_argument("--refresh-context", action="store_true", help="Force re-download context data")
    p.add_argument("--wf-train-days", type=int, default=504)
    p.add_argument("--wf-valid-days", type=int, default=126)
    p.add_argument("--wf-test-days", type=int, default=21)
    return p


def load_rklb(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    req = {"date", "open", "high", "low", "close", "volume"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="first").set_index("date")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["close"])


def sanitize_symbol(sym: str) -> str:
    return sym.replace("^", "").replace("=", "_").replace("-", "_").replace("/", "_").lower()


def fetch_context_data(base_index: pd.DatetimeIndex, symbols: Tuple[str, ...], cache_dir: str, refresh: bool) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)

    start = (base_index.min() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end = (base_index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    ctx = pd.DataFrame(index=base_index)

    for sym in symbols:
        cache_path = os.path.join(cache_dir, f"{sanitize_symbol(sym)}_daily.csv")

        if (not refresh) and os.path.exists(cache_path):
            s = pd.read_csv(cache_path)
            s["date"] = pd.to_datetime(s["date"])
            s = s.sort_values("date").drop_duplicates("date", keep="first").set_index("date")
            if "close" not in s.columns:
                continue
            close_series = pd.to_numeric(s["close"], errors="coerce")
        else:
            df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                continue
            close_series = pd.to_numeric(df["Close"], errors="coerce")
            to_save = pd.DataFrame({"date": close_series.index, "close": close_series.values})
            to_save.to_csv(cache_path, index=False)

        close_series = close_series.sort_index().ffill()
        aligned = close_series.reindex(base_index).ffill()
        ctx[f"ctx_close_{sanitize_symbol(sym)}"] = aligned

    return ctx


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_2"] = out["close"].pct_change(2)
    out["ret_5"] = out["close"].pct_change(5)
    out["ret_10"] = out["close"].pct_change(10)

    out["hl_spread"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_spread"] = (out["open"] - out["close"]) / out["close"].replace(0, np.nan)

    for w in (3, 5, 10, 20, 60):
        ma = out["close"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["close"] / ma - 1

    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    lv = np.log1p(out["volume"].clip(lower=0))
    out["vol_z_20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    for l in (1, 2, 3, 5, 10):
        out[f"close_lag_{l}"] = out["close"].shift(l)

    return out


def add_context_features(df: pd.DataFrame, context_raw: pd.DataFrame) -> pd.DataFrame:
    out = df.join(context_raw, how="left")

    ctx_cols = [c for c in out.columns if c.startswith("ctx_close_")]
    for c in ctx_cols:
        out[f"{c}_ret_1"] = out[c].pct_change(1)
        out[f"{c}_ret_5"] = out[c].pct_change(5)
        out[f"{c}_ret_10"] = out[c].pct_change(10)
        ma20 = out[c].rolling(20).mean()
        out[f"{c}_ma_ratio_20"] = out[c] / ma20 - 1

    return out


def make_dataset(raw: pd.DataFrame, context_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    out = add_base_features(raw)
    out = add_context_features(out, context_raw)

    # Target: next-day return.
    out["y_next_ret"] = out["close"].shift(-1) / out["close"] - 1.0

    feature_cols = [
        c
        for c in out.columns
        if c not in {"open", "high", "low", "close", "volume", "y_next_ret"}
    ]

    out = out.replace([np.inf, -np.inf], np.nan)
    model_df = out.dropna(subset=feature_cols + ["y_next_ret"])  # training/eval rows
    predict_df = out.dropna(subset=feature_cols)  # includes latest row for forecast
    return model_df, predict_df, feature_cols


def split_data(df: pd.DataFrame, train_end: str, valid_end: str, test_end: str) -> Dict[str, pd.DataFrame]:
    tr = pd.to_datetime(train_end)
    va = pd.to_datetime(valid_end)
    te = pd.to_datetime(test_end)

    return {
        "train": df[df.index <= tr],
        "valid": df[(df.index > tr) & (df.index <= va)],
        "test": df[(df.index > va) & (df.index <= te)],
    }


class StandardizedRidge:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma = np.where(self.sigma < 1e-12, 1.0, self.sigma)

        Z = (X - self.mu) / self.sigma
        Z = np.hstack([np.ones((Z.shape[0], 1)), Z])

        reg = np.eye(Z.shape[1]) * self.alpha
        reg[0, 0] = 0.0
        self.w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None or self.w is None:
            raise RuntimeError("Model not fitted")
        Z = (X - self.mu) / self.sigma
        Z = np.hstack([np.ones((Z.shape[0], 1)), Z])
        return Z @ self.w


class EnsembleReturnModel:
    def __init__(self, alpha: float, gamma: float, rff_dim: int, seed: int):
        self.linear = StandardizedRidge(alpha=alpha)
        self.rff = StandardizedRidge(alpha=alpha)
        self.gamma = float(gamma)
        self.rff_dim = int(rff_dim)
        self.seed = int(seed)
        self.omega: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def _fit_rff_map(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n_features = X.shape[1]
        self.omega = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(n_features, self.rff_dim))
        self.bias = rng.uniform(0.0, 2.0 * np.pi, size=(self.rff_dim,))

    def _rff_transform(self, X: np.ndarray) -> np.ndarray:
        if self.omega is None or self.bias is None:
            raise RuntimeError("RFF map not initialized")
        proj = X @ self.omega + self.bias
        return np.sqrt(2.0 / self.rff_dim) * np.cos(proj)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.linear.fit(X, y)
        self._fit_rff_map(X)
        X_rff = self._rff_transform(X)
        self.rff.fit(X_rff, y)

    def predict(self, X: np.ndarray, blend: float) -> np.ndarray:
        p1 = self.linear.predict(X)
        p2 = self.rff.predict(self._rff_transform(X))
        return blend * p2 + (1.0 - blend) * p1


def return_metrics(y_true_ret: np.ndarray, y_pred_ret: np.ndarray, prev_close: np.ndarray) -> Dict[str, float]:
    pred_close = prev_close * (1.0 + y_pred_ret)
    true_close = prev_close * (1.0 + y_true_ret)
    naive_close = prev_close

    err = pred_close - true_close
    naive_err = naive_close - true_close

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(true_close), 1e-12)))

    naive_mae = float(np.mean(np.abs(naive_err)))
    naive_rmse = float(np.sqrt(np.mean(naive_err**2)))
    naive_mape = float(np.mean(np.abs(naive_err) / np.maximum(np.abs(true_close), 1e-12)))

    model_dir = np.sign(y_pred_ret)
    true_dir = np.sign(y_true_ret)
    naive_dir = np.zeros_like(true_dir)

    return {
        "model_mae": mae,
        "model_rmse": rmse,
        "model_mape": mape,
        "model_directional_acc": float(np.mean(model_dir == true_dir)),
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_mape": naive_mape,
        "naive_directional_acc": float(np.mean(naive_dir == true_dir)),
        "mae_improvement_vs_naive": naive_mae - mae,
        "rmse_improvement_vs_naive": naive_rmse - rmse,
    }


def next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts + pd.offsets.BDay(1))


def tune_params(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
) -> Dict[str, float]:
    X_train = train_df[feature_cols].to_numpy(float)
    y_train = train_df["y_next_ret"].to_numpy(float)

    best = {
        "alpha": None,
        "gamma": None,
        "blend": None,
        "valid_mae": float("inf"),
    }

    for alpha in cfg.alpha_grid:
        for gamma in cfg.gamma_grid:
            model = EnsembleReturnModel(alpha=alpha, gamma=gamma, rff_dim=cfg.rff_dim, seed=cfg.rff_seed)
            model.fit(X_train, y_train)

            if valid_df.empty:
                best = {"alpha": alpha, "gamma": gamma, "blend": 0.5, "valid_mae": 0.0}
                break

            X_valid = valid_df[feature_cols].to_numpy(float)
            y_valid = valid_df["y_next_ret"].to_numpy(float)
            prev_valid = valid_df["close"].to_numpy(float)

            for blend in cfg.blend_grid:
                p_valid = model.predict(X_valid, blend=blend)
                m = return_metrics(y_valid, p_valid, prev_valid)
                if m["model_mae"] < best["valid_mae"]:
                    best = {"alpha": alpha, "gamma": gamma, "blend": blend, "valid_mae": m["model_mae"]}

    if best["alpha"] is None:
        best["alpha"] = cfg.alpha_grid[0]
        best["gamma"] = cfg.gamma_grid[0]
        best["blend"] = 0.5
        best["valid_mae"] = float("nan")
    return best


def run_walk_forward(
    model_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict[str, float]]]:
    data = model_df.sort_index()
    n = len(data)
    start_idx = cfg.wf_train_days + cfg.wf_valid_days

    if n < start_idx + 5:
        return {}, pd.DataFrame(), []

    all_preds = []
    fold_summaries = []
    fold_id = 0

    for i in range(start_idx, n, cfg.wf_test_days):
        tr_s = i - cfg.wf_valid_days - cfg.wf_train_days
        tr_e = i - cfg.wf_valid_days
        va_s = tr_e
        va_e = i
        te_s = i
        te_e = min(i + cfg.wf_test_days, n)

        train_df = data.iloc[tr_s:tr_e]
        valid_df = data.iloc[va_s:va_e]
        test_df = data.iloc[te_s:te_e]

        if len(train_df) < cfg.min_train_rows or test_df.empty:
            continue

        best = tune_params(train_df, valid_df, feature_cols, cfg)

        train_valid_df = pd.concat([train_df, valid_df], axis=0)
        model = EnsembleReturnModel(
            alpha=float(best["alpha"]),
            gamma=float(best["gamma"]),
            rff_dim=cfg.rff_dim,
            seed=cfg.rff_seed,
        )
        model.fit(
            train_valid_df[feature_cols].to_numpy(float),
            train_valid_df["y_next_ret"].to_numpy(float),
        )

        X_test = test_df[feature_cols].to_numpy(float)
        y_test = test_df["y_next_ret"].to_numpy(float)
        prev_test = test_df["close"].to_numpy(float)
        pred_test = model.predict(X_test, blend=float(best["blend"]))
        m = return_metrics(y_test, pred_test, prev_test)

        fold_summaries.append(
            {
                "fold_id": fold_id,
                "test_start": str(test_df.index[0].date()),
                "test_end": str(test_df.index[-1].date()),
                "rows": int(len(test_df)),
                "alpha": float(best["alpha"]),
                "gamma": float(best["gamma"]),
                "blend": float(best["blend"]),
                "model_mae": m["model_mae"],
                "naive_mae": m["naive_mae"],
                "model_rmse": m["model_rmse"],
                "naive_rmse": m["naive_rmse"],
                "model_directional_acc": m["model_directional_acc"],
            }
        )

        out = test_df[["open", "high", "low", "close", "volume", "y_next_ret"]].copy()
        out["pred_next_ret"] = pred_test
        out["pred_next_close"] = out["close"] * (1.0 + out["pred_next_ret"])
        out["true_next_close"] = out["close"] * (1.0 + out["y_next_ret"])
        out["abs_error"] = (out["pred_next_close"] - out["true_next_close"]).abs()
        out["fold_id"] = fold_id
        all_preds.append(out)
        fold_id += 1

    if not all_preds:
        return {}, pd.DataFrame(), []

    wf_df = pd.concat(all_preds, axis=0).sort_index()
    wf_metrics = return_metrics(
        wf_df["y_next_ret"].to_numpy(float),
        wf_df["pred_next_ret"].to_numpy(float),
        wf_df["close"].to_numpy(float),
    )
    return wf_metrics, wf_df, fold_summaries


def run(cfg: Config, refresh_context: bool) -> Dict:
    os.makedirs(cfg.output_dir, exist_ok=True)

    raw = load_rklb(cfg.csv_path)
    context = fetch_context_data(
        base_index=raw.index,
        symbols=cfg.context_symbols,
        cache_dir=os.path.join("market_data", "context_cache"),
        refresh=refresh_context,
    )

    model_df, predict_df, feature_cols = make_dataset(raw, context)
    splits = split_data(model_df, cfg.train_end, cfg.valid_end, cfg.test_end)

    if len(splits["train"]) < cfg.min_train_rows:
        raise ValueError("Training samples too few. Enlarge train period.")

    best = tune_params(splits["train"], splits["valid"], feature_cols, cfg)

    train_valid = pd.concat([splits["train"], splits["valid"]], axis=0).sort_index()
    final_model = EnsembleReturnModel(
        alpha=float(best["alpha"]),
        gamma=float(best["gamma"]),
        rff_dim=cfg.rff_dim,
        seed=cfg.rff_seed,
    )
    final_model.fit(
        train_valid[feature_cols].to_numpy(float),
        train_valid["y_next_ret"].to_numpy(float),
    )

    split_metrics: Dict[str, Dict[str, float]] = {}
    details = []

    for name in ("train", "valid", "test"):
        d = splits[name]
        if d.empty:
            split_metrics[name] = {}
            continue

        X = d[feature_cols].to_numpy(float)
        y = d["y_next_ret"].to_numpy(float)
        prev = d["close"].to_numpy(float)

        pred_ret = final_model.predict(X, blend=float(best["blend"]))
        m = return_metrics(y, pred_ret, prev)
        split_metrics[name] = m

        out = d[["open", "high", "low", "close", "volume", "y_next_ret"]].copy()
        out["pred_next_ret"] = pred_ret
        out["pred_next_close"] = out["close"] * (1.0 + out["pred_next_ret"])
        out["true_next_close"] = out["close"] * (1.0 + out["y_next_ret"])
        out["abs_error"] = (out["pred_next_close"] - out["true_next_close"]).abs()
        out["split"] = name
        details.append(out)

    detail_df = pd.concat(details, axis=0).sort_index() if details else pd.DataFrame()

    latest_x = predict_df[feature_cols].iloc[[-1]].to_numpy(float)
    pred_next_ret = float(final_model.predict(latest_x, blend=float(best["blend"]))[0])
    last_close = float(predict_df["close"].iloc[-1])
    pred_next_close = float(last_close * (1.0 + pred_next_ret))

    last_date = pd.Timestamp(predict_df.index[-1])
    pred_date = next_business_day(last_date)

    if not splits["test"].empty:
        X_test = splits["test"][feature_cols].to_numpy(float)
        y_test = splits["test"]["y_next_ret"].to_numpy(float)
        test_pred = final_model.predict(X_test, blend=float(best["blend"]))
        resid_std = float(np.std(y_test - test_pred))
    else:
        resid_std = 0.02

    low = last_close * (1.0 + pred_next_ret - resid_std)
    high = last_close * (1.0 + pred_next_ret + resid_std)

    result = {
        "config": {
            "csv_path": cfg.csv_path,
            "train_end": cfg.train_end,
            "valid_end": cfg.valid_end,
            "test_end": cfg.test_end,
            "context_symbols": list(cfg.context_symbols),
            "alpha_grid": list(cfg.alpha_grid),
            "gamma_grid": list(cfg.gamma_grid),
            "blend_grid": list(cfg.blend_grid),
            "best_alpha": best["alpha"],
            "best_gamma": best["gamma"],
            "best_blend": best["blend"],
            "feature_count": len(feature_cols),
        },
        "samples": {k: int(len(v)) for k, v in splits.items()},
        "metrics": split_metrics,
        "forecast": {
            "last_observed_date": str(last_date.date()),
            "approx_next_trading_date": str(pred_date.date()),
            "last_close": last_close,
            "pred_next_return": pred_next_ret,
            "pred_next_close": pred_next_close,
            "approx_68pct_interval_close": [float(low), float(high)],
        },
    }

    detail_df.to_csv(os.path.join(cfg.output_dir, "predictions_detail.csv"))

    rows = []
    for s, vals in split_metrics.items():
        if vals:
            row = {"split": s}
            row.update(vals)
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(cfg.output_dir, "metrics_summary.csv"), index=False)

    with open(os.path.join(cfg.output_dir, "forecast.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    wf_metrics, wf_df, wf_folds = run_walk_forward(model_df, feature_cols, cfg)
    result["walk_forward"] = {
        "enabled": True,
        "train_days": cfg.wf_train_days,
        "valid_days": cfg.wf_valid_days,
        "test_days": cfg.wf_test_days,
        "metrics": wf_metrics,
        "num_folds": len(wf_folds),
    }
    with open(os.path.join(cfg.output_dir, "walk_forward_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": result["walk_forward"],
                "folds": wf_folds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if not wf_df.empty:
        wf_df.to_csv(os.path.join(cfg.output_dir, "walk_forward_detail.csv"))

    return result


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(
        csv_path=args.csv,
        output_dir=args.output_dir,
        train_end=args.train_end,
        valid_end=args.valid_end,
        test_end=args.test_end,
        wf_train_days=args.wf_train_days,
        wf_valid_days=args.wf_valid_days,
        wf_test_days=args.wf_test_days,
    )

    res = run(cfg, refresh_context=args.refresh_context)

    print("=" * 76)
    print("Enhanced next-day forecasting completed")
    print(f"Data: {cfg.csv_path}")
    print(f"Output: {os.path.abspath(cfg.output_dir)}")
    print(
        f"Best params: alpha={res['config']['best_alpha']}, "
        f"gamma={res['config']['best_gamma']}, blend={res['config']['best_blend']}"
    )
    print(f"Features used: {res['config']['feature_count']}")
    print("=" * 76)

    for s in ("train", "valid", "test"):
        m = res["metrics"].get(s, {})
        if not m:
            print(f"[{s}] empty")
            continue
        print(
            f"[{s}] MAE={m['model_mae']:.4f} (naive {m['naive_mae']:.4f}), "
            f"RMSE={m['model_rmse']:.4f} (naive {m['naive_rmse']:.4f}), "
            f"DirAcc={m['model_directional_acc']:.2%}"
        )

    fc = res["forecast"]
    lo, hi = fc["approx_68pct_interval_close"]
    print("-" * 76)
    print(f"Last close ({fc['last_observed_date']}): {fc['last_close']:.4f}")
    print(f"Pred next close ({fc['approx_next_trading_date']}): {fc['pred_next_close']:.4f}")
    print(f"Pred next return: {fc['pred_next_return']:.2%}")
    print(f"Approx 68% close interval: [{lo:.4f}, {hi:.4f}]")
    wf = res.get("walk_forward", {})
    wm = wf.get("metrics", {})
    if wm:
        print("-" * 76)
        print(
            f"Walk-forward ({wf.get('num_folds', 0)} folds): "
            f"MAE={wm['model_mae']:.4f} (naive {wm['naive_mae']:.4f}), "
            f"RMSE={wm['model_rmse']:.4f} (naive {wm['naive_rmse']:.4f}), "
            f"DirAcc={wm['model_directional_acc']:.2%}"
        )


if __name__ == "__main__":
    main()
