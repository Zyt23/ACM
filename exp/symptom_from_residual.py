# -*- coding: utf-8 -*-
"""
From residual -> symptom (rule-based, interpretable).
Use together with classifier outputs.

Features:
- mse, mae
- lag between valve-change and residual/temperature-change (cross-corr)
- drift slope
- low-freq oscillation score
- energy ratio proxy (std(discharge)/std(compr), corr(discharge,compr))
"""
from __future__ import annotations
import numpy as np


def _zscore(x: np.ndarray, eps: float = 1e-6):
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    return (x - m) / (s + eps)


def _crosscorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 12) -> int:
    """
    return lag that maximizes abs correlation corr(a(t), b(t+lag))
    lag>0 means b is delayed vs a.
    """
    a = _zscore(a)
    b = _zscore(b)
    best_lag, best = 0, -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = a[:len(a) - lag]
            y = b[lag:]
        else:
            k = -lag
            x = a[k:]
            y = b[:len(b) - k]
        if len(x) < 8:
            continue
        c = float(np.nanmean(x * y))
        if abs(c) > best:
            best = abs(c)
            best_lag = lag
    return int(best_lag)


def _drift_slope(x: np.ndarray) -> float:
    L = len(x)
    t = np.linspace(0.0, 1.0, L, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # simple linear regression slope
    cov = float(np.mean((t - t.mean()) * (x - x.mean())))
    var = float(np.mean((t - t.mean()) ** 2) + 1e-6)
    return cov / var


def _lowfreq_osc_score(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    L = len(x)
    if L < 16:
        return 0.0
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    # exclude DC, focus on very low frequencies: 1~3 bins
    lo = mag[1:4].sum()
    allp = mag[1:].sum() + 1e-6
    return float(lo / allp)


def extract_symptom_features(win_6: np.ndarray, y_pred: np.ndarray):
    """
    win_6: [L,6] (BYPASS,DISCH,RAM_I,RAM_O,FLOW,COMPR)
    y_pred: [L] or [L,1] predicted COMPR
    """
    w = np.asarray(win_6, dtype=np.float32)
    L = w.shape[0]
    bypass = w[:, 0]
    disch = w[:, 1]
    rami = w[:, 2]
    ramo = w[:, 3]
    flow = w[:, 4]
    compr = w[:, 5]

    yp = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    yt = compr.reshape(-1)
    resid = yt - yp

    mse = float(np.mean(resid ** 2))
    mae = float(np.mean(np.abs(resid)))

    # lag between control change and residual change proxy
    dbypass = np.diff(bypass, prepend=bypass[:1])
    dcompr = np.diff(compr, prepend=compr[:1])
    dresid = np.diff(resid, prepend=resid[:1])

    lag_bypass_resid = _crosscorr_lag(dbypass, dresid, max_lag=12)
    lag_rami_resid = _crosscorr_lag(np.diff(rami, prepend=rami[:1]), dresid, max_lag=12)

    drift = _drift_slope(resid)
    osc = _lowfreq_osc_score(resid)

    # energy proxy
    std_ratio = float(np.std(disch) / (np.std(compr) + 1e-6))
    corr_dc = float(np.corrcoef(disch, compr)[0, 1]) if np.std(disch) > 1e-6 and np.std(compr) > 1e-6 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "lag_bypass_resid": lag_bypass_resid,
        "lag_rami_resid": lag_rami_resid,
        "drift": drift,
        "osc": osc,
        "std_ratio_disch_over_compr": std_ratio,
        "corr_disch_compr": corr_dc,
    }


def rule_symptom(features: dict, thr: dict | None = None) -> str:
    """
    Simple interpretable rules.
    thr can be loaded from a baseline stats json (optional).
    """
    thr = thr or {}
    # defaults (tune on your normal validation)
    mse_thr = float(thr.get("mse_thr", 0.5))
    lag_thr = int(thr.get("lag_thr", 6))
    drift_thr = float(thr.get("drift_thr", 0.25))
    osc_thr = float(thr.get("osc_thr", 0.25))
    ratio_lo = float(thr.get("ratio_lo", 0.6))
    ratio_hi = float(thr.get("ratio_hi", 1.6))

    if features["mse"] < mse_thr:
        # even if mse small, we can still call out pattern anomalies
        if abs(features["lag_bypass_resid"]) >= lag_thr or abs(features["lag_rami_resid"]) >= lag_thr:
            return "响应错位/迟滞异常(轻)"
        if abs(features["drift"]) >= drift_thr:
            return "缓慢漂移退化(轻)"
        if features["osc"] >= osc_thr:
            return "低频振荡(轻)"
        if (features["std_ratio_disch_over_compr"] < ratio_lo) or (features["std_ratio_disch_over_compr"] > ratio_hi):
            return "能量比例异常(轻)"
        return "正常"

    # mse高：再细分
    if (features["std_ratio_disch_over_compr"] < ratio_lo) or (features["std_ratio_disch_over_compr"] > ratio_hi):
        return "能量失衡/换热效率异常"
    if abs(features["lag_bypass_resid"]) >= lag_thr or abs(features["lag_rami_resid"]) >= lag_thr:
        return "控制-温度响应错位/迟滞异常"
    if abs(features["drift"]) >= drift_thr:
        return "缓慢退化趋势(效率下降)"
    if features["osc"] >= osc_thr:
        return "振荡/不稳定(机械或控制环)"
    return "综合异常(需进一步定位)"
