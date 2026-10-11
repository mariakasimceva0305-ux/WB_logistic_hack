"""Leakage-aware temporal training. --demo uses synthetic data, never competition data."""

from pathlib import Path
import argparse, json, sys, copy
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference_src"))
from common import GRULite, metric_parts, choose_global_k

LOOKBACK = 24
HORIZON = 10
EMBARGO = pd.Timedelta(hours=2)


def synthetic():
    rng = np.random.default_rng(42)
    parts = []
    ts = pd.date_range("2025-01-01", periods=18 * 48, freq="30min")
    for route in range(8):
        rate = np.maximum(
            1,
            10
            + route * 3
            + 6 * np.sin(2 * np.pi * (ts.hour + ts.minute / 60) / 24)
            + 2 * (ts.dayofweek < 5),
        )
        raw = rng.poisson(rate)
        parts.append(
            pd.DataFrame(
                {
                    "route_id": route,
                    "office_from_id": route // 2,
                    "timestamp": ts,
                    "target_2h": pd.Series(raw)
                    .rolling(4, min_periods=4)
                    .sum()
                    .to_numpy(),
                }
            )
        )
    return pd.concat(parts).dropna().reset_index(drop=True)


def preprocessing(df, cut):
    fit = df[df.timestamp < cut]
    cap = float(fit.target_2h.quantile(0.995))
    v = fit.target_2h.clip(0, cap)
    return {"cap": cap, "mean": float(v.mean()), "std": max(float(v.std()), 1e-6)}


def partition(first, last, cuts):
    a, b, c = cuts
    if last < a:
        return "train"
    if first >= a + EMBARGO and last < b:
        return "validation"
    if first >= b + EMBARGO and last < c:
        return "calibration"
    if first >= c + EMBARGO:
        return "test"
    return None


def prepare(df):
    d = df.copy()
    d.timestamp = pd.to_datetime(d.timestamp)
    if d.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Duplicate route/timestamp")
    if not np.isfinite(d.target_2h).all() or (d.target_2h < 0).any():
        raise ValueError("Invalid target")
    end = d.timestamp.max().normalize()
    cuts = [end - pd.Timedelta(days=k) for k in [5, 3, 1]]
    prep = preprocessing(d, cuts[0])
    fit = d[d.timestamp < cuts[0]]
    routes = {int(r): int(i) for i, r in enumerate(sorted(fit.route_id.unique()))}
    offices = {
        int(r): int(i) for i, r in enumerate(sorted(fit.office_from_id.unique()))
    }
    records = {k: [] for k in ["train", "validation", "calibration", "test"]}
    audit = []
    for route, g in d.groupby("route_id"):
        if route not in routes:
            continue
        g = g.sort_values("timestamp").reset_index(drop=True)
        hour = g.timestamp.dt.hour + g.timestamp.dt.minute / 60
        dow = g.timestamp.dt.dayofweek
        target = g.target_2h.to_numpy(np.float32)
        norm = (np.clip(target, 0, prep["cap"]) - prep["mean"]) / prep["std"]
        X = np.column_stack(
            [
                norm,
                np.sin(hour * 2 * np.pi / 24),
                np.cos(hour * 2 * np.pi / 24),
                np.sin(dow * 2 * np.pi / 7),
                np.cos(dow * 2 * np.pi / 7),
            ]
        ).astype(np.float32)
        previous_end = {}
        for start in range(LOOKBACK, len(g) - HORIZON + 1):
            first = g.timestamp.iloc[start]
            last = g.timestamp.iloc[start + HORIZON - 1]
            part = partition(first, last, cuts)
            if part is None:
                continue
            if not (
                g.timestamp.iloc[start - LOOKBACK : start + HORIZON].diff().dropna()
                == pd.Timedelta(minutes=30)
            ).all():
                continue
            if part != "train" and first <= previous_end.get(part, pd.Timestamp.min):
                continue
            previous_end[part] = last
            office = offices.get(g.office_from_id.iloc[start - 1], len(offices))
            records[part].append(
                (
                    X[start - LOOKBACK : start],
                    norm[start : start + HORIZON],
                    target[start : start + HORIZON],
                    routes[route],
                    office,
                    np.full(HORIZON, target[start - 1], np.float32),
                    np.full(
                        HORIZON, target[start - LOOKBACK : start].mean(), np.float32
                    ),
                )
            )
            audit.append(
                {
                    "route": int(route),
                    "split": part,
                    "first_target": str(first),
                    "last_target": str(last),
                    "last_input": str(g.timestamp.iloc[start - 1]),
                }
            )
    for part, rs in records.items():
        if not rs:
            raise ValueError("Insufficient windows for " + part)
    return records, prep, routes, offices, audit, cuts


def tensors(rows):
    cols = list(zip(*rows))
    return TensorDataset(
        torch.tensor(np.stack(cols[0])),
        torch.tensor(np.stack(cols[1])),
        torch.tensor(np.stack(cols[2])),
        torch.tensor(cols[3], dtype=torch.long),
        torch.tensor(cols[4], dtype=torch.long),
        torch.tensor(np.stack(cols[5])),
        torch.tensor(np.stack(cols[6])),
    )


def prediction(model, rows, prep):
    model.eval()
    pred = []
    with torch.no_grad():
        for x, y, raw, r, o, last, mean in DataLoader(tensors(rows), batch_size=128):
            pred.append(model(x, r, o).numpy() * prep["std"] + prep["mean"])
    return np.maximum(0, np.concatenate(pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--output", type=Path, default=ROOT / "results")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    if a.demo == bool(a.train):
        raise ValueError("Specify exactly one of --demo or --train")
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    d = synthetic() if a.demo else pd.read_parquet(a.train)
    records, prep, routes, offices, audit, cuts = prepare(d)
    pd.DataFrame(audit).to_csv(out / "window_manifest.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "synthetic_demo": a.demo,
                "cuts": [str(t) for t in cuts],
                "trailing_target_support_hours": 2,
                "embargo_hours": 2,
                "lookback": 24,
                "horizon": 10,
                "preprocessing": prep,
                "windows": {k: len(v) for k, v in records.items()},
                "selection": "validation chooses checkpoint, calibration fits scale and stack, final test only evaluates",
            },
            indent=2,
        )
    )
    m = GRULite(
        seq_input_dim=5,
        n_routes=len(routes),
        n_offices=len(offices),
        hidden_size=32,
        pred_len=10,
    )
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    loader = DataLoader(tensors(records["train"]), batch_size=128, shuffle=True)
    best = float("inf")
    history = []
    for epoch in range(a.epochs):
        m.train()
        for x, y, raw, r, o, last, mean in loader:
            opt.zero_grad()
            loss = nn.functional.l1_loss(m(x, r, o), y)
            loss.backward()
            opt.step()
        vp = prediction(m, records["validation"], prep)
        vy = np.stack([r[2] for r in records["validation"]])
        mae = float(abs(vp - vy).mean())
        history.append({"epoch": epoch + 1, "validation_mae": mae})
        if mae < best:
            best = mae
            weights = copy.deepcopy(m.state_dict())
    m.load_state_dict(weights)
    cp = prediction(m, records["calibration"], prep)
    cy = np.stack([r[2] for r in records["calibration"]])
    k, *_ = choose_global_k(cy.ravel(), cp.ravel())
    cx = np.column_stack(
        [
            cp.ravel(),
            np.stack([r[5] for r in records["calibration"]]).ravel(),
            np.stack([r[6] for r in records["calibration"]]).ravel(),
        ]
    )
    stack = Ridge(alpha=1.0, positive=True).fit(cx, cy.ravel())
    tp = prediction(m, records["test"], prep)
    ty = np.stack([r[2] for r in records["test"]])
    last = np.stack([r[5] for r in records["test"]])
    mean = np.stack([r[6] for r in records["test"]])
    sx = np.column_stack([tp.ravel(), last.ravel(), mean.ravel()])
    sp = np.maximum(0, stack.predict(sx)).reshape(tp.shape)
    result = []
    for name, p in [
        ("persistence", last),
        ("rolling_mean", mean),
        ("gru", tp),
        ("gru_calibrated", tp * k),
        ("ridge_stack", sp),
    ]:
        score, wape, bias = metric_parts(ty.ravel(), p.ravel())
        result.append(
            {
                "model": name,
                "score": score,
                "wape": wape,
                "absolute_relative_bias": bias,
                "synthetic_demo": a.demo,
            }
        )
    pd.DataFrame(result).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out / "learning_curve.csv", index=False)
    np.savez(
        out / "test_predictions.npz",
        target=ty,
        gru=tp,
        calibrated=tp * k,
        stack=sp,
        persistence=last,
        rolling_mean=mean,
    )
    torch.save(
        {
            "state_dict": m.state_dict(),
            "preprocessing": prep,
            "route2idx": routes,
            "office2idx": offices,
            "calibration_k": k,
            "stack_coef": stack.coef_.tolist(),
            "stack_intercept": float(stack.intercept_),
        },
        out / "model.pt",
    )
    print(
        json.dumps(
            {
                "synthetic_demo": a.demo,
                "windows": {k: len(v) for k, v in records.items()},
                "metrics": result,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
