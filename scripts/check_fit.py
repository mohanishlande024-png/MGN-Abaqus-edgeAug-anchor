r"""
Over- or under-fitting check for Run B1 (or any edge-augmented MGN).

    python scripts/check_fit.py --ckpt results_aug20_anchors_redraw/mgn_best.pt --draws 8

Run eval_own_unseen.py with the same --draws first: this script reads its
metrics_all.csv for the test sets and adds the TRAINING set itself.

WHAT IT DOES
    1. Loss curve (loss_log.csv): has training finished improving?
       The loss is the MSE of z-scored stress, so  1 - loss  is roughly the
       R2 of the training fit (all training nodes pooled).
    2. Training-set fit: predicts every training case (27 geometries x 20
       loads) and computes R2 / peak error exactly like eval_own_unseen.py.
    3. The "fit ladder": the same numbers on
           train            seen geometry, seen load
           unseen_loads     seen geometry, new load
           unseen_interp    new geometry inside the training range
           unseen_extrap    new geometry outside the training range
       and a verdict:
           underfitting  -> the TRAIN level itself is poor
           overfitting   -> train is excellent but unseen_interp is much worse
           extrapolation -> train and interp are good, only extrap is poor
                            (a data-coverage problem, not overfitting)

WRITES  (into <checkpoint folder>/fit_check/)
    fit_train_metrics.csv   every training case: R2, RMSE, peak error
    fit_ladder.csv          the summary table
    fit_check.png           loss curve + R2 and peak error at every level
"""

import argparse
import csv
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import eval_own_unseen as E                                   # noqa: E402  (load_model, metrics, plotting style)
from mgn.dataset import Case, load_dataset                    # noqa: E402
from mgn.graph import augment_edges, geometry_seed            # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

LEVELS = [("train", "train  (seen geometry, seen load)"),
          ("unseen_loads", "unseen loads  (seen geometry)"),
          ("unseen_interp", "unseen geometry - interpolation"),
          ("unseen_extrap", "unseen geometry - EXTRAPOLATION")]
TICK = dict(train="train", unseen_loads="unseen\nloads", unseen_interp="new shape\n(interp)",
            unseen_extrap="new shape\n(EXTRAP)")

# thresholds used for the verdict (change them if you want a stricter test)
TRAIN_R2_OK = 0.99          # mean R2 on the training set below this -> underfitting
TRAIN_PEAK_OK = 5.0         # mean |peak error| on the training set above this (%) -> underfitting at peaks
GAP_R2 = 0.03               # train R2 - interp R2 above this -> generalisation gap (overfitting sign)


def summarise(rows):
    r2 = np.array([r["r2"] for r in rows], dtype=float)
    pk = np.array([abs(r["peak_err_pct"]) for r in rows], dtype=float)
    nr = np.array([r["rmse"] / r["peak_true"] for r in rows], dtype=float)
    worst = rows[int(np.nanargmin(r2))]
    return dict(n=len(rows), mean_r2=float(np.nanmean(r2)), median_r2=float(np.nanmedian(r2)),
                worst_r2=float(np.nanmin(r2)), worst_case="%s @ %g" % (E.short(worst["geometry"]), worst["load"]),
                mean_abs_peak=float(np.nanmean(pk)), mean_nrmse=100 * float(np.nanmean(nr)))


def read_test_csv(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rmse_col = [k for k in r if k.startswith("rmse")][0]
            peak_col = [k for k in r if k.startswith("peak_true")][0]
            out.setdefault(r["set"], []).append(dict(
                geometry=r["geometry"], load=float(r["load"]), r2=float(r["r2"]),
                rmse=float(r[rmse_col]), peak_true=float(r[peak_col]),
                peak_err_pct=float(r["peak_err_pct"])))
    return out


def loss_report(log_path):
    if not os.path.isfile(log_path):
        print("no loss_log.csv at %s - skipping the loss curve" % log_path)
        return None
    ep, loss = [], []
    with open(log_path) as fh:
        for r in csv.DictReader(fh):
            ep.append(int(r["epoch"]))
            loss.append(float(r["loss"]))
    ep, loss = np.array(ep), np.array(loss)
    n = len(loss)
    w = max(1, min(500, n // 5))
    late, before = loss[-w:].mean(), loss[-2 * w:-w].mean() if n >= 2 * w else np.nan
    drop = 100 * (before - late) / before if np.isfinite(before) else np.nan
    print("\nLOSS CURVE  (%s)" % log_path)
    print("  epochs logged              %d" % n)
    for e in (1, 100, 1000, 2000, 5000, n):
        if e <= n:
            print("  loss at epoch %-6d       %.3e   (training fit R2 ~ %.5f)" % (e, loss[e - 1], 1 - loss[e - 1]))
    print("  lowest loss                %.3e at epoch %d" % (loss.min(), ep[int(loss.argmin())]))
    if np.isfinite(drop):
        print("  mean of last %d epochs vs the %d before: %.1f %% lower" % (w, w, drop))
        print("  -> %s" % ("STILL IMPROVING: more epochs (or a higher lr) could still help = not fully trained"
                         if drop > 10 else "levelled off: training has converged"))
    return ep, loss, drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join("results_aug20_anchors_redraw", "mgn_best.pt"))
    ap.add_argument("--train-data", default=os.path.join("data", "dataset_own_aug20.pt"))
    ap.add_argument("--draws", type=int, default=1, help="average over this many random-edge draws (use the same as for eval_own_unseen.py)")
    ap.add_argument("--test-dir", default=None, help="default: <ckpt folder>/unseen_eval[_drawsK]")
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    a = ap.parse_args()

    folder = os.path.dirname(a.ckpt) or "."
    out = os.path.join(folder, "fit_check")
    os.makedirs(out, exist_ok=True)
    test_dir = a.test_dir or os.path.join(folder, "unseen_eval" + ("_draws%d" % a.draws if a.draws > 1 else ""))

    # ---- 1. loss curve ---------------------------------------------------
    lr = loss_report(os.path.join(folder, "loss_log.csv"))

    # ---- 2. training-set fit ---------------------------------------------
    margs = types.SimpleNamespace(ckpt=a.ckpt, train_data=a.train_data, layers=a.layers,
                                  hidden=a.hidden, embedding=a.embedding)
    mgn, desc, _ = E.load_model(margs)
    print("\nmodel   %s" % desc)
    cases = load_dataset(a.train_data)
    print("train   %d cases from %s,  prediction = mean of %d random-edge draw(s)" % (len(cases), a.train_data, a.draws))
    train_rows = []
    for i, c in enumerate(cases):
        preds = []
        for k in range(max(1, a.draws)):
            if k == 0:
                case = c                                         # the stored draw = seed aug_seed + 0
            else:
                mesh = mgn._mesh_edges(c)
                ei, fl = augment_edges(mesh, len(c.coordinates), perc=mgn.aug_perc,
                                       seed=geometry_seed(c.geometry, mgn.aug_seed + k))
                case = Case(geometry=c.geometry, coordinates=c.coordinates, edge_index=torch.from_numpy(ei),
                            node_types=c.node_types, von_mises=c.von_mises, metadata=c.metadata,
                            edge_flag=torch.from_numpy(fl))
            with torch.no_grad():
                preds.append(np.asarray(mgn.predict(case), dtype=np.float64).ravel())
        pred = np.mean(preds, axis=0)
        truth = np.asarray(c.von_mises, dtype=np.float64)
        m = E.metrics(truth, pred)
        train_rows.append(dict(geometry=c.geometry, load=float(c.metadata["load"]), r2=m["r2"], rmse=m["rmse"],
                               peak_true=m["peak_true"], peak_pred=m["peak_pred"], peak_err_pct=m["peak_err_pct"]))
        if (i + 1) % 60 == 0 or i + 1 == len(cases):
            print("  %d/%d" % (i + 1, len(cases)))
    with open(os.path.join(out, "fit_train_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geometry", "load", "r2", "rmse", "peak_true", "peak_pred", "peak_err_pct"])
        for r in sorted(train_rows, key=lambda r: (r["geometry"], r["load"])):
            w.writerow([r["geometry"], "%g" % r["load"], "%.6f" % r["r2"], "%.4f" % r["rmse"],
                        "%.4f" % r["peak_true"], "%.4f" % r["peak_pred"], "%.4f" % r["peak_err_pct"]])

    # ---- 3. the fit ladder ------------------------------------------------
    sets = read_test_csv(os.path.join(test_dir, "metrics_all.csv"))
    if not sets:
        print("\nNo %s - run eval_own_unseen.py with --draws %d first for the test levels."
              % (os.path.join(test_dir, "metrics_all.csv"), a.draws))
    sets["train"] = train_rows
    summ = dict((k, summarise(sets[k])) for k, _ in LEVELS if sets.get(k))

    print("\nFIT LADDER   (test levels from %s)" % test_dir)
    print("  %-36s %6s %9s %9s %9s  %-24s %13s %12s" % ("level", "files", "mean R2", "median", "worst", "worst case",
                                                       "mean|peak err|", "RMSE/peak"))
    print("  " + "-" * 130)
    with open(os.path.join(out, "fit_ladder.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "files", "mean_r2", "median_r2", "worst_r2", "worst_case", "mean_abs_peak_err_pct", "mean_rmse_over_peak_pct"])
        for k, title in LEVELS:
            if k not in summ:
                continue
            s = summ[k]
            print("  %-36s %6d %9.4f %9.4f %9.4f  %-24s %12.2f%% %11.2f%%" % (title, s["n"], s["mean_r2"], s["median_r2"],
                  s["worst_r2"], s["worst_case"], s["mean_abs_peak"], s["mean_nrmse"]))
            w.writerow([k, s["n"], "%.6f" % s["mean_r2"], "%.6f" % s["median_r2"], "%.6f" % s["worst_r2"], s["worst_case"],
                        "%.4f" % s["mean_abs_peak"], "%.4f" % s["mean_nrmse"]])

    # ---- verdict ----------------------------------------------------------
    t = summ["train"]
    print("\nVERDICT")
    under = t["mean_r2"] < TRAIN_R2_OK or t["mean_abs_peak"] > TRAIN_PEAK_OK
    if under:
        print("  UNDERFITTING: the model does not even fit its own training data well "
              "(train mean R2 %.4f, mean |peak err| %.1f %%)." % (t["mean_r2"], t["mean_abs_peak"]))
        print("  -> more epochs / capacity / better features; more data will NOT fix this.")
    else:
        print("  training fit is good (train mean R2 %.4f, mean |peak err| %.1f %%) -> no underfitting."
              % (t["mean_r2"], t["mean_abs_peak"]))
    if "unseen_interp" in summ:
        gap = t["mean_r2"] - summ["unseen_interp"]["mean_r2"]
        pk_ratio = summ["unseen_interp"]["mean_abs_peak"] / max(t["mean_abs_peak"], 1e-9)
        if gap > GAP_R2:
            print("  OVERFITTING / MEMORISATION: R2 drops by %.3f from train to new shapes inside the range "
                  "(peak error x%.1f)." % (gap, pk_ratio))
            print("  -> more (or more varied) training geometries, regularisation, earlier stopping on a validation set.")
        else:
            print("  generalisation gap to new shapes inside the range is small (R2 drop %.3f, peak error x%.1f) "
                  "-> no meaningful overfitting." % (gap, pk_ratio))
    if "unseen_extrap" in summ and "unseen_interp" in summ:
        if summ["unseen_interp"]["mean_r2"] - summ["unseen_extrap"]["mean_r2"] > GAP_R2:
            print("  EXTRAPOLATION GAP: shapes outside the training range are worse (worst: %s). This is missing "
                  "data coverage, not overfitting." % summ["unseen_extrap"]["worst_case"])

    # ---- figure -----------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axs[0]
    if lr:
        ep, loss, _ = lr
        ax.semilogy(ep, loss, color="#bbbbbb", lw=0.6, label="loss each epoch")
        wdw = max(1, min(200, len(loss) // 20))
        sm = np.convolve(loss, np.ones(wdw) / wdw, mode="valid")
        ax.semilogy(ep[wdw - 1:], sm, color="#2a78d6", lw=2, label="moving average (%d epochs)" % wdw)
        ax.axvline(ep[int(loss.argmin())], color="#eb6834", ls="--", lw=1, label="lowest loss (mgn_best.pt)")
        ax.set_xlabel("epoch"); ax.set_ylabel("training loss (MSE of z-scored stress)"); ax.legend(fontsize=9)
    ax.set_title("(a) training loss: still falling = under-trained", loc="left", fontsize=11)
    keys = [k for k, _ in LEVELS if k in summ]
    cols = ["#555555", "#2a78d6", "#eb6834", "#1baf7a"]
    for j, (ax, field, ylab, ttl) in enumerate(((axs[1], "r2", "R2 per file (clipped at -1)", "(b) R2 at every level"),
                                                (axs[2], "peak", "|peak error| per file (%)", "(c) peak-stress error at every level"))):
        for i, k in enumerate(keys):
            if field == "r2":
                v = np.clip([r["r2"] for r in sets[k]], -1, 1)
            else:
                v = np.array([abs(r["peak_err_pct"]) for r in sets[k]])
            x = i + np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=7, color=cols[i % 4], alpha=0.45)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="k", lw=2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([TICK[k] for k in keys], fontsize=9.5)
        ax.set_ylabel(ylab); ax.set_title(ttl + "  (black line = median)", loc="left", fontsize=11)
        ax.grid(alpha=0.3)
        if field == "r2":
            ax.set_ylim(-1.05, 1.02)
        else:
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(bottom=0)
    fig.suptitle("Fit check  -  %s" % a.ckpt, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_check.png"), dpi=130)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
