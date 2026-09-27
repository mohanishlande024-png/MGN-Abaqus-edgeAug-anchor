r"""
Train Run B1: edge-augmented MGN + anchor features in every layer + random
edges re-drawn every epoch.

    python scripts/train_anchor.py --data data/dataset_own_aug20.pt
    python scripts/train_anchor.py --data data/dataset_own_aug20.pt --resume
    python scripts/train_anchor.py --data data/dataset_own_aug20.pt --epochs 50      quick check

    ablations (same script, one change switched off):
    --no-anchors      re-drawn edges only
    --no-redraw       anchors only, random edges fixed as before

Uses the SAME dataset file as your current edge-augmented run - nothing to
rebuild. The anchors are computed from the mesh stored in it (first epoch
prints a table of what was found per geometry, so you can check it).

Everything else is identical to scripts/train.py and the paper: 20 layers,
hidden 64, embedding 16, lr 1e-5, batch 1, 10000 epochs, gradient clipping in
the paper's own _train_batch(), seed 0.

WRITES (default folder results_aug20_anchors_redraw/ - your other results are untouched)
    mgn_latest.pt        weights + optimiser + epoch, for --resume
    mgn_best.pt          weights at the lowest training loss (+ the run's config)
    loss_log.csv         epoch, loss, seconds
    anchors_summary.csv  what the anchor features found in each training geometry
    multi_geometry_mgn.pt   full model with its normalisation, when the run finishes
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import load_dataset                                  # noqa: E402
from mgn.anchor_trainer import AnchorMGN, results_dir_for_b1         # noqa: E402
from mgn.anchors import ANCHOR_NAMES, anchor_features                 # noqa: E402


def fmt(seconds):
    s = int(seconds)
    return "%dh%02dm%02ds" % (s // 3600, (s % 3600) // 60, s % 60)


def anchor_summary(mgn, cases, path):
    """One line per training geometry: hole nodes, corners, strengths."""
    seen, rows = set(), []
    for c in cases:
        if c.geometry in seen:
            continue
        seen.add(c.geometry)
        _, info = anchor_features(np.asarray(c.coordinates), mgn._mesh_edges(c), return_info=True)
        rows.append((c.geometry, info["n_hole_loops"], info["n_hole_nodes"], len(info["corners"]),
                     float(np.max(info["hole_strength"])) if info["n_hole_nodes"] else 0.0,
                     float(np.max(info["corner_strength"])) if len(info["corners"]) else 0.0))
    with open(path, "w") as fh:
        fh.write("geometry,hole_loops,hole_nodes,corners,max_hole_strength,max_corner_strength\n")
        for r in rows:
            fh.write("%s,%d,%d,%d,%.4f,%.4f\n" % r)
    print("\nANCHORS FOUND  (hole strength = log(1 + curvature x H); corner strength = excess turning, rad)")
    print("  %-26s %6s %7s %8s %10s %10s" % ("geometry", "holes", "h.nodes", "corners", "max hole s", "max corn s"))
    for r in rows:
        print("  %-26s %6d %7d %8d %10.2f %10.2f" % r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join("data", "dataset_own_aug20.pt"))
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-anchors", action="store_true", help="ablation: switch the anchor features off")
    ap.add_argument("--no-redraw", action="store_true", help="ablation: keep one fixed random-edge draw")
    ap.add_argument("--results", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    if not os.path.isfile(a.data):
        print("No %s." % a.data)
        return 1
    cases, aug = load_dataset(a.data, return_meta=True)
    if aug["aug_perc"] <= 0 and not a.no_redraw:
        print("%s has no random edges (aug_perc = 0), so there is nothing to re-draw. "
              "Use an augmented dataset (e.g. data/dataset_own_aug20.pt) or pass --no-redraw." % a.data)
        return 1

    anchors, redraw = not a.no_anchors, not a.no_redraw
    RESULTS = a.results or results_dir_for_b1(aug["aug_perc"], anchors, redraw)
    LATEST, BEST = os.path.join(RESULTS, "mgn_latest.pt"), os.path.join(RESULTS, "mgn_best.pt")
    LOG = os.path.join(RESULTS, "loss_log.csv")
    os.makedirs(RESULTS, exist_ok=True)

    y = np.concatenate([c.von_mises for c in cases])
    geoms = sorted(set(c.geometry for c in cases))
    mgn = AnchorMGN(num_layers=a.layers, hidden_channels=a.hidden, embedding_dim=a.embedding,
                    learning_rate=a.lr, epochs=a.epochs, global_features=["load"],
                    aug_perc=aug["aug_perc"], aug_seed=aug["aug_seed"], anchors=anchors, redraw=redraw)
    config = mgn.config()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("MGN TRAINING - RUN B1")
    print("=" * 70)
    print("cases          %d  (%d geometries x %d loads)" % (len(cases), len(geoms), len(cases) // max(len(geoms), 1)))
    print("device         %s%s" % (dev, "  (" + torch.cuda.get_device_name(0) + ")" if dev == "cuda" else ""))
    print("epochs         %d   batch %d   lr %g   layers %d   hidden %d" % (a.epochs, a.batch_size, a.lr, a.layers, a.hidden))
    print("augmentation   %.0f%% random edges, base seed %d, %s"
          % (100 * aug["aug_perc"], aug["aug_seed"], "RE-DRAWN every epoch" if redraw else "fixed (one draw)"))
    print("anchors        %s" % ("hole + corner, 14 numbers per node, in every layer" if anchors else "off"))
    print("results        %s" % RESULTS)
    print("=" * 70)

    t_prep = time.time()
    mgn._train_fem = cases
    mgn._y_train = torch.tensor(y, dtype=torch.float).squeeze()
    fem_data = mgn._preprocess_fems(cases)
    mgn._compute_normalization_stats(fem_data)
    mgn._build_model()
    if anchors:
        anchor_summary(mgn, cases, os.path.join(RESULTS, "anchors_summary.csv"))
        names = ["load"] + ANCHOR_NAMES
        print("\nglobal inputs  %d columns: %s" % (len(names), ", ".join(names)))
    print("parameters     %d" % sum(p.numel() for p in mgn._model.parameters()))
    print("preprocessing  %.0f s" % (time.time() - t_prep))

    optimizer = torch.optim.Adam(mgn._model.parameters(), lr=a.lr)
    loss_fn = torch.nn.MSELoss()
    start_epoch, best = 0, float("inf")

    if a.resume and os.path.isfile(LATEST):
        ck = torch.load(LATEST, map_location=mgn.device, weights_only=False)
        if ck.get("augmentation") != aug or ck.get("config", {}).get("anchors") != anchors \
                or ck.get("config", {}).get("redraw") != redraw:
            print("\nREFUSING TO RESUME: %s was trained with a different setup (%s, %s)."
                  % (LATEST, ck.get("augmentation"), ck.get("config")))
            return 1
        mgn._model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best = ck["epoch"], ck.get("best", float("inf"))
        mgn.train_losses = ck.get("train_losses", [])
        np.random.seed(a.seed + start_epoch)             # shuffling continues from a fresh, reproducible state
        print("\nRESUMED from epoch %d  (best loss %.6f)" % (start_epoch, best))
    elif a.resume:
        print("\n--resume given but no %s found; starting fresh." % LATEST)

    if not os.path.isfile(LOG) or start_epoch == 0:
        open(LOG, "w").write("epoch,loss,seconds\n")

    def state(epoch):
        return {"model": mgn._model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best": best, "train_losses": mgn.train_losses, "augmentation": aug, "config": config}

    print("\nepoch      loss        elapsed     eta")
    print("-" * 52)
    t0, interrupted = time.time(), False
    try:
        for epoch in range(start_epoch, a.epochs):
            mgn.redraw_edges(cases, fem_data, epoch)                  # new random edges for this epoch
            loss = mgn._train_epoch(cases, fem_data, a.batch_size, optimizer, loss_fn)   # the paper's step
            mgn.train_losses.append(loss)
            elapsed = time.time() - t0
            eta = elapsed / (epoch - start_epoch + 1) * (a.epochs - epoch - 1)
            open(LOG, "a").write("%d,%.8f,%.1f\n" % (epoch + 1, loss, elapsed))
            if loss < best:
                best = loss
                torch.save({"model": mgn._model.state_dict(), "epoch": epoch + 1, "loss": loss,
                            "augmentation": aug, "config": config}, BEST)
            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print("%6d   %.6f   %s   %s" % (epoch + 1, loss, fmt(elapsed), fmt(eta)))
            if (epoch + 1) % a.ckpt_every == 0:
                torch.save(state(epoch + 1), LATEST)
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted. Saving checkpoint...")

    torch.save(state(len(mgn.train_losses)), LATEST)
    print("\n" + "=" * 70)
    print("%s at epoch %d after %s" % ("Stopped" if interrupted else "Finished", len(mgn.train_losses),
                                      fmt(time.time() - t0)))
    if mgn.train_losses:
        print("final loss     %.6f\nbest loss      %.6f" % (mgn.train_losses[-1], best))
    if interrupted or len(mgn.train_losses) < a.epochs:
        print("\nContinue with:  python scripts/train_anchor.py --resume --data %s%s%s"
              % (a.data, " --no-anchors" if not anchors else "", " --no-redraw" if not redraw else ""))
    else:
        mgn.save(os.path.join(RESULTS, "multi_geometry_mgn.pt"))
    print("\nEvaluate with:  python scripts/eval_own_unseen.py --ckpt %s" % BEST)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
