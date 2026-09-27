"""
AnchorMGN - Run B1: the edge-augmented MGN + anchor features in every layer
+ random edges re-drawn every epoch.

BUILT ON AugMGN, WHICH IS BUILT ON THE PAPER'S MGN
    mgn/trainer.py (the paper's code) and mgn/augmented_trainer.py (your
    edge-augmented model) are NOT modified. Everything for B1 lives here, and
    with anchors=False, redraw=False this class behaves exactly like AugMGN.

1. ANCHOR FEATURES IN EVERY LAYER
    mgn/anchors.py gives each node 14 numbers (hole + corner, 7 each). They are
    appended to the node's global feature vector, next to the load:

        global features per node:  [load]  ->  [load, 14 anchor numbers]

    The paper's model already feeds that vector through the global encoder and
    into EVERY one of the 20 node updates:
        h_i(l+1) = MLP_update( [ h_i(l),  sum_j m_ij(l),  g_i ] ),  g_i = MLP_global([load, anchors_i])
    so the anchors reach every layer with no change to model.py - only the
    global input grows from 1 to 15 columns. (Adding them only at layer 0,
    as the transformer paper does, was measured to fade out in this model: it
    has no residual connections, each layer overwrites h.)
    The 15 columns are z-scored per column with training-set statistics by the
    paper's own normalisation code, and those statistics are saved in the
    checkpoint.

2. RANDOM EDGES RE-DRAWN EVERY EPOCH
    AugMGN keeps ONE fixed set of random edges per geometry, and the model
    learned to recognise each training shape by it (a training shape falls from
    R2 0.9995 to ~0.82 when only its random edges are re-drawn). Here the random
    edges are re-drawn at the start of every epoch:

        epoch e uses seed = crc32(geometry name) + aug_seed + e

    so epoch 0 is exactly the draw stored in the dataset, every epoch is
    reproducible, and a resumed run continues with the same edges it would have
    had. The real mesh edges never change. The normalisation of the random edges
    (mean / std of their length and direction) is fixed from the epoch-0 draw -
    uniform random pairs have the same statistics in every draw.

    At test time any draw is valid; eval_own_unseen.py --draws K averages K draws.
"""

import numpy as np
import torch

from .anchors import ANCHOR_DIM, ANCHOR_NAMES, PARAMS as ANCHOR_PARAMS, anchor_features
from .augmented_trainer import AugMGN
from .graph import augment_edges, geometry_seed
from .model import MGNModel


def results_dir_for_b1(aug_perc, anchors=True, redraw=True):
    """results_aug20_anchors_redraw etc. - never the folder of another run."""
    name = "results_aug%d" % round(100 * aug_perc) if aug_perc > 0 else "results"
    if anchors:
        name += "_anchors"
    if redraw and aug_perc > 0:
        name += "_redraw"
    return name


class AnchorMGN(AugMGN):

    def __init__(self, *, anchors=True, redraw=True, **kwargs):
        super().__init__(**kwargs)
        self.use_anchors = bool(anchors)
        self.redraw = bool(redraw) and self.augmented
        self.anchor_dim = ANCHOR_DIM if self.use_anchors else 0
        self._anchor_cache = {}

    def config(self):
        return dict(anchors=self.use_anchors, redraw=self.redraw, anchor_dim=self.anchor_dim,
                    anchor_params=ANCHOR_PARAMS if self.use_anchors else None,
                    layers=self.num_layers, hidden=self.hidden_channels,
                    embedding=self.embedding_dim, lr=self.learning_rate)

    # ------------------------------------------------------------------ anchors

    def _mesh_edges(self, fem):
        """The real mesh edges of a graph (edge_flag == 0), as a numpy (2, E) array."""
        ei = fem.edge_index
        ei = ei.cpu().numpy() if torch.is_tensor(ei) else np.asarray(ei)
        return ei[:, self._flag_of(fem).numpy() < 0.5]

    def anchors_for(self, fem):
        """(N, 14) anchor features of one mesh, computed once per geometry."""
        key = (getattr(fem, "geometry", ""), len(fem.coordinates))
        if key not in self._anchor_cache:
            self._anchor_cache[key] = anchor_features(np.asarray(fem.coordinates), self._mesh_edges(fem))
        return self._anchor_cache[key]

    def _fem_to_graph(self, fem_list):
        node_types, coords, edge_index, glob = super()._fem_to_graph(fem_list)
        if not self.use_anchors:
            return node_types, coords, edge_index, glob
        A = torch.from_numpy(np.concatenate([self.anchors_for(f) for f in fem_list], axis=0))
        glob = A if glob is None else torch.cat([glob, A.to(glob.dtype)], dim=1)
        return node_types, coords, edge_index, glob

    def _compute_normalization_stats(self, fem_data):
        super()._compute_normalization_stats(fem_data)      # z-scores [load, anchors...] per column
        if self.use_anchors and self._global_std is not None:
            flat = self._global_std < 1e-6                   # a column that never varies: leave it unscaled
            self._global_std = torch.where(flat, torch.ones_like(self._global_std), self._global_std)

    def _build_model(self):
        self._model = MGNModel(
            num_node_types=self.num_node_types,
            embedding_dim=self.embedding_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            global_feature_dim=len(self.global_features) + self.anchor_dim,
        ).to(self.device)

    # ------------------------------------------------------------------ re-draw

    def redraw_edges(self, cases, fem_data, epoch):
        """Replace the random edges of every training graph with the draw for
        this epoch. Graphs of the same geometry (its 20 loads) share one draw."""
        if not self.redraw:
            return
        done = {}
        for case, fd in zip(cases, fem_data):
            key = (case.geometry, len(case.coordinates))
            if key not in done:
                mesh = self._mesh_edges(case)
                ei, flag = augment_edges(mesh, len(case.coordinates), perc=self.aug_perc,
                                         seed=geometry_seed(case.geometry, self.aug_seed + int(epoch)))
                ei = torch.from_numpy(ei)
                raw = self._compute_edge_features(np.asarray(case.coordinates), ei)     # [dx, dy, length]
                attr = self._normalize_edges(torch.cat([raw, torch.from_numpy(flag).unsqueeze(1)], dim=1))
                done[key] = (ei, attr)
            fd["edge_index"], fd["edge_attr"] = done[key]

    # ------------------------------------------------------------------ save / load

    def save(self, filepath):
        super().save(filepath)
        ck = torch.load(filepath, map_location="cpu", weights_only=False)
        ck["b1"] = self.config()
        ck["anchor_names"] = ANCHOR_NAMES if self.use_anchors else []
        torch.save(ck, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(filepath, map_location=device, weights_only=False)
        if "b1" not in ck:
            raise ValueError("%s is not an AnchorMGN checkpoint - load it with AugMGN.load" % filepath)
        cfg = ck["b1"]
        mgn = cls(embedding_dim=ck["embedding_dim"], hidden_channels=ck["hidden_channels"],
                  num_layers=ck["num_layers"], learning_rate=ck["learning_rate"],
                  global_features=ck.get("global_features", []),
                  aug_perc=ck.get("aug_perc", 0.0), aug_seed=ck.get("aug_seed", 0),
                  anchors=cfg["anchors"], redraw=cfg["redraw"])
        mgn.device = device
        mgn.num_node_types = ck["num_node_types"]
        mgn.node_type_to_id = ck["node_type_to_id"]
        mgn._y_mean, mgn._y_std = ck["y_mean"].to(device), ck["y_std"].to(device)
        mgn._edge_attr_mean = ck["edge_attr_mean"].to(device)
        mgn._edge_attr_std = ck["edge_attr_std"].to(device)
        if mgn.augmented:
            for k in ("real_edge_mean", "real_edge_std", "aug_edge_mean", "aug_edge_std"):
                setattr(mgn, "_" + k, ck[k].to(device))
        if ck.get("global_mean") is not None:
            mgn._global_mean, mgn._global_std = ck["global_mean"].to(device), ck["global_std"].to(device)
        mgn.train_losses = ck.get("train_losses", [])
        mgn._build_model()
        mgn._model.load_state_dict(ck["model_state_dict"])
        mgn._model.eval()
        print("Model loaded from %s  (edge augmentation %.0f%%, anchors %s, re-drawn edges %s)"
              % (filepath, 100 * mgn.aug_perc, cfg["anchors"], cfg["redraw"]))
        return mgn
