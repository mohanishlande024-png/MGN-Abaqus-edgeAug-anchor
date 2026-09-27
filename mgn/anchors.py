"""
Concentration-aware ANCHOR FEATURES for the MGN (Run B1).

Stress concentrates at a few known kinds of places. Every node is told, for
each kind, how far away the nearest one is, in which direction, and how severe
it is - so the network does not have to discover this through message passing.

TWO ANCHOR TYPES (the ones that vary between your plates)
    hole    every node on a hole boundary is an anchor.
            strength s = sign(k) * log(1 + |k| * H)
                k = local curvature of the hole edge at that anchor (1 / radius),
                    positive where the hole bulges into the material (a stress
                    raiser), negative where the material bulges into the hole
                H = plate height, so s has no units (same numbers in inches or mm)
            e.g. H = 10 in: straight side 0, 3 in circle 2.04, 0.12 in fillet 4.4,
                 tip of the b/a = 8 ellipse (radius 0.066 in) 5.0
    corner  the sharpest turning points of each hole edge: nodes where the edge
            turns most within a short window (+-0.02 H of arc), kept only if they
            turn more than the edge does on average.
            strength s = excess turning in radians
                (turning in the window - the loop's average turning per window)
            rectangle fillets ~1.4, triangle corners ~1.8, decagon corners ~0.4,
            ellipse tips from ~0.1 (a/b = 1.25) to ~2.7 (a/b = 8), circle: none

SEVEN NUMBERS PER TYPE (the paper's channels)
    [ r/H,  log(r/H + eps),  cos t,  sin t,  cos 2t,  sin 2t,  s ]
    r = distance from the node to the nearest anchor of that type
    t = atan2(y_node - y_anchor, x_node - x_anchor), global x-y frame as in
        the paper's Eq. (1); the load always pulls along x, so the frame is fixed
    For a node lying ON the hole edge (r = 0) t is the direction into the
    material. A type that does not exist on a plate (no corners on a circle,
    no hole at all) gets r/H = ABSENT_R and zeros for the angles and strength.
    -> 2 x 7 = 14 numbers per node (ANCHOR_DIM).

NO TRIANGLES NEEDED
    The hole outline is rebuilt from the real mesh edges (edge_flag == 0) that
    are already stored in dataset_own_aug20.pt: an edge on the boundary belongs
    to exactly one triangle, i.e. its two end nodes share exactly one common
    neighbour. Training and every evaluation script use this same path, so the
    features are identical everywhere.

Pure numpy, no scipy.
"""

import math

import numpy as np

ANCHOR_TYPES = ("hole", "corner")
CHANNELS = ("r", "log_r", "cos", "sin", "cos2", "sin2", "s")
ANCHOR_DIM = len(ANCHOR_TYPES) * len(CHANNELS)          # 14
ANCHOR_NAMES = ["%s_%s" % (t, c) for t in ANCHOR_TYPES for c in CHANNELS]

EPS = 1e-3            # inside log(r/H + eps)
WINDOW = 0.02         # corner window half-length, as a fraction of H
MIN_EXCESS = 0.05     # rad - a turning point must beat the loop average by this to count as a corner
ABSENT_R = 6.0        # r/H used when a plate has no anchor of that type (the plate is 6 H long)
VERSION = 1           # stored in checkpoints; bump if the definitions above change

PARAMS = dict(version=VERSION, eps=EPS, window=WINDOW, min_excess=MIN_EXCESS, absent_r=ABSENT_R,
              types=list(ANCHOR_TYPES), channels=list(CHANNELS))


# --------------------------------------------------------------- boundary loops

def boundary_edges_from_mesh(mesh_edge_index, n_nodes):
    """Undirected boundary edges (M, 2) from the REAL mesh edges only.
    An interior edge borders 2 triangles, a boundary edge exactly 1, and each
    triangle an edge borders shows up as one common neighbour of its two ends."""
    e = np.asarray(mesh_edge_index, dtype=np.int64)
    und = np.unique(np.sort(e.T, axis=1), axis=0)
    adj = [set() for _ in range(n_nodes)]
    for a, b in und:
        adj[a].add(int(b))
        adj[b].add(int(a))
    keep = [len(adj[a] & adj[b]) == 1 for a, b in und]
    return und[np.array(keep, dtype=bool)]


def ordered_loops(bedges):
    """Walk the boundary edges into closed, ordered loops of node indices."""
    nbr = {}
    for a, b in bedges:
        nbr.setdefault(int(a), []).append(int(b))
        nbr.setdefault(int(b), []).append(int(a))
    seen, loops = set(), []
    for start in sorted(nbr):
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [v for v in nbr[cur] if v != prev]
            if not nxt:
                break
            nxt = nxt[0] if (prev is not None or len(nxt) == 1) else min(nxt)
            if nxt == start or nxt in seen:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
    return loops


def signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def hole_loops(coords, mesh_edge_index):
    """Hole boundary loops, each ordered COUNTER-CLOCKWISE. The outer plate
    boundary is the loop with the largest bounding box (same rule as
    mgn.graph.boundary_loops) and is dropped."""
    loops = ordered_loops(boundary_edges_from_mesh(mesh_edge_index, len(coords)))
    if not loops:
        return []

    def bbox_area(l):
        c = coords[l]
        span = c.max(axis=0) - c.min(axis=0)
        return float(span[0] * span[1])

    outer = max(range(len(loops)), key=lambda i: bbox_area(loops[i]))
    out = []
    for i, l in enumerate(loops):
        if i == outer:
            continue
        out.append(l if signed_area(coords[l]) > 0 else l[::-1])
    return out


# --------------------------------------------------------------- loop geometry

def loop_geometry(coords, loop, H):
    """Per node of one CCW hole loop:
        kappa    signed curvature (1/length); > 0 where the hole is convex, which
                 for the material around it is a re-entrant, stress-raising edge
        normal   unit normal pointing INTO the material
        turn     total turning of the edge within +-w of arc (rad) - the corner STRENGTH
        turn_s   the same within +-w/4 - peaks sharply at a vertex, used to LOCATE corners
        arc      arc-length position along the loop
    w = WINDOW * H, widened to 1.5 x the longest boundary segment on coarse
    meshes so that a circle drawn with few nodes is not read as a ring of
    corners. For a CCW loop the material is on the right, so a left turn is
    convex for the hole."""
    p = coords[loop].astype(np.float64)
    prv, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    d1, d2 = p - prv, nxt - p
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    dot = (d1 * d2).sum(axis=1)
    phi = np.arctan2(cross, dot)                                     # signed turning at each node
    chord = np.linalg.norm(nxt - prv, axis=1)
    kappa = 2.0 * np.sin(phi) / np.maximum(chord, 1e-12)             # 1 / circumradius, signed

    tang = nxt - prv
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)             # right of the CCW direction = material

    seg = np.linalg.norm(d2, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    P = float(seg.sum())
    w = max(WINDOW * H, 1.5 * float(seg.max()))
    da = np.abs(arc[:, None] - arc[None, :])
    da = np.minimum(da, P - da)                                      # distance along a closed loop
    turn = (phi[None, :] * (da <= w)).sum(axis=1)
    turn_s = (phi[None, :] * (da <= 0.25 * w)).sum(axis=1)
    return dict(kappa=kappa, normal=normal, turn=turn, turn_s=turn_s, arc=arc, perimeter=P, w=w,
                dist=da, mean_turn=2.0 * math.pi * (2.0 * w) / max(P, 1e-12))


def corner_indices(g, H):
    """Corners = nodes where the narrow-window turning peaks (within +-w) and
    the wide-window turning beats the loop's average by more than MIN_EXCESS.
    Returns (indices, strengths); strength = excess wide-window turning (rad)."""
    excess = g["turn"] - g["mean_turn"]
    turn_s, da, w = g["turn_s"], g["dist"], g["w"]
    idx, strength = [], []
    for i in range(len(excess)):
        if excess[i] <= MIN_EXCESS:
            continue
        win = np.where(da[i] <= w)[0]
        best = win[np.lexsort((win, -turn_s[win]))[0]]               # sharpest node, ties -> lowest index
        if best == i:
            idx.append(i)
            strength.append(excess[i])
    return np.array(idx, dtype=np.int64), np.array(strength, dtype=np.float64)


# --------------------------------------------------------------- the features

def _nearest(points, anchors):
    """Index of, and distance to, the nearest anchor for every point (chunked brute force)."""
    best_i = np.zeros(len(points), dtype=np.int64)
    best_d = np.zeros(len(points))
    for s in range(0, len(points), 4096):
        d2 = ((points[s:s + 4096, None, :] - anchors[None, :, :]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        best_i[s:s + 4096] = j
        best_d[s:s + 4096] = np.sqrt(d2[np.arange(len(j)), j])
    return best_i, best_d


def _channels(coords, H, anchor_xy, anchor_s, anchor_normal):
    n = len(coords)
    if anchor_xy is None or len(anchor_xy) == 0:
        out = np.zeros((n, len(CHANNELS)))
        out[:, 0] = ABSENT_R
        out[:, 1] = math.log(ABSENT_R + EPS)
        return out
    j, r = _nearest(coords, anchor_xy)
    d = coords - anchor_xy[j]
    theta = np.arctan2(d[:, 1], d[:, 0])
    on_edge = r < 1e-9 * H                                           # the node IS the anchor
    if on_edge.any():
        nrm = anchor_normal[j[on_edge]]
        theta[on_edge] = np.arctan2(nrm[:, 1], nrm[:, 0])
    rh = r / H
    return np.stack([rh, np.log(rh + EPS), np.cos(theta), np.sin(theta),
                     np.cos(2 * theta), np.sin(2 * theta), anchor_s[j]], axis=1)


def anchor_features(coords, mesh_edge_index, return_info=False):
    """
    (N, 14) float32 anchor features for one mesh.

    coords           (N, 2) node coordinates (any length unit)
    mesh_edge_index  (2, E) the REAL mesh edges only (no augmented edges)
    """
    coords = np.asarray(coords, dtype=np.float64)
    H = float(coords[:, 1].max() - coords[:, 1].min())
    loops = hole_loops(coords, mesh_edge_index)

    h_xy, h_s, h_n, c_xy, c_s, c_n = [], [], [], [], [], []
    for l in loops:
        g = loop_geometry(coords, l, H)
        h_xy.append(coords[l])
        h_s.append(np.sign(g["kappa"]) * np.log1p(np.abs(g["kappa"]) * H))
        h_n.append(g["normal"])
        ci, cs = corner_indices(g, H)
        if len(ci):
            c_xy.append(coords[l][ci])
            c_s.append(cs)
            c_n.append(g["normal"][ci])

    cat = lambda xs, k: np.concatenate(xs, axis=0) if xs else np.zeros((0,) + k)
    hole = _channels(coords, H, cat(h_xy, (2,)), cat(h_s, ()), cat(h_n, (2,)))
    corner = _channels(coords, H, cat(c_xy, (2,)), cat(c_s, ()), cat(c_n, (2,)))
    feats = np.concatenate([hole, corner], axis=1).astype(np.float32)
    if not return_info:
        return feats
    return feats, dict(H=H, n_hole_loops=len(loops), n_hole_nodes=int(sum(len(l) for l in loops)),
                       corners=cat(c_xy, (2,)), corner_strength=cat(c_s, ()),
                       hole_xy=cat(h_xy, (2,)), hole_strength=cat(h_s, ()))
