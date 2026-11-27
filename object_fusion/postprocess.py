"""Clean-up passes between instance fusion and naming.

Fusion leaves two artefacts that show up directly in the scene graph:

  * **Fragments.** One physical object can end up as several instances when it
    is seen from angles whose voxel sets barely overlap — e.g. a 3RScan living
    room produced `raven_0` and `ursus americanus_0` at the same (x, y),
    stacked 30 cm apart, and three more fragments of one object near the
    ceiling. Each fragment then becomes its own node, and the relation builder
    reports nonsense like `floor lamp on pillow`.
  * **Room structure as objects.** Prompt-free YOLOE segments the floor as
    readily as a chair, so a large flat instance at floor height becomes an
    object node — `parquet_0` (2.2 x 1.4 x 0.15 m at z=0.03) — competing with
    the graph's own floor root.

Both passes run before naming, so a merged instance is named once as a whole.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _bounds(view_dataset, obj) -> tuple[np.ndarray, np.ndarray]:
    pts = view_dataset.index_to_point(obj["indexes"])
    return pts.min(axis=0), pts.max(axis=0)


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-8 else 0.0


def _weighted_mean_embedding(a, n_a: int, b, n_b: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    mean = (a * n_a + b * n_b) / (n_a + n_b)
    return mean / max(float(np.linalg.norm(mean)), 1e-8)


def _boxes_touch(box_a, box_b, margin: float) -> bool:
    (lo_a, hi_a), (lo_b, hi_b) = box_a, box_b
    return bool(np.all(lo_a - margin <= hi_b) and np.all(lo_b - margin <= hi_a))


def _point_containment(part_points: np.ndarray, host_tree: cKDTree, distance: float) -> float:
    """Share of the part's points that lie within `distance` of a host point."""
    dist, _ = host_tree.query(part_points, k=1, distance_upper_bound=distance)
    return float(np.isfinite(dist).mean())


def _best_track_name(obj) -> dict | None:
    """The track-end decode backed by the most detections."""
    records = obj.get("track_names") or []
    return max(records, key=lambda r: r["num_detections"]) if records else None


def _names_conflict(a, b, min_score: float) -> bool:
    """Both objects were confidently named, the names differ, and neither name
    is among the other's beam alternatives ("laptop" vs "laptop computer" is
    not a conflict when each lists the other)."""
    name_a, name_b = _best_track_name(a), _best_track_name(b)
    if name_a is None or name_b is None or min(name_a["score"], name_b["score"]) < min_score:
        return False
    if name_a["label"] == name_b["label"]:
        return False
    alternatives_a = {name_a["label"], *(label for label, _ in name_a.get("top_k") or [])}
    alternatives_b = {name_b["label"], *(label for label, _ in name_b.get("top_k") or [])}
    return name_a["label"] not in alternatives_b and name_b["label"] not in alternatives_a


def drop_structural_instances(objects, view_dataset, min_area_m2: float = 1.0,
                              max_thickness_m: float = 0.2, floor_band_m: float = 0.25):
    """Removes large, flat, floor-height instances — floor, rug, parquet.

    The scene graph already has a synthetic floor root that everything rests
    on, so keeping these as objects adds a duplicate parent and steals
    `on` relations from real furniture. Only applies in a gravity-aligned
    world, where z is up and the floor sits at z=0.
    """
    kept, dropped = [], []
    for obj in objects:
        lo, hi = _bounds(view_dataset, obj)
        size = hi - lo
        area = float(size[0] * size[1])
        is_floorish = (area >= min_area_m2
                       and size[2] <= max_thickness_m
                       and float(lo[2]) <= floor_band_m)
        (dropped if is_floorish else kept).append(obj)

    if dropped:
        logger.info("Dropped %d floor-like instance(s): %s", len(dropped),
                    [o.get("class_id") or o.get("class_name") for o in dropped])
    return kept, dropped


def _densest_slab(z: np.ndarray, width_m: float) -> tuple[float, float]:
    """(share of points, upper height) of the horizontal slab `width_m` thick
    holding the most points."""
    z = np.sort(z)
    ends = np.searchsorted(z, z + width_m, side="right")
    start = int(np.argmax(ends - np.arange(len(z))))
    return (ends[start] - start) / len(z), float(z[ends[start] - 1])


def find_support_surfaces(objects, view_dataset, min_area_m2: float = 0.8, min_top_m: float = 0.3,
                          max_top_m: float = 1.3, min_slab_fraction: float = 0.5,
                          slab_m: float = 0.10, cell_m: float = 0.05) -> set[int]:
    """Indexes of instances shaped like a table or desk top: an XY footprint of
    at least `min_area_m2`, and at least `min_slab_fraction` of the points in
    one horizontal slab `slab_m` thick whose top is at furniture height. A
    slab, not "points near the top": run-to-run pose differences tilt a 1.7 m
    desk by a few degrees, which spreads its top over several centimetres.

    NOVIC names such an instance after whatever stands on it, because its crop
    is mostly that clutter: on IMG_6100 the desk (1.7 x 1.6 m, top at 0.9 m)
    decoded as "laptop". Needs a gravity-aligned world (z up, floor at 0).

    Largest footprint first; a candidate whose top is within `slab_m` of an
    accepted surface's and whose footprint touches it stands on that surface
    (an object that absorbed some of the table top around it), not a second
    one: on Lap_remove_2 the iron, with a 0.16 m2 patch of the table, was
    otherwise renamed "table".
    """
    candidates = []
    for i, obj in enumerate(objects):
        points = view_dataset.index_to_point(obj["indexes"])
        footprint_cells = np.unique(np.floor(points[:, :2] / cell_m).astype(np.int64), axis=0)
        if len(footprint_cells) * cell_m ** 2 < min_area_m2:
            continue
        fraction, top = _densest_slab(points[:, 2], slab_m)
        if fraction >= min_slab_fraction and min_top_m <= top <= max_top_m:
            candidates.append((len(footprint_cells), i, top, {tuple(c) for c in footprint_cells.tolist()}))
    surfaces, accepted = set(), []
    for _, i, top, cells in sorted(candidates, key=lambda c: -c[0]):
        if any(abs(top - t) <= slab_m and _touches(cells, other) for t, other in accepted):
            logger.info("Not a support surface: '%s' lies in the plane of a larger one it touches.",
                        objects[i].get("class_id") or objects[i].get("class_name") or i)
            continue
        surfaces.add(i)
        accepted.append((top, cells))
    return surfaces


def _touches(cells: set, other: set) -> bool:
    """Whether two sets of XY grid cells overlap or are 8-neighbours."""
    return any((x + dx, y + dy) in other for x, y in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))


def _split_raised_part(objects, k, view_dataset, coef, split_height_m: float, split_min_voxels: int):
    """Splits the largest compact cluster of objects[k]'s voxels lying more than
    `split_height_m` above the plane z = coef . (x, y, 1) into a new instance
    appended to `objects` (same names, embedding and detections), when it has
    at least `split_min_voxels`. On lap_in_fast the only large flat instance was
    the laptop, merged with the tablet and the table edge around them (YOLOE
    never segments that yellow table top): it became the table and the laptop
    was lost. Its screen, 14 cm above the plane, is what this keeps. Returns the
    new instance's index, or None."""
    import open3d as o3d

    obj = objects[k]
    idx = np.asarray(obj["indexes"])
    pts = np.asarray(view_dataset.index_to_point(idx))
    raised = np.where(pts[:, 2] - (pts[:, :2] @ coef[:2] + coef[2]) > split_height_m)[0]
    if len(raised) < split_min_voxels:
        return None
    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[raised].astype(np.float64)))
                        .cluster_dbscan(eps=0.03, min_points=10))
    if labels.max() < 0:
        return None
    largest = np.bincount(labels[labels >= 0]).argmax()
    part = raised[labels == largest]
    if len(part) < split_min_voxels:
        return None
    rest = np.ones(len(idx), bool)
    rest[part] = False
    import copy

    objects.append({**copy.deepcopy({key: v for key, v in obj.items() if key not in ("indexes", "mask")}),
                    "indexes": idx[part], "split_from_surface": True})
    obj["indexes"] = idx[rest]
    logger.info("Support surface seed '%s': %d of its voxels stand more than %.2f m above the plane, split off as "
                "their own instance (an object it had absorbed); %d remain for the table top.",
                obj.get("class_id") or obj.get("class_name") or k, len(part), split_height_m, int(rest.sum()))
    return len(objects) - 1


def complete_support_surfaces(objects, view_dataset, min_top_m: float = 0.3, max_top_m: float = 1.3,
                              seed_min_area_m2: float = 0.15, seed_min_fraction: float = 0.25,
                              plane_tol_m: float = 0.06, bridge_m: float = 0.06,
                              vertical_band_m: float = 0.30, slab_m: float = 0.05, cell_m: float = 0.02,
                              split_height_m: float = 0.10, split_min_voxels: int = 500):
    """Fills in a table top from the voxel map where its instance has holes.

    YOLOE's table mask stops at the edge of everything standing on the table,
    so the fused table misses exactly the patches under the objects. Its 2D
    footprint then doesn't reach them and the relation builder puts them on the
    floor: on the fyp initial video the watch, mouse, cube and backpack all
    stood in such a hole (0-1% footprint overlap with any table piece).

    Seeds are instances with a horizontal slab (`slab_m` thick, at least
    `seed_min_fraction` of their points, top at furniture height) covering at
    least `seed_min_area_m2`. A plane is fitted to each seed's slab, and the
    voxel-map voxels within `plane_tol_m` of it that are connected to the seed
    (gaps up to `bridge_m` bridged) are added to that instance. Excluded:
    voxels of other instances, and XY cells with map points both above and
    below the plane (between 1.5 * `plane_tol_m` and `vertical_band_m`) -- walls
    and chair backs cross the plane there, a table top does not. The tolerance
    is wide because the fused top is thick: on the fyp initial video the table
    top spans 0.63-0.78 m, from depth that differs by a few cm between views. Largest seed first, so a bag's flat
    bottom on the table cannot claim the table top; a smaller seed in the plane
    of a grown one (within `plane_tol_m` at its centre) that touches it is an
    object standing on that table and is not grown (on Lap_remove_2 the iron
    grew 8692 -> 13657 voxels of table top otherwise). A seed with a compact
    part standing more than `split_height_m` above its plane is an object that
    absorbed the table around it, not just a table: that part (at least
    `split_min_voxels`) is split off as its own instance, keeping the seed's
    names, and only the rest becomes the table (see _split_raised_part()).
    Needs a gravity-aligned world (z up, floor at 0).
    """
    from scipy import ndimage

    all_idx = np.fromiter(view_dataset.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    all_pts = np.asarray(view_dataset.index_to_point(all_idx))
    owner = {}
    for k, obj in enumerate(objects):
        for ix in np.asarray(obj["indexes"]).tolist():
            owner.setdefault(ix, k)
    owner_of = np.array([owner.get(ix, -1) for ix in all_idx.tolist()])

    seeds = []
    for k, obj in enumerate(objects):
        pts = view_dataset.index_to_point(obj["indexes"])
        if len(pts) < 50:
            continue
        fraction, top = _densest_slab(pts[:, 2], slab_m)
        if fraction < seed_min_fraction or not min_top_m <= top <= max_top_m:
            continue
        slab = pts[(pts[:, 2] > top - slab_m) & (pts[:, 2] <= top)]
        area = len(np.unique(np.floor(slab[:, :2] / cell_m).astype(np.int64), axis=0)) * cell_m ** 2
        if area >= seed_min_area_m2:
            seeds.append((area, k, slab))

    claimed = np.zeros(len(all_idx), dtype=bool)
    origin = all_pts[:, :2].min(axis=0)
    cells = np.floor((all_pts[:, :2] - origin) / cell_m).astype(np.int64)
    shape = tuple(cells.max(axis=0) + 1)
    flat_cells = cells[:, 0] * shape[1] + cells[:, 1]
    grown_surfaces = []  # (plane coefficients, XY grid of the grown surface)
    for area, k, slab in sorted(seeds, key=lambda s: -s[0]):
        # z = a x + b y + c, refit once without the slab's outliers
        A = np.c_[slab[:, :2], np.ones(len(slab))]
        coef = np.linalg.lstsq(A, slab[:, 2], rcond=None)[0]
        keep = np.abs(A @ coef - slab[:, 2]) < plane_tol_m
        if keep.sum() >= 3:
            coef = np.linalg.lstsq(A[keep], slab[keep, 2], rcond=None)[0]
        centre = slab[:, :2].mean(axis=0)
        seed_cells = np.floor((slab[:, :2] - origin) / cell_m).astype(np.int64)
        seed_cells = seed_cells[np.all((seed_cells >= 0) & (seed_cells < shape), axis=1)]
        on_bigger = [g for c, g in grown_surfaces
                     if abs(np.r_[centre, 1.0] @ (coef - c)) < plane_tol_m
                     and ndimage.binary_dilation(g, np.ones((3, 3), bool))[seed_cells[:, 0], seed_cells[:, 1]].any()]
        if on_bigger:
            logger.info("Support surface seed '%s' (slab %.2f m2) lies in the plane of a larger table top it "
                        "touches: an object standing on it, not grown.",
                        objects[k].get("class_id") or objects[k].get("class_name") or k, area)
            continue
        split = _split_raised_part(objects, k, view_dataset, coef, split_height_m, split_min_voxels)
        if split is not None:
            owner_of[np.isin(all_idx, objects[split]["indexes"])] = split
        height = all_pts[:, 2] - (all_pts[:, :2] @ coef[:2] + coef[2])

        def filled_share(h):
            """Per XY cell, the share of the 2.5 cm height bins between
            vertical_gap and vertical_band_m (h already sign-flipped) that
            hold a map point."""
            gap = 1.5 * plane_tol_m
            n_bins = max(1, int((vertical_band_m - gap) / 0.025))
            m = (h > gap) & (h < vertical_band_m)
            pairs = np.unique(flat_cells[m] * n_bins + ((h[m] - gap) / 0.025).astype(np.int64).clip(0, n_bins - 1))
            counts = np.bincount(pairs // n_bins, minlength=shape[0] * shape[1])
            return (counts / n_bins).reshape(shape)

        # A wall or chair back fills the column both above and below the
        # plane; a table top, even with things standing on it, not below.
        vertical = (filled_share(height) >= 0.5) & (filled_share(-height) >= 0.5)

        on_plane = ((np.abs(height) < plane_tol_m) & ((owner_of == -1) | (owner_of == k)) & ~claimed
                    & ~vertical[cells[:, 0], cells[:, 1]])
        grid = np.zeros(shape, bool)
        grid[cells[on_plane, 0], cells[on_plane, 1]] = True
        bridge = max(1, int(round(bridge_m / cell_m)))
        labels, _ = ndimage.label(ndimage.binary_closing(grid, np.ones((bridge, bridge), bool)) | grid,
                                  structure=np.ones((3, 3), bool))
        seed_labels = np.unique(labels[seed_cells[:, 0], seed_cells[:, 1]])
        seed_labels = seed_labels[seed_labels > 0]
        grown = on_plane & np.isin(labels[cells[:, 0], cells[:, 1]], seed_labels)

        obj = objects[k]
        before = len(obj["indexes"])
        obj["indexes"] = np.unique(np.concatenate([np.asarray(obj["indexes"]), all_idx[grown]]))
        claimed |= grown
        surface_grid = np.zeros(shape, bool)
        surface_grid[seed_cells[:, 0], seed_cells[:, 1]] = True
        surface_grid[cells[grown, 0], cells[grown, 1]] = True
        grown_surfaces.append((coef, surface_grid))
        logger.info("Support surface '%s' (slab %.2f m2, plane z=%.2f m at its centre): %d -> %d voxels.",
                    obj.get("class_id") or obj.get("class_name") or k, area,
                    float(np.r_[slab[:, :2].mean(axis=0), 1.0] @ coef), before, len(obj["indexes"]))
    return objects


def merge_fragmented_instances(objects, view_dataset, containment: float = 0.5,
                               embedding_similarity: float = 0.75, point_distance_m: float = 0.05,
                               conflict_min_score: float = 0.3, max_rounds: int = 3):
    """Merges a smaller instance into a larger one when all three hold:

      * at least `containment` of the smaller one's points lie within
        `point_distance_m` of the larger one's points. Points, not boxes: a
        desk's bounding box contains everything standing on it, so a box test
        swallows the laptop, bottle and bag along with real fragments;
      * their running-average NOVIC embeddings have cosine >=
        `embedding_similarity`;
      * they don't both carry a confident (score >= `conflict_min_score`)
        track-end name that disagrees (see _names_conflict()).

    Merging is by voxel-index union, the running-average embedding is combined
    weighted by detection count (as DoVSG's merge_obj_to_obj() does), and the
    list fields fusion maintains (track_ids, track_names, image_name, mask_idx)
    are concatenated. Repeats until nothing merges, since a merge can expose a
    further one.
    """
    objects = list(objects)
    total_merged = 0

    for _ in range(max_rounds):
        if len(objects) < 2:
            break
        points = [view_dataset.index_to_point(o["indexes"]) for o in objects]
        boxes = [(p.min(axis=0), p.max(axis=0)) for p in points]
        trees: dict[int, cKDTree] = {}
        absorbed: dict[int, int] = {}

        order = sorted(range(len(objects)), key=lambda i: -len(points[i]))
        for a_pos, i in enumerate(order):
            if i in absorbed:
                continue
            for j in order[a_pos + 1:]:
                if j in absorbed:
                    continue
                if not _boxes_touch(boxes[i], boxes[j], point_distance_m):
                    continue
                if _cosine(objects[i]["clip_ft"], objects[j]["clip_ft"]) < embedding_similarity:
                    continue
                if _names_conflict(objects[i], objects[j], conflict_min_score):
                    continue
                if i not in trees:
                    trees[i] = cKDTree(points[i])
                if _point_containment(points[j], trees[i], point_distance_m) < containment:
                    continue
                absorbed[j] = i

        if not absorbed:
            break

        for j, i in absorbed.items():
            host, part = objects[i], objects[j]
            host["indexes"] = np.unique(np.concatenate([host["indexes"], part["indexes"]]))
            n_host, n_part = host.get("num_detections", 1), part.get("num_detections", 1)
            host["clip_ft"] = _weighted_mean_embedding(host["clip_ft"], n_host, part["clip_ft"], n_part)
            host["num_detections"] = n_host + n_part
            host["conf"] = max(host.get("conf", 0.0), part.get("conf", 0.0))
            for key in ("track_ids", "track_names", "image_name", "mask_idx", "mask", "xyxy"):
                if isinstance(host.get(key), list) and isinstance(part.get(key), list):
                    host[key] = host[key] + part[key]
        objects = [o for k, o in enumerate(objects) if k not in absorbed]
        total_merged += len(absorbed)

    if total_merged:
        logger.info("Merged %d fragmented instance(s); %d remain.", total_merged, len(objects))
    return objects


def suppress_nested_names(objects, view_dataset, embed_text=None, similarity: float = 0.85,
                          min_size_ratio: float = 1.5, on_top_tol_m: float = 0.15) -> int:
    """label_mapper.nested_name_suppression (stage 7, after naming): a parent named like an object lying on
    or inside it (a table whose crops are dominated by the laptop on it, named "laptop") takes NOVIC's next
    beam candidate that is not the child's name. Parent / child here is geometric, as the relations will be:
    the child is >= min_size_ratio times smaller, its footprint centre lies inside the parent's, and it either
    rests on the parent's top (bottom within on_top_tol_m of it) or lies >= 50% inside the parent's box.
    Names match when equal or (with embed_text) text cosine >= similarity. Returns the number renamed;
    class_id is reassigned for every object afterwards so ids stay unique."""
    from semantic_decoding.label_mapper.label_mapper import canonicalize
    if len(objects) < 2:
        return 0
    pts = [view_dataset.index_to_point(np.asarray(o["indexes"])) for o in objects]
    lo = [p.min(0) for p in pts]
    hi = [p.max(0) for p in pts]
    n = [len(p) for p in pts]
    cache = {}

    def same(a: str, b: str) -> bool:
        if a == b:
            return True
        if embed_text is None:
            return False
        for t in (a, b):
            if t not in cache:
                e = np.asarray(embed_text([f"an image of {t}"])[0], np.float32)
                cache[t] = e / (np.linalg.norm(e) + 1e-12)
        return float(cache[a] @ cache[b]) >= similarity

    renamed = 0
    for i, par in enumerate(objects):
        name = par.get("class_name", "unknown")
        if name == "unknown":
            continue
        kids = []
        for j, ch in enumerate(objects):
            if j == i or n[i] < min_size_ratio * n[j]:
                continue
            c = pts[j].mean(0)
            if not (lo[i][0] <= c[0] <= hi[i][0] and lo[i][1] <= c[1] <= hi[i][1]):
                continue
            on_top = abs(lo[j][2] - np.percentile(pts[i][:, 2], 90)) <= on_top_tol_m
            inside = np.all((pts[j] >= lo[i] - 0.02) & (pts[j] <= hi[i] + 0.02), axis=1).mean() >= 0.5
            if on_top or inside:
                kids.append(ch.get("class_name", "unknown"))
        clash = [k for k in kids if k != "unknown" and same(name, k)]
        if not clash:
            continue
        for label, _ in par.get("novic_top_k") or []:
            cand = canonicalize(str(label))
            if cand and cand != "unknown" and not any(same(cand, k) for k in kids if k != "unknown"):
                logger.info("Nested name: '%s' is also the name of an object on / in it (%s); renamed '%s'",
                            name, ", ".join(sorted(set(clash))), cand)
                par["renamed_from"], par["class_name"] = name, cand
                renamed += 1
                break
    counts: dict[str, int] = {}
    for o in objects:
        k = counts.get(o["class_name"], 0)
        counts[o["class_name"]] = k + 1
        o["class_id"] = f"{o['class_name']}_{k}"
    return renamed


def merge_same_name_neighbours(objects, view_dataset, gap_m: float = 0.03, similarity: float = 0.8,
                               containment: float = 0.2, max_rounds: int = 3):
    """label_mapper.same_name_merge (stage 7, after naming): two objects carrying the same (known) name that touch
    (a point of the smaller within gap_m of the larger), whose embeddings have cosine >= similarity, and where
    >= containment of the smaller one's points lie within gap_m of the larger, are one object split in two; the
    smaller is merged into the larger (voxel union, detection-weighted embedding, list fields concatenated), as in
    merge_fragmented_instances() but with the shared name allowing a lower overlap. Two separate objects of the same
    kind side by side (two cushions) touch along an edge only and stay apart. Returns (objects, number merged);
    class_id is reassigned afterwards."""
    objects = list(objects)
    merged = 0
    for _ in range(max_rounds):
        pts = [view_dataset.index_to_point(np.asarray(o["indexes"])) for o in objects]
        boxes = [(p.min(0), p.max(0)) if len(p) else (np.zeros(3), np.zeros(3)) for p in pts]
        order = sorted(range(len(objects)), key=lambda i: -len(pts[i]))
        absorbed: dict[int, int] = {}
        trees: dict[int, cKDTree] = {}
        for a, i in enumerate(order):
            name = objects[i].get("class_name", "unknown")
            if i in absorbed or name == "unknown" or not len(pts[i]):
                continue
            for j in order[a + 1:]:
                if j in absorbed or objects[j].get("class_name") != name or not len(pts[j]):
                    continue
                if not _boxes_touch(boxes[i], boxes[j], gap_m):
                    continue
                if _cosine(objects[i]["clip_ft"], objects[j]["clip_ft"]) < similarity:
                    continue
                if i not in trees:
                    trees[i] = cKDTree(pts[i])
                if _point_containment(pts[j], trees[i], gap_m) < containment:
                    continue
                absorbed[j] = i
        if not absorbed:
            break
        for j, i in absorbed.items():
            host, part = objects[i], objects[j]
            logger.info("Same-name merge: %s (%d voxels) into %s (%d voxels)", part.get("class_id"),
                        len(part["indexes"]), host.get("class_id"), len(host["indexes"]))
            host["indexes"] = np.unique(np.concatenate([host["indexes"], part["indexes"]]))
            n_h, n_p = host.get("num_detections", 1), part.get("num_detections", 1)
            host["clip_ft"] = _weighted_mean_embedding(host["clip_ft"], n_h, part["clip_ft"], n_p)
            host["num_detections"] = n_h + n_p
            for key in ("track_ids", "track_names", "image_name", "mask_idx", "mask", "xyxy"):
                if isinstance(host.get(key), list) and isinstance(part.get(key), list):
                    host[key] = host[key] + part[key]
        objects = [o for k, o in enumerate(objects) if k not in absorbed]
        merged += len(absorbed)
    counts: dict[str, int] = {}
    for o in objects:
        k = counts.get(o["class_name"], 0)
        counts[o["class_name"]] = k + 1
        o["class_id"] = f"{o['class_name']}_{k}"
    return objects, merged
