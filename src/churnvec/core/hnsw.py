"""From-scratch HNSW.

This is a clean, array-backed implementation of Malkov & Yashunin's HNSW:
multi-layer proximity graph, greedy descent, ef-bounded best-first search,
geometric level sampling, the neighbour-selection heuristic, and link pruning.

Week 1-5 scope: insert + search + persistence + invariants.  Deletion,
tiering, and compaction deliberately live outside this class for now.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from .distance import Metric, distance_batch, distance_one
from .exceptions import DimensionMismatchError, DuplicateIDError, EmptyIndexError
from .graph import GraphStore, InvariantReport


@dataclass(frozen=True)
class Neighbor:
    id: int
    distance: float


class HNSWIndex:
    """An in-memory HNSW index with external int64 IDs."""

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        metric: Metric = "l2",
        seed: int = 42,
        capacity: int = 1024,
        selection: Literal["heuristic", "simple"] = "heuristic",
        keep_pruned: bool = False,
        max_level_cap: int = 32,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if M < 2:
            raise ValueError("M must be at least 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be positive")
        if metric not in {"l2", "cosine", "ip"}:
            raise ValueError(f"unsupported metric: {metric}")
        if selection not in {"heuristic", "simple"}:
            raise ValueError(f"unsupported selection: {selection}")

        self.dim = int(dim)
        self.M = int(M)
        self.M0 = 2 * self.M
        self.ef_construction = int(ef_construction)
        self.metric: Metric = metric
        self.seed = int(seed)
        self.selection = selection
        self.keep_pruned = bool(keep_pruned)
        self.max_level_cap = int(max_level_cap)

        self._store = GraphStore(dim=self.dim, M=self.M, capacity=capacity)
        self._rng = np.random.default_rng(self.seed)
        self._visited = np.zeros(self._store.capacity, dtype=np.int32)
        self._visit_token = 0
        self._insert_count = 0
        self._distance_calls = 0
        self._visited_nodes = 0
        self._repair_count = 0

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._store.size

    @property
    def entry_point(self) -> int:
        return self._store.entry_point

    @property
    def max_level(self) -> int:
        return self._store.max_level

    @property
    def store(self) -> GraphStore:
        return self._store

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(size={self.size}, dim={self.dim}, M={self.M}, "
            f"ef_construction={self.ef_construction}, max_level={self.max_level})"
        )

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def _sample_level(self) -> int:
        # P(level >= l) = M^-l.  Clamp the rare pathological sample.
        u = 1.0 - float(self._rng.random())
        level = int(-math.log(u) / math.log(self.M))
        return min(level, self.max_level_cap)

    def add_item(self, external_id: int, vector: np.ndarray) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise DimensionMismatchError(f"expected vector shape ({self.dim},), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("vector contains NaN or infinity")
        if int(external_id) in self._store.id_to_slot:
            raise DuplicateIDError(f"duplicate external id: {external_id}")

        level = self._sample_level()

        if self._store.size == 0:
            self._store.add_node(int(external_id), vec, level)
            self._insert_count += 1
            self._sync_visited()
            return

        old_entry = self._store.entry_point
        old_max_level = self._store.max_level
        slot = self._store.add_node(int(external_id), vec, level)
        self._sync_visited()

        q = self._store.vectors[slot]
        q_norm = float(self._store.norms[slot])
        ep = old_entry

        # Descend through layers above the new node's level.
        for layer in range(old_max_level, level, -1):
            ep, _ = self._greedy_search(q, q_norm, ep, layer)

        # Search and connect on every layer where the new node exists.
        for layer in range(min(level, old_max_level), -1, -1):
            candidates = self._search_layer(q, q_norm, [ep], self.ef_construction, layer)
            selected = self._select_neighbors(candidates, self.M, exclude=slot)
            if not selected and candidates:
                # Defensive fallback; the heuristic with keep_pruned should
                # normally never return an empty list for non-empty input.
                selected = candidates[: self.M]
            self._store.set_neighbors(slot, layer, [s for _, s in selected])
            kept_reverse = False
            for _, neighbor in selected:
                kept_reverse = self._add_link(neighbor, slot, layer) or kept_reverse
            if selected and not kept_reverse:
                # Degenerate data (for example exact duplicate vectors) can
                # make every pruning decision drop the new node.  Force one
                # incoming link, trying several sources so repairs do not
                # repeatedly rotate the same neighbour out.
                repaired = any(
                    self._force_link(neighbor, slot, layer) for _, neighbor in selected
                )
                if not repaired:
                    self._force_link(selected[0][1], slot, layer, allow_unsafe=True)
            if selected:
                ep = selected[0][1]

        self._insert_count += 1

    def add_items(
        self,
        vectors: np.ndarray,
        ids: Sequence[int] | np.ndarray | None = None,
        verbose: bool = False,
        progress_every: int = 10000,
    ) -> None:
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise DimensionMismatchError(f"expected (n, {self.dim}) vectors, got {data.shape}")
        if ids is None:
            ids = np.arange(self.size, self.size + data.shape[0], dtype=np.int64)
        id_array = np.asarray(ids, dtype=np.int64)
        if id_array.shape != (data.shape[0],):
            raise ValueError("ids length must match the number of vectors")
        for i, (external_id, vector) in enumerate(zip(id_array, data)):
            self.add_item(int(external_id), vector)
            if verbose and progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"  inserted {i + 1:,}/{data.shape[0]:,}", flush=True)
        if self.size > 1:
            self.repair_connectivity()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _sync_visited(self) -> None:
        if self._visited.size < self._store.capacity:
            new = np.zeros(self._store.capacity, dtype=np.int32)
            new[: self._visited.size] = self._visited
            self._visited = new

    def _next_token(self) -> int:
        self._visit_token += 1
        if self._visit_token >= 2_000_000_000:
            self._visited.fill(0)
            self._visit_token = 1
        return self._visit_token

    def _distance_to_slot(self, q: np.ndarray, q_norm: float, slot: int) -> float:
        self._distance_calls += 1
        if self.metric == "l2":
            d2 = float(self._store.norms[slot]) - 2.0 * float(np.dot(q, self._store.vectors[slot])) + q_norm
            return max(d2, 0.0)
        return distance_one(
            q,
            self._store.vectors[slot],
            self.metric,
            a_norm=q_norm,
            b_norm=float(self._store.norms[slot]),
        )

    def _distances_to_slots(self, q: np.ndarray, q_norm: float, slots: np.ndarray) -> np.ndarray:
        if slots.size == 0:
            return np.empty(0, dtype=np.float32)
        self._distance_calls += int(slots.size)
        return distance_batch(
            self._store.vectors[slots],
            q,
            self.metric,
            x_norms=self._store.norms[slots],
            q_norm=q_norm,
        )

    def _distances_between_slots(self, slot: int, others: Sequence[int]) -> np.ndarray:
        if not others:
            return np.empty(0, dtype=np.float32)
        indices = np.asarray(others, dtype=np.int32)
        if self.metric == "l2":
            dots = self._store.vectors[indices] @ self._store.vectors[slot]
            d2 = self._store.norms[indices] + self._store.norms[slot] - 2.0 * dots
            return np.maximum(d2, 0.0).astype(np.float32)
        return distance_batch(
            self._store.vectors[indices],
            self._store.vectors[slot],
            self.metric,
            x_norms=self._store.norms[indices],
            q_norm=float(self._store.norms[slot]),
        )

    def _greedy_search(self, q: np.ndarray, q_norm: float, ep: int, layer: int) -> tuple[int, float]:
        best = ep
        best_dist = self._distance_to_slot(q, q_norm, ep)
        while True:
            neighbors = self._store.get_neighbors(best, layer)
            if neighbors.size == 0:
                break
            dists = self._distances_to_slots(q, q_norm, neighbors)
            local = int(np.argmin(dists))
            local_dist = float(dists[local])
            if local_dist < best_dist:
                best = int(neighbors[local])
                best_dist = local_dist
            else:
                break
        return best, best_dist

    def _search_layer(
        self,
        q: np.ndarray,
        q_norm: float,
        entries: Iterable[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        if ef <= 0:
            return []
        entry_list = list(dict.fromkeys(int(e) for e in entries))
        if not entry_list:
            return []

        token = self._next_token()
        visited = self._visited
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []  # max-heap via negative distance

        for entry in entry_list:
            if visited[entry] == token:
                continue
            visited[entry] = token
            d = self._distance_to_slot(q, q_norm, entry)
            heapq.heappush(candidates, (d, entry))
            heapq.heappush(results, (-d, entry))
            self._visited_nodes += 1

        while candidates:
            dist, node = heapq.heappop(candidates)
            if len(results) >= ef and dist > -results[0][0]:
                break
            neighbors = self._store.get_neighbors(node, layer)
            if neighbors.size == 0:
                continue
            mask = visited[neighbors] != token
            if not np.any(mask):
                continue
            fresh = neighbors[mask]
            visited[fresh] = token
            dists = self._distances_to_slots(q, q_norm, fresh)
            self._visited_nodes += int(fresh.size)
            farthest = -results[0][0] if len(results) >= ef else math.inf
            for nb, dn in zip(fresh.tolist(), dists.tolist()):
                if len(results) < ef or dn < farthest:
                    heapq.heappush(candidates, (dn, nb))
                    heapq.heappush(results, (-dn, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
                    if len(results) >= ef:
                        farthest = -results[0][0]

        return sorted((-neg_dist, node) for neg_dist, node in results)

    def _select_neighbors(
        self,
        candidates: Sequence[tuple[float, int]],
        m: int,
        exclude: int | None = None,
        layer: int | None = None,
    ) -> list[tuple[float, int]]:
        if m <= 0:
            return []
        ordered = sorted(candidates)
        if self.selection == "simple":
            return [(d, s) for d, s in ordered if s != exclude][:m]

        selected: list[tuple[float, int]] = []
        rejected: list[tuple[float, int]] = []
        for dist, slot in ordered:
            if slot == exclude:
                continue
            if len(selected) >= m:
                break
            if not selected:
                selected.append((dist, slot))
                continue
            selected_slots = [s for _, s in selected]
            between = self._distances_between_slots(slot, selected_slots)
            # HNSW diversity rule: keep slot only if it is not closer to an
            # already-selected neighbour than it is to the query.
            if np.all(between >= dist):
                selected.append((dist, slot))
            else:
                rejected.append((dist, slot))

        if self.keep_pruned:
            for item in rejected:
                if len(selected) >= m:
                    break
                selected.append(item)
        return selected

    def _add_link(self, src: int, dst: int, layer: int) -> bool:
        """Add ``dst`` to ``src``'s neighbour list, pruning if necessary.

        Returns True when ``dst`` is present in the final list.
        """
        if src == dst:
            return False
        current = self._store.get_neighbors(src, layer)
        if current.size and np.any(current == dst):
            return True
        capacity = self.M0 if layer == 0 else self.M
        if current.size < capacity:
            self._store.append_neighbor(src, layer, dst)
            return True

        # The neighbour list is full: add dst temporarily and prune with the
        # same diversity heuristic, using src as the query.
        current_dists = self._distances_to_slots(
            self._store.vectors[src], float(self._store.norms[src]), current
        )
        dst_dist = self._distance_to_slot(
            self._store.vectors[src], float(self._store.norms[src]), dst
        )
        candidates = [(float(d), int(s)) for d, s in zip(current_dists, current)]
        candidates.append((dst_dist, dst))
        selected = self._select_neighbors(candidates, capacity, exclude=src, layer=layer)
        selected_slots = [s for _, s in selected]
        if dst not in selected_slots:
            # Do not rewrite an already-valid neighbour list just because the
            # heuristic rejected the new node.
            return False

        self._store.set_neighbors(src, layer, selected_slots)
        return True

    def _force_link(
        self, src: int, dst: int, layer: int, allow_unsafe: bool = False
    ) -> bool:
        """Guarantee one incoming edge for ``dst`` without creating a self-loop.

        Returns False when every existing neighbour would lose its last
        incoming edge, unless ``allow_unsafe`` is set.
        """
        if src == dst:
            return False
        current = self._store.get_neighbors(src, layer)
        if current.size and np.any(current == dst):
            return True
        capacity = self.M0 if layer == 0 else self.M
        if current.size < capacity:
            self._store.append_neighbor(src, layer, dst)
            return True

        # Prefer dropping a neighbour that remains reachable through another
        # incoming edge.
        order = self._distances_to_slots(
            self._store.vectors[src], float(self._store.norms[src]), current
        )
        drop = None
        for index in np.argsort(order)[::-1]:
            candidate = int(current[index])
            if self._store.get_in_degree(candidate, layer) > 1:
                drop = candidate
                break
        if drop is None:
            if not allow_unsafe:
                return False
            drop = int(current[int(np.argmax(order))])
        kept = [int(s) for s in current if int(s) != drop]
        kept.append(dst)
        self._store.set_neighbors(src, layer, kept[:capacity])
        return True

    def _search_slots(self, query: np.ndarray, k: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise EmptyIndexError("cannot search an empty index")
        if k <= 0:
            raise ValueError("k must be positive")
        q = np.ascontiguousarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise DimensionMismatchError(f"expected query shape ({self.dim},), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("query contains NaN or infinity")
        ef = max(int(ef), int(k))

        q_norm = float(np.dot(q, q))
        ep = self._store.entry_point
        for layer in range(self._store.max_level, 0, -1):
            ep, _ = self._greedy_search(q, q_norm, ep, layer)
        found = self._search_layer(q, q_norm, [ep], ef, 0)
        top = found[:k]
        slots = np.asarray([slot for _, slot in top], dtype=np.int32)
        distances = np.asarray([dist for dist, _ in top], dtype=np.float32)
        return slots, distances

    def search(self, query: np.ndarray, k: int = 10, ef: int = 64) -> list[Neighbor]:
        slots, distances = self._search_slots(query, k, ef)
        return [
            Neighbor(id=int(self._store.ids[slot]), distance=float(dist))
            for slot, dist in zip(slots, distances)
        ]

    def search_ids(self, query: np.ndarray, k: int = 10, ef: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Search and return ``(external_ids, distances)`` as arrays."""
        slots, distances = self._search_slots(query, k, ef)
        return self._store.ids[slots].copy(), distances

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        store = self._store
        meta = {
            "format": "churnvec-hnsw-v1",
            "dim": self.dim,
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "seed": self.seed,
            "selection": self.selection,
            "keep_pruned": self.keep_pruned,
            "max_level_cap": self.max_level_cap,
            "size": store.size,
            "entry_point": store.entry_point,
            "max_level": store.max_level,
            "rng_state": self._rng.bit_generator.state,
        }
        payload: dict[str, np.ndarray] = {
            "meta": np.asarray(json.dumps(meta)),
            "vectors": store.vectors[: store.size],
            "norms": store.norms[: store.size],
            "ids": store.ids[: store.size],
            "levels": store.levels[: store.size],
            "neighbors0": store.neighbors0[: store.size],
            "degree0": store.degree0[: store.size],
        }
        for layer in range(1, store.max_level + 1):
            nodes = sorted(store.upper_neighbors[layer - 1].keys())
            payload[f"upper_{layer}_nodes"] = np.asarray(nodes, dtype=np.int32)
            if nodes:
                payload[f"upper_{layer}_neighbors"] = np.stack(
                    [store.upper_neighbors[layer - 1][node].copy() for node in nodes]
                ).astype(np.int32)
                payload[f"upper_{layer}_degrees"] = np.asarray(
                    [store.upper_degree[layer - 1].get(node, 0) for node in nodes],
                    dtype=np.int32,
                )
            else:
                payload[f"upper_{layer}_neighbors"] = np.empty((0, self.M), dtype=np.int32)
                payload[f"upper_{layer}_degrees"] = np.empty(0, dtype=np.int32)
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HNSWIndex":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            size = int(meta["size"])
            index = cls(
                dim=int(meta["dim"]),
                M=int(meta["M"]),
                ef_construction=int(meta["ef_construction"]),
                metric=meta["metric"],
                seed=int(meta["seed"]),
                capacity=max(size, 1),
                selection=meta.get("selection", "heuristic"),
                keep_pruned=bool(meta.get("keep_pruned", False)),
                max_level_cap=int(meta.get("max_level_cap", 32)),
            )
            if size == 0:
                return index
            store = index._store
            store.vectors = data["vectors"].astype(np.float32)
            store.norms = data["norms"].astype(np.float32)
            store.ids = data["ids"].astype(np.int64)
            store.levels = data["levels"].astype(np.int32)
            store.neighbors0 = data["neighbors0"].astype(np.int32)
            store.degree0 = data["degree0"].astype(np.int16)
            store.size = size
            store.entry_point = int(meta["entry_point"])
            store.max_level = int(meta["max_level"])
            store.id_to_slot = {int(external_id): slot for slot, external_id in enumerate(store.ids)}
            store.upper_neighbors = [{} for _ in range(store.max_level)]
            store.upper_degree = [{} for _ in range(store.max_level)]
            for layer in range(1, store.max_level + 1):
                nodes = data[f"upper_{layer}_nodes"].astype(np.int32)
                neighbors = data[f"upper_{layer}_neighbors"].astype(np.int32)
                degrees = data[f"upper_{layer}_degrees"].astype(np.int32)
                for node, arr, degree in zip(nodes.tolist(), neighbors, degrees.tolist()):
                    store.upper_neighbors[layer - 1][int(node)] = arr.copy()
                    store.upper_degree[layer - 1][int(node)] = int(degree)
            store.rebuild_in_degrees()
            index._visited = np.zeros(store.capacity, dtype=np.int32)
            if "rng_state" in meta:
                index._rng.bit_generator.state = meta["rng_state"]
            return index

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------
    def repair_connectivity(self) -> int:
        """Make layer 0 reachable from the entry point.

        Healthy HNSW graphs need no repair and this method returns 0 without
        touching them.  Degenerate data (notably large sets of exactly equal
        vectors) can otherwise leave a node without any incoming layer-0 link.
        The repair threads a deterministic chain through every node, which
        guarantees reachability at the cost of one layer-0 link per node.  It
        is intentionally a last resort and is not part of normal search.
        """
        if self.size <= 1:
            return 0
        report = self.check_invariants(reachability=True)
        if report.unreachable == 0:
            return 0

        entry = self._store.entry_point
        order = [entry] + [slot for slot in range(self.size) if slot != entry]
        repaired = 0
        for src, node in zip(order, order[1:]):
            current = self._store.get_neighbors(src, 0)
            if current.size and np.any(current == node):
                continue
            if current.size < self._store.M0:
                self._store.append_neighbor(src, 0, node)
            else:
                dists = self._distances_to_slots(
                    self._store.vectors[src], float(self._store.norms[src]), current
                )
                drop = int(current[int(np.argmax(dists))])
                kept = [int(s) for s in current if int(s) != drop]
                kept.append(node)
                self._store.set_neighbors(src, 0, kept[: self._store.M0])
            repaired += 1
        self._repair_count += repaired
        return repaired

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def check_invariants(self, reachability: bool = True) -> InvariantReport:
        return self._store.validate(reachability=reachability)

    def stats(self) -> dict:
        report = self.check_invariants(reachability=True)
        return {
            "size": self.size,
            "dim": self.dim,
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "selection": self.selection,
            "keep_pruned": self.keep_pruned,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "layer_counts": report.layer_counts,
            "max_degree0": report.max_degree0,
            "max_degree_upper": report.max_degree_upper,
            "reachable": report.reachable,
            "unreachable": report.unreachable,
            "memory_bytes": self._store.memory_bytes(),
            "distance_calls": self._distance_calls,
            "visited_nodes": self._visited_nodes,
            "insert_count": self._insert_count,
            "repair_links": self._repair_count,
        }
