"""Array-backed graph storage for HNSW.

Layer 0 is a dense ``(capacity, 2*M)`` int32 matrix because every node lives
there.  Upper layers are sparse dictionaries: only ~1/M of nodes reach layer 1,
so allocating a full matrix for every upper layer would waste memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .exceptions import DuplicateIDError, InvariantError


@dataclass
class InvariantReport:
    ok: bool
    n_nodes: int
    entry_point: int
    max_level: int
    reachable: int
    unreachable: int
    max_degree0: int
    max_degree_upper: int
    layer_counts: dict[int, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def assert_ok(self, require_reachable: bool = True) -> None:
        if not self.ok:
            raise InvariantError("graph invariants failed: " + "; ".join(self.errors[:8]))
        if require_reachable and self.unreachable:
            raise InvariantError(f"{self.unreachable} node(s) unreachable from the entry point")


class GraphStore:
    """Storage for vectors, IDs, levels, and per-layer adjacency."""

    _EMPTY = np.empty(0, dtype=np.int32)

    def __init__(
        self,
        dim: int,
        M: int,
        capacity: int = 1024,
        dtype: np.dtype = np.float32,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if M < 2:
            raise ValueError("M must be at least 2")
        self.dim = int(dim)
        self.M = int(M)
        self.M0 = 2 * self.M
        self.dtype = np.dtype(dtype)
        self._capacity = max(int(capacity), 1)

        self.vectors = np.empty((self._capacity, self.dim), dtype=self.dtype)
        self.norms = np.empty(self._capacity, dtype=np.float32)
        self.ids = np.empty(self._capacity, dtype=np.int64)
        self.levels = np.empty(self._capacity, dtype=np.int32)
        self.neighbors0 = np.full((self._capacity, self.M0), -1, dtype=np.int32)
        self.degree0 = np.zeros(self._capacity, dtype=np.int16)
        self.in_degree0 = np.zeros(self._capacity, dtype=np.int16)

        self.upper_neighbors: list[dict[int, np.ndarray]] = []
        self.upper_degree: list[dict[int, int]] = []
        self.upper_in_degree: list[dict[int, int]] = []

        self.id_to_slot: dict[int, int] = {}
        self.size = 0
        self.entry_point = -1
        self.max_level = -1

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self.size

    def _reserve(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = self._capacity
        while new_capacity < required:
            new_capacity *= 2

        vectors = np.empty((new_capacity, self.dim), dtype=self.dtype)
        vectors[: self.size] = self.vectors[: self.size]
        norms = np.empty(new_capacity, dtype=np.float32)
        norms[: self.size] = self.norms[: self.size]
        ids = np.empty(new_capacity, dtype=np.int64)
        ids[: self.size] = self.ids[: self.size]
        levels = np.empty(new_capacity, dtype=np.int32)
        levels[: self.size] = self.levels[: self.size]
        neighbors0 = np.full((new_capacity, self.M0), -1, dtype=np.int32)
        neighbors0[: self.size] = self.neighbors0[: self.size]
        degree0 = np.zeros(new_capacity, dtype=np.int16)
        degree0[: self.size] = self.degree0[: self.size]
        in_degree0 = np.zeros(new_capacity, dtype=np.int16)
        in_degree0[: self.size] = self.in_degree0[: self.size]

        self.vectors = vectors
        self.norms = norms
        self.ids = ids
        self.levels = levels
        self.neighbors0 = neighbors0
        self.degree0 = degree0
        self.in_degree0 = in_degree0
        self._capacity = new_capacity

    def ensure_upper_layers(self, max_level: int) -> None:
        while len(self.upper_neighbors) < max_level:
            self.upper_neighbors.append({})
            self.upper_degree.append({})
            self.upper_in_degree.append({})

    def _ensure_upper_node(self, slot: int, layer: int) -> None:
        if layer <= 0:
            return
        self.ensure_upper_layers(layer)
        index = layer - 1
        if slot not in self.upper_neighbors[index]:
            self.upper_neighbors[index][slot] = np.full(self.M, -1, dtype=np.int32)
            self.upper_degree[index][slot] = 0

    def add_node(self, external_id: int, vector: np.ndarray, level: int) -> int:
        if level < 0:
            raise ValueError("level must be non-negative")
        vec = np.ascontiguousarray(vector, dtype=self.dtype)
        if vec.shape != (self.dim,):
            raise ValueError(f"expected vector shape ({self.dim},), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("vector contains NaN or infinity")
        external_id = int(external_id)
        if external_id in self.id_to_slot:
            raise DuplicateIDError(f"duplicate external id: {external_id}")

        self._reserve(self.size + 1)
        slot = self.size
        self.vectors[slot] = vec
        self.norms[slot] = float(np.dot(vec, vec))
        self.ids[slot] = external_id
        self.levels[slot] = level
        self.neighbors0[slot].fill(-1)
        self.degree0[slot] = 0
        self.in_degree0[slot] = 0
        self.id_to_slot[external_id] = slot
        self.size += 1

        if self.size == 1:
            self.entry_point = slot
            self.max_level = level
            self.ensure_upper_layers(level)
        elif level > self.max_level:
            self.max_level = level
            self.entry_point = slot
            self.ensure_upper_layers(level)
        return slot

    def get_vector(self, slot: int) -> np.ndarray:
        return self.vectors[slot]

    def get_id(self, slot: int) -> int:
        return int(self.ids[slot])

    def slot_of(self, external_id: int) -> int:
        return self.id_to_slot[int(external_id)]

    def get_neighbors(self, slot: int, layer: int) -> np.ndarray:
        if layer == 0:
            degree = int(self.degree0[slot])
            if degree == 0:
                return self._EMPTY
            return self.neighbors0[slot, :degree]
        if layer < 0:
            raise ValueError("layer must be non-negative")
        index = layer - 1
        if index >= len(self.upper_neighbors):
            return self._EMPTY
        arr = self.upper_neighbors[index].get(slot)
        if arr is None:
            return self._EMPTY
        degree = self.upper_degree[index].get(slot, 0)
        if degree == 0:
            return self._EMPTY
        return arr[:degree]

    def get_degree(self, slot: int, layer: int) -> int:
        if layer == 0:
            return int(self.degree0[slot])
        index = layer - 1
        if index >= len(self.upper_neighbors):
            return 0
        return int(self.upper_degree[index].get(slot, 0))

    def get_in_degree(self, slot: int, layer: int) -> int:
        if layer == 0:
            return int(self.in_degree0[slot])
        index = layer - 1
        if index >= len(self.upper_in_degree):
            return 0
        return int(self.upper_in_degree[index].get(slot, 0))

    def rebuild_in_degrees(self) -> None:
        """Recompute incoming-link counts from the adjacency arrays."""
        self.in_degree0.fill(0)
        for slot in range(self.size):
            for neighbor in self.neighbors0[slot, : int(self.degree0[slot])]:
                self.in_degree0[int(neighbor)] += 1
        self.upper_in_degree = [{} for _ in self.upper_neighbors]
        for layer_index, layer_map in enumerate(self.upper_neighbors, start=1):
            degrees = self.upper_degree[layer_index - 1]
            in_degrees = self.upper_in_degree[layer_index - 1]
            for slot, arr in layer_map.items():
                for neighbor in arr[: int(degrees.get(slot, 0))]:
                    neighbor = int(neighbor)
                    in_degrees[neighbor] = in_degrees.get(neighbor, 0) + 1

    def append_neighbor(self, slot: int, layer: int, neighbor: int) -> bool:
        """Append one link if capacity allows.  Returns False when full."""
        if slot == neighbor:
            return False
        if neighbor < 0 or neighbor >= self.size:
            raise IndexError(f"neighbor slot out of range: {neighbor}")
        degree = self.get_degree(slot, layer)
        capacity = self.M0 if layer == 0 else self.M
        if degree >= capacity:
            return False
        if layer == 0:
            self.neighbors0[slot, degree] = neighbor
            self.degree0[slot] = degree + 1
            self.in_degree0[neighbor] += 1
            return True
        self._ensure_upper_node(slot, layer)
        index = layer - 1
        self.upper_neighbors[index][slot][degree] = neighbor
        self.upper_degree[index][slot] = degree + 1
        self.upper_in_degree[index][neighbor] = self.upper_in_degree[index].get(neighbor, 0) + 1
        return True

    def set_neighbors(self, slot: int, layer: int, neighbors: list[int] | np.ndarray) -> None:
        arr = np.asarray(neighbors, dtype=np.int32)
        capacity = self.M0 if layer == 0 else self.M
        if arr.size > capacity:
            raise ValueError(f"{arr.size} neighbors exceed capacity {capacity} at layer {layer}")
        if arr.size and (arr.min() < 0 or arr.max() >= self.size):
            raise IndexError("neighbor slot out of range")
        if np.any(arr == slot):
            raise ValueError("self-loop is not allowed")
        if np.unique(arr).size != arr.size:
            raise ValueError("duplicate neighbors are not allowed")

        if layer == 0:
            old_degree = int(self.degree0[slot])
            for old in self.neighbors0[slot, :old_degree]:
                self.in_degree0[int(old)] -= 1
            self.neighbors0[slot].fill(-1)
            if arr.size:
                self.neighbors0[slot, : arr.size] = arr
            for neighbor in arr:
                self.in_degree0[int(neighbor)] += 1
            self.degree0[slot] = arr.size
            return
        self._ensure_upper_node(slot, layer)
        index = layer - 1
        upper = self.upper_neighbors[index][slot]
        old_degree = int(self.upper_degree[index][slot])
        for old in upper[:old_degree]:
            old = int(old)
            self.upper_in_degree[index][old] = self.upper_in_degree[index].get(old, 1) - 1
        upper.fill(-1)
        if arr.size:
            upper[: arr.size] = arr
        for neighbor in arr:
            neighbor = int(neighbor)
            self.upper_in_degree[index][neighbor] = self.upper_in_degree[index].get(neighbor, 0) + 1
        self.upper_degree[index][slot] = int(arr.size)

    def layer_counts(self) -> dict[int, int]:
        if self.max_level < 0:
            return {}
        return {layer: int(np.count_nonzero(self.levels[: self.size] >= layer)) for layer in range(self.max_level + 1)}

    def memory_bytes(self) -> int:
        total = (
            self.vectors.nbytes
            + self.norms.nbytes
            + self.ids.nbytes
            + self.levels.nbytes
            + self.neighbors0.nbytes
            + self.degree0.nbytes
            + self.in_degree0.nbytes
        )
        total += sum(arr.nbytes for layer in self.upper_neighbors for arr in layer.values())
        # Rough per-dict-entry overhead, useful for reporting but not exact.
        total += 64 * (len(self.upper_neighbors) + sum(len(layer) for layer in self.upper_neighbors))
        return int(total)

    def validate(self, reachability: bool = True) -> InvariantReport:
        errors: list[str] = []
        layer_counts = self.layer_counts()
        max_degree0 = 0
        max_degree_upper = 0

        if self.size == 0:
            ok = self.entry_point == -1 and self.max_level == -1
            if not ok:
                errors.append("empty store has an entry point or max_level")
            return InvariantReport(ok, 0, self.entry_point, self.max_level, 0, 0, 0, 0, layer_counts, errors)

        if not (0 <= self.entry_point < self.size):
            errors.append(f"entry_point out of range: {self.entry_point}")
        elif int(self.levels[self.entry_point]) != self.max_level:
            errors.append("entry_point does not have max_level")
        if self.max_level < 0:
            errors.append("negative max_level on non-empty store")
        if len(self.id_to_slot) != self.size:
            errors.append("id_to_slot size mismatch")
        if len(np.unique(self.ids[: self.size])) != self.size:
            errors.append("duplicate external ids")

        for slot in range(self.size):
            level = int(self.levels[slot])
            if level < 0:
                errors.append(f"node {slot} has negative level")
            degree = int(self.degree0[slot])
            max_degree0 = max(max_degree0, degree)
            if degree < 0 or degree > self.M0:
                errors.append(f"node {slot} degree0={degree} outside [0,{self.M0}]")
                continue
            row = self.neighbors0[slot, :degree]
            if row.size:
                if np.any(row < 0) or np.any(row >= self.size):
                    errors.append(f"node {slot} has out-of-range layer-0 neighbor")
                if np.any(row == slot):
                    errors.append(f"node {slot} has a self-loop at layer 0")
                if np.unique(row).size != row.size:
                    errors.append(f"node {slot} has duplicate layer-0 neighbors")
            if self.id_to_slot.get(int(self.ids[slot])) != slot:
                errors.append(f"node {slot} id map is inconsistent")

        for layer_index, layer_map in enumerate(self.upper_neighbors, start=1):
            degrees = self.upper_degree[layer_index - 1]
            for slot, arr in layer_map.items():
                if slot < 0 or slot >= self.size:
                    errors.append(f"upper layer {layer_index} has out-of-range node {slot}")
                    continue
                if int(self.levels[slot]) < layer_index:
                    errors.append(f"node {slot} stored at layer {layer_index} but has level {int(self.levels[slot])}")
                degree = int(degrees.get(slot, 0))
                max_degree_upper = max(max_degree_upper, degree)
                if degree < 0 or degree > self.M:
                    errors.append(f"node {slot} upper degree {degree} outside [0,{self.M}]")
                    continue
                if arr.shape != (self.M,):
                    errors.append(f"node {slot} upper array has shape {arr.shape}")
                    continue
                row = arr[:degree]
                if row.size:
                    if np.any(row < 0) or np.any(row >= self.size):
                        errors.append(f"node {slot} has out-of-range upper neighbor")
                    if np.any(row == slot):
                        errors.append(f"node {slot} has a self-loop at layer {layer_index}")
                    if np.unique(row).size != row.size:
                        errors.append(f"node {slot} has duplicate upper neighbors")

        actual_in0 = np.zeros(self.size, dtype=np.int32)
        for slot in range(self.size):
            for neighbor in self.neighbors0[slot, : int(self.degree0[slot])]:
                actual_in0[int(neighbor)] += 1
        if not np.array_equal(actual_in0, self.in_degree0[: self.size].astype(np.int32)):
            errors.append("layer-0 in-degree bookkeeping is inconsistent")

        for layer_index, layer_map in enumerate(self.upper_neighbors, start=1):
            actual = {}
            degrees = self.upper_degree[layer_index - 1]
            for slot, arr in layer_map.items():
                for neighbor in arr[: int(degrees.get(slot, 0))]:
                    neighbor = int(neighbor)
                    actual[neighbor] = actual.get(neighbor, 0) + 1
            expected = {
                node: count
                for node, count in self.upper_in_degree[layer_index - 1].items()
                if count
            }
            if actual != expected:
                errors.append(f"layer-{layer_index} in-degree bookkeeping is inconsistent")

        reachable = 0
        if reachability:
            seen = bytearray(self.size)
            if 0 <= self.entry_point < self.size:
                seen[self.entry_point] = 1
                queue = deque([self.entry_point])
                reachable = 1
                while queue:
                    node = queue.popleft()
                    for neighbor in self.neighbors0[node, : int(self.degree0[node])]:
                        nb = int(neighbor)
                        if not seen[nb]:
                            seen[nb] = 1
                            reachable += 1
                            queue.append(nb)
            else:
                reachable = 0

        ok = not errors
        return InvariantReport(
            ok=ok,
            n_nodes=self.size,
            entry_point=self.entry_point,
            max_level=self.max_level,
            reachable=reachable,
            unreachable=self.size - reachable if reachability else 0,
            max_degree0=max_degree0,
            max_degree_upper=max_degree_upper,
            layer_counts=layer_counts,
            errors=errors,
        )
