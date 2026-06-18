"""Demo for tracking Pauli branching and anti-commutation clusters."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from paulitools import GLOBAL_INTEGER, commutes, row_reduce, save_pauli_data, toString, toZX


@dataclass
class BranchingState:
    """State tracker for the branching demo."""

    k: int
    paulis: List[np.ndarray]
    commute_graph: Dict[int, Set[int]]
    anti_commute_graph: Dict[int, Set[int]]

    def __init__(self, k: int):
        self.k = k
        self.paulis = []
        self.commute_graph = defaultdict(set)
        self.anti_commute_graph = defaultdict(set)

    def add_pauli(self, pauli: np.ndarray) -> int:
        index = len(self.paulis)
        self.paulis.append(pauli)
        return index

    def record_commutation(self, idx_a: int, idx_b: int, commute: bool) -> None:
        if commute:
            self.commute_graph[idx_a].add(idx_b)
            self.commute_graph[idx_b].add(idx_a)
        else:
            self.anti_commute_graph[idx_a].add(idx_b)
            self.anti_commute_graph[idx_b].add(idx_a)

    def row_reduced_basis(self) -> np.ndarray:
        packed = np.zeros(len(self.paulis) + 1, dtype=GLOBAL_INTEGER)
        packed[0] = self.k
        for i, pauli in enumerate(self.paulis, start=1):
            packed[i] = pauli[1]
        return row_reduce(packed)

    def anti_commute_clusters(self) -> Dict[int, List[int]]:
        visited: Set[int] = set()
        clusters: Dict[int, List[int]] = {}

        def dfs(node: int, cluster_id: int) -> None:
            stack = [node]
            component = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(self.anti_commute_graph[current] - visited)
            clusters[cluster_id] = sorted(component)

        cluster_label = 0
        for node in range(len(self.paulis)):
            if node not in visited and self.anti_commute_graph[node]:
                dfs(node, cluster_label)
                cluster_label += 1
        return clusters

    def anti_commute_histogram(self) -> Counter:
        clusters = self.anti_commute_clusters()
        return Counter(len(nodes) for nodes in clusters.values())


PAULI_LETTERS = ["I", "X", "Y", "Z"]


def random_pauli_string(k: int) -> str:
    return "".join(random.choice(PAULI_LETTERS[1:]) for _ in range(k))


def store_pauli(path: Path, pauli: np.ndarray) -> None:
    zx_form = np.zeros(len(pauli), dtype=GLOBAL_INTEGER)
    zx_form[0] = pauli[0]
    zx_form[1] = pauli[1]
    save_pauli_data(path, zx_form, append=True)


def add_pauli_and_track(state: BranchingState, path: Path) -> Tuple[int, np.ndarray]:
    pauli_str = random_pauli_string(state.k)
    packed = toZX(pauli_str)
    store_pauli(path, packed)

    idx = state.add_pauli(packed)
    for existing_idx, existing_pauli in enumerate(state.paulis[:-1]):
        is_commuting = commutes(existing_pauli[1], packed[1], state.k)
        state.record_commutation(existing_idx, idx, is_commuting)
    return idx, packed


def demo_branching(k: int = 2, rounds: int = 5, store_path: str | Path = "pauli_branching.ptstore") -> None:
    random.seed(1234)
    np.random.seed(1234)

    path = Path(store_path)
    if path.exists():
        path.unlink()

    state = BranchingState(k)

    print(f"\nStarting Pauli branching demo with k={k}, rounds={rounds}\n")

    for step in range(rounds):
        idx, pauli = add_pauli_and_track(state, path)
        print(f"Step {step:02d}: added {toString(pauli)} (index {idx})")
        print("  Row-reduced basis:")
        reduced = state.row_reduced_basis()
        for row_idx in range(1, len(reduced)):
            print(f"    {toString(np.array([state.k, reduced[row_idx]]))}")

        clusters = state.anti_commute_clusters()
        if clusters:
            print("  Anti-commutation clusters:")
            for label, members in clusters.items():
                labels = ", ".join(str(m) for m in members)
                print(f"    Cluster {label}: [{labels}]")
            hist = state.anti_commute_histogram()
            stats = ", ".join(f"size {size}: {count}" for size, count in sorted(hist.items()))
            print(f"  Cluster size histogram -> {stats}")
        else:
            print("  No anti-commutation clusters yet.")
        print()

    print("Demo finished. Pauli operators stored in:", path.resolve())


def _parse_args(argv: List[str]) -> Tuple[int, int, Path]:
    parser = argparse.ArgumentParser(description="Track Pauli branching statistics.")
    parser.add_argument("k", nargs="?", type=int, default=2, help="Number of qubits per Pauli string")
    parser.add_argument("rounds", nargs="?", type=int, default=5, help="How many Pauli strings to generate")
    parser.add_argument(
        "store_path", nargs="?", type=Path, default=Path("pauli_branching.ptstore"), help="Output archive path"
    )
    args = parser.parse_args(argv)
    return args.k, args.rounds, args.store_path


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    k, rounds, store_path = _parse_args(argv)
    demo_branching(k=k, rounds=rounds, store_path=store_path)


if __name__ == "__main__":
    main()
