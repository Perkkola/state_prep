from collections import deque
from itertools import combinations
from math import inf


# ══════════════════════════════════════════════════════════════════
#  Part 1: Graph utilities & cluster finding (unchanged)
# ══════════════════════════════════════════════════════════════════

def bfs_distances(source: int, neighbors: dict[int, set[int]]) -> dict[int, int]:
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nb in neighbors.get(node, set()):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def all_pairs_distances(neighbors: dict[int, set[int]]) -> dict[int, dict[int, int]]:
    return {node: bfs_distances(node, neighbors) for node in neighbors}


def cluster_cost(nodes: list[int], dist: dict[int, dict[int, int]]) -> float:
    total = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = dist.get(nodes[i], {}).get(nodes[j], inf)
            if d == inf:
                return inf
            total += d
    return total


def order_cluster(
    cluster: list[int], dist: dict[int, dict[int, int]]
) -> list[int]:
    remaining = set(cluster)

    def total_dist(node):
        return sum(dist[node].get(other, inf) for other in cluster if other != node)

    center = min(remaining, key=lambda n: (total_dist(n), n))
    ordered = [center]
    remaining.remove(center)

    while remaining:
        def cost_to_ordered(node):
            return sum(dist[node].get(o, inf) for o in ordered)
        nxt = min(remaining, key=lambda n: (cost_to_ordered(n), n))
        ordered.append(nxt)
        remaining.remove(nxt)

    return ordered


def find_closest_cluster_exact(n, neighbors):
    dist = all_pairs_distances(neighbors)
    all_nodes = list(neighbors.keys())
    if n > len(all_nodes):
        raise ValueError(f"Requested cluster size {n} exceeds graph size {len(all_nodes)}")
    best_cluster, best_cost = None, inf
    for combo in combinations(all_nodes, n):
        cost = cluster_cost(list(combo), dist)
        if cost < best_cost:
            best_cost = cost
            best_cluster = list(combo)
    return order_cluster(best_cluster, dist), best_cost


def find_closest_cluster_greedy(n, neighbors):
    dist = all_pairs_distances(neighbors)
    all_nodes = list(neighbors.keys())
    if n > len(all_nodes):
        raise ValueError(f"Requested cluster size {n} exceeds graph size {len(all_nodes)}")
    if n == 1:
        return [all_nodes[0]], 0

    best_pair, best_pair_score = None, inf
    for u in all_nodes:
        for v in neighbors[u]:
            if u >= v:
                continue
            score = sum(dist[u].get(w, inf) + dist[v].get(w, inf) for w in all_nodes)
            if score < best_pair_score:
                best_pair_score = score
                best_pair = [u, v]

    cluster = list(best_pair)
    in_cluster = set(cluster)
    candidate_cost = {}
    for node in all_nodes:
        if node not in in_cluster:
            candidate_cost[node] = sum(dist[node].get(c, inf) for c in cluster)

    while len(cluster) < n:
        best_node = min(candidate_cost, key=candidate_cost.get)
        cluster.append(best_node)
        in_cluster.add(best_node)
        added_cost = candidate_cost.pop(best_node)
        for node in list(candidate_cost):
            candidate_cost[node] += dist[node].get(best_node, inf)

    return order_cluster(cluster, dist), cluster_cost(cluster, dist)


def find_closest_cluster(n, edges, neighbors, method="auto"):
    from math import comb
    num_nodes = len(neighbors)
    if method == "auto":
        method = "exact" if comb(num_nodes, n) <= 50_000 else "greedy"
    if method == "exact":
        return find_closest_cluster_exact(n, neighbors)
    else:
        return find_closest_cluster_greedy(n, neighbors)


# ══════════════════════════════════════════════════════════════════
#  Part 2: Cyclic Gray code enumeration
# ══════════════════════════════════════════════════════════════════

def enumerate_cyclic_gray_codes(num_bits: int) -> list[list[int]]:
    """
    Enumerate all cyclic Gray codes on num_bits bits via Hamiltonian
    cycle search on the hypercube.  Feasible for num_bits <= 4.
    """
    n = 2 ** num_bits
    results = []
    path = [0]
    visited = {0}

    def backtrack():
        if len(path) == n:
            last = path[-1]
            if last != 0 and (last & (last - 1)) == 0:
                results.append(list(path))
            return
        current = path[-1]
        for bit in range(num_bits):
            nxt = current ^ (1 << bit)
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                backtrack()
                path.pop()
                visited.remove(nxt)

    backtrack()
    return results


# ══════════════════════════════════════════════════════════════════
#  Part 3: Gray code selection to minimise long-distance CNOTs
# ══════════════════════════════════════════════════════════════════

def gray_code_cnot_cost(
    code: list[int],
    control_distances: list[float],
) -> float:
    """
    Total CNOT cost for a cyclic Gray code, where each CNOT's cost
    is the graph distance from the flipping control qubit to the target.

    Parameters
    ----------
    code               : cyclic Gray code as a list of 2^k integers
    control_distances  : control_distances[bit] = graph distance from
                         control qubit 'bit' to the target qubit
    """
    cost = 0.0
    size = len(code)
    for step in range(size):
        prev = code[step - 1]          # Python wraps -1 → last element
        curr = code[step]
        flipped_bit = (prev ^ curr).bit_length() - 1   # 0-indexed
        cost += control_distances[flipped_bit]
    return cost


def construct_permuted_brgc(
    num_controls: int,
    control_distances: list[float],
) -> list[int]:
    """
    Heuristic Gray code for large num_controls: take the standard
    binary reflected Gray code and permute the bit positions so that
    the most-frequently-flipped bit (bit 0 in standard BRGC) is
    assigned to the closest control qubit.

    In standard BRGC bit j flips 2^{k-1-j} times (bit 0 flips most,
    bit k-1 flips least = twice for the cyclic closure).
    """
    # Sort bit indices by distance, ascending → closest first
    sorted_bits = sorted(range(num_controls), key=lambda b: control_distances[b])

    size = 2 ** num_controls
    code = []
    for i in range(size):
        standard_gray = i ^ (i >> 1)
        permuted = 0
        for j in range(num_controls):
            if (standard_gray >> j) & 1:
                permuted |= 1 << sorted_bits[j]
        code.append(permuted)
    return code


def select_best_gray_code(
    num_controls: int,
    control_distances: list[float],
    exact_threshold: int = 4,
) -> tuple[list[int], float]:
    """
    Select the cyclic Gray code minimising total weighted CNOT distance.

    For num_controls <= exact_threshold, all cyclic Gray codes are
    enumerated and the optimum is returned.  Otherwise a heuristic
    (permuted BRGC) is used.

    Parameters
    ----------
    num_controls       : k, number of control qubits
    control_distances  : distance from each control bit to the target
    exact_threshold    : max k for exhaustive search (default 4)

    Returns
    -------
    (best_gray_code, best_cost)
    """
    if num_controls <= exact_threshold:
        codes = enumerate_cyclic_gray_codes(num_controls)
        best_code, best_cost = None, inf
        for code in codes:
            cost = gray_code_cnot_cost(code, control_distances)
            if cost < best_cost:
                best_cost = cost
                best_code = code
        return best_code, best_cost
    else:
        code = construct_permuted_brgc(num_controls, control_distances)
        cost = gray_code_cnot_cost(code, control_distances)
        return code, cost


# ══════════════════════════════════════════════════════════════════
#  Part 4: Build all multiplexer Gray codes for the cluster
# ══════════════════════════════════════════════════════════════════

def select_gray_codes_for_cluster(
    cluster: list[int],
    dist: dict[int, dict[int, int]],
) -> list[dict]:
    """
    For each multiplexer in the state-preparation cascade, select the
    best cyclic Gray code.

    The cluster is ordered [index 0, index 1, ..., index n-1] where
    index 0 is the center.  Multiplexers are applied from index n-1
    down to index 1:

      - Target qubit = cluster[t]           (t = n-1, n-2, ..., 1)
      - Control qubits = cluster[0..t-1]    (bit j ↔ cluster[j])

    Returns a list of dicts (one per multiplexer, ordered t = n-1 → 1):
      { "target_index": t,
        "target_node": ...,
        "num_controls": t,
        "control_nodes": [...],
        "control_distances": [...],
        "gray_code": [...],
        "cnot_cost": ... }
    """
    n = len(cluster)
    results = []

    for t in range(n - 1, 0, -1):
        target_node = cluster[t]
        control_nodes = [cluster[j] for j in range(t)]
        control_dists = [
            dist[cluster[j]].get(target_node, inf) for j in range(t)
        ]

        code, cost = select_best_gray_code(t, control_dists)
        results.append({
            "target_index": t,
            "target_node": target_node,
            "num_controls": t,
            "control_nodes": control_nodes,
            "control_distances": control_dists,
            "gray_code": code,
            "cnot_cost": cost,
        })

    return results


# ══════════════════════════════════════════════════════════════════
#  Part 5: Angle transformation  θ = M · α  (Eq. 3, general code)
# ══════════════════════════════════════════════════════════════════

def transform_angles(
    alphas: list[float],
    gray_code: list[int],
) -> list[float]:
    """
    Transform multiplexer angles α into physical rotation angles θ
    using Eq. (3) of Möttönen et al., generalised to an arbitrary
    cyclic Gray code.

        θ_i = (1 / 2^k) Σ_j (-1)^{popcount(j & G_i)} α_j

    where G_i is the i-th codeword of the chosen Gray code and j
    runs over standard binary integers 0 … 2^k − 1.

    Parameters
    ----------
    alphas    : list of 2^k multiplexer angles  [α_0, ..., α_{2^k − 1}]
    gray_code : cyclic Gray code as list of 2^k integers

    Returns
    -------
    thetas    : list of 2^k rotation angles     [θ_0, ..., θ_{2^k − 1}]
    """
    size = len(gray_code)
    k = size.bit_length() - 1
    inv = 1.0 / size

    thetas = []
    for i in range(size):
        g_i = gray_code[i]
        theta_i = 0.0
        for j in range(size):
            dot_parity = bin(j & g_i).count('1') & 1   # mod-2 dot product
            sign = 1 - 2 * dot_parity                  # (-1)^dot
            theta_i += sign * alphas[j]
        thetas.append(theta_i * inv)

    return thetas


def inverse_transform_angles(
    thetas: list[float],
    gray_code: list[int],
) -> list[float]:
    """
    Inverse of transform_angles:  α = M^{-1} · θ.

    Since M is a symmetric Hadamard-like matrix scaled by 2^{-k},
    its inverse is M itself scaled by 2^{k}, i.e.

        α_j = Σ_i (-1)^{popcount(j & G_i)} θ_i
    """
    size = len(gray_code)

    alphas = []
    for j in range(size):
        alpha_j = 0.0
        for i in range(size):
            g_i = gray_code[i]
            dot_parity = bin(j & g_i).count('1') & 1
            sign = 1 - 2 * dot_parity
            alpha_j += sign * thetas[i]
        alphas.append(alpha_j)

    return alphas


# ══════════════════════════════════════════════════════════════════
#  Part 6: Full gate sequence for a multiplexer (RZ + CX list)
# ══════════════════════════════════════════════════════════════════

def multiplexer_gate_sequence(
    alphas: list[float],
    gray_code: list[int],
    control_nodes: list[int],
    target_node: int,
) -> tuple[list[tuple], list[tuple[str, float]]]:
    """
    Produce the gate list for a single uniformly controlled rotation
    using the selected Gray code.

    Returns
    -------
    gates : list of operations
        ("RZ", target_node, theta)
        ("CX", control_node, target_node)
    code_angle_pairs : list of (gray_code_bitstring, theta)
        e.g. [("000", 0.1234), ("001", -0.0567), ...]
        One entry per step, pairing the Gray code word at that step
        with the corresponding rotation angle.
    """
    thetas = transform_angles(alphas, gray_code)
    size = len(gray_code)
    num_controls = size.bit_length() - 1
    gates = []
    code_angle_pairs = []

    for step in range(size):
        bitstring = format(gray_code[step], f'0{num_controls}b')
        code_angle_pairs.append((bitstring, thetas[step]))

        # Rotation
        gates.append(("RZ", target_node, thetas[step]))

        # CNOT: which bit flipped?
        prev = gray_code[step - 1]
        curr = gray_code[step]
        flipped_bit = (prev ^ curr).bit_length() - 1
        gates.append(("CX", control_nodes[flipped_bit], target_node))

    return gates, code_angle_pairs


# ══════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import math
    import json

    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    edges = fake_garnet
    neighbors: dict[int, set[int]] = {}
    for u, v in edges:
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    cluster_size = 5
    print(f"Graph: {len(neighbors)} nodes, {len(edges)} edges")
    print(f"Finding closest cluster of {cluster_size} nodes\n")

    cluster, cost = find_closest_cluster(cluster_size, edges, neighbors)
    dist = all_pairs_distances(neighbors)

    print(f"Cluster (total pairwise dist = {cost}):")
    for idx, node in enumerate(cluster):
        print(f"  index {idx}  →  node {node}")
    print()

    # ── Select Gray codes for each multiplexer ───────────────────
    mux_info = select_gray_codes_for_cluster(cluster, dist)

    print("=" * 60)
    print("Multiplexer Gray code selection")
    print("=" * 60)
    for info in mux_info:
        t = info["target_index"]
        k = info["num_controls"]
        code_bin = [format(c, f'0{k}b') for c in info["gray_code"]]
        print(f"\nTarget index {t} (node {info['target_node']}), "
              f"{k} controls {info['control_nodes']}")
        print(f"  Control distances to target: {info['control_distances']}")
        print(f"  Selected Gray code: {code_bin}")
        print(f"  Total CNOT distance cost:    {info['cnot_cost']}")

    # ── Demo angle transformation ────────────────────────────────
    print("\n" + "=" * 60)
    print("Angle transformation demo (target index 3, 3 controls)")
    print("=" * 60)

    # Pick the multiplexer with 3 controls
    info_3 = next(i for i in mux_info if i["num_controls"] == 3)
    code_3 = info_3["gray_code"]

    # Example multiplexer angles
    alphas = [0.1 * (i + 1) for i in range(8)]
    print(f"\nMultiplexer angles α: {alphas}")

    thetas = transform_angles(alphas, code_3)
    print(f"Rotation angles   θ: {[round(t, 6) for t in thetas]}")

    # Verify round-trip
    alphas_back = inverse_transform_angles(thetas, code_3)
    print(f"Recovered         α: {[round(a, 6) for a in alphas_back]}")
    err = max(abs(a - b) for a, b in zip(alphas, alphas_back))
    print(f"Round-trip error:    {err:.2e}")

    # ── Demo full gate sequence ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Gate sequence for the 3-control multiplexer")
    print("=" * 60)
    gates, code_angle_pairs = multiplexer_gate_sequence(
        alphas, code_3, info_3["control_nodes"], info_3["target_node"]
    )
    for g in gates:
        if g[0] == "RZ":
            print(f"  RZ(θ={g[2]:+.6f})  on qubit {g[1]}")
        else:
            print(f"  CX  control={g[1]} → target={g[2]}")

    print("\n" + "=" * 60)
    print("Gray code ↔ rotation angle pairs")
    print("=" * 60)
    for bitstring, theta in code_angle_pairs:
        print(f"  ('{bitstring}', {theta:+.6f})")