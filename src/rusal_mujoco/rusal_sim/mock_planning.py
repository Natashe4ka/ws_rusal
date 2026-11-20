"""Deterministic joint-space RRT fallback; collision checks stay in the planning copy."""

import numpy as np
from .experiment import smoothstep


def joint_route(start, goal, limits, collision, rng, iterations=700, edge_resolution=0.04):
    def edge(a, b):
        return all(
            not collision(q)
            for q in np.linspace(a, b, max(2, int(np.max(np.abs(b - a)) / edge_resolution) + 2))
        )

    if edge(start, goal):
        return [start, goal]
    trees = [[start.copy()], [goal.copy()]]
    parents = [[-1], [-1]]
    swapped = False

    def extend(index, target):
        nodes = trees[index]
        nearest = int(np.argmin([np.linalg.norm(q - target) for q in nodes]))
        delta = target - nodes[nearest]
        length = np.linalg.norm(delta)
        candidate = nodes[nearest] + delta * min(1.0, 0.25 / max(length, 1e-12))
        if not edge(nodes[nearest], candidate):
            return None
        nodes.append(candidate)
        parents[index].append(nearest)
        return len(nodes) - 1

    def trace(index, node):
        result = []
        while node >= 0:
            result.append(trees[index][node])
            node = parents[index][node]
        return result[::-1]

    for _ in range(iterations):
        sample = rng.uniform(limits[:, 0], limits[:, 1])
        a = extend(0, sample)
        if a is not None:
            target = trees[0][a]
            for _ in range(100):
                b = extend(1, target)
                if b is None:
                    break
                if np.linalg.norm(trees[1][b] - target) < 1e-8:
                    route = trace(0, a) + trace(1, b)[::-1][1:]
                    if swapped:
                        route = route[::-1]
                    # Remove unnecessary bends with collision-checked shortcuts.
                    simplified = [route[0]]
                    i = 0
                    while i < len(route) - 1:
                        j = len(route) - 1
                        while j > i + 1 and not edge(route[i], route[j]):
                            j -= 1
                        simplified.append(route[j])
                        i = j
                    return simplified
        trees.reverse()
        parents.reverse()
        swapped = not swapped
    return None


def timed_route(route, speed_limits, minimum_duration, speed_fraction=0.4):
    times, qs, elapsed = [0.0], [route[0]], 0.0
    for a, b in zip(route, route[1:]):
        seconds = max(
            1.0,
            float(np.max(1.5 * np.abs(b - a) / (speed_fraction * speed_limits))),
            minimum_duration / (len(route) - 1),
        )
        for t in np.linspace(0, seconds, int(seconds / 0.04) + 2)[1:]:
            times.append(elapsed + t)
            qs.append(a + (b - a) * smoothstep(t / seconds))
        elapsed += seconds
    return np.array(times), np.array(qs)
