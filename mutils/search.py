"""Golden-section search for unimodal functions.

Use case in fuzzing experiments: σ-vs-elicitation curves are unimodal (0 → peak → 0,
with breakdown at high σ). Golden-section finds the peak in O(log) evaluations
instead of grid-searching, enabling cheap-probe σ identification.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["golden_section_max"]

PHI = (5**0.5 - 1) / 2  # ~0.618


def golden_section_max(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    n_iter: int = 10,
) -> tuple(float, float, list[tuple[float, float]], dict[float, Any]):
    """Maximize unimodal f(x) over [lo, hi] using golden-section search.

    Args:
        f: function to maximize. Assumed unimodal on [lo, hi].
        lo, hi: search bounds.
        n_iter: number of additional evaluations after the initial 2 (so 2+n_iter total).
    Returns:
        (x_star, f_star, trajectory) where trajectory is a list of (x, f(x)) in evaluation order.
    """
    others = {}
    a, b = lo, hi
    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    (fc, c_gens), (fd, d_gens) = f(c), f(d)
    others[c] = c_gens
    others[d] = d_gens
    trajectory = [(c, fc), (d, fd)]
    for _ in range(n_iter):
        if fc > fd:
            b = d
            d, fd = c, fc
            c = b - PHI * (b - a)
            fc, c_gens = f(c)
            others[c] = c_gens
            trajectory.append((c, fc))
        else:
            a = c
            c, fc = d, fd
            d = a + PHI * (b - a)
            fd, d_gens = f(d)
            others[d] = d_gens
            trajectory.append((d, fd))
    if fc > fd:
        return c, fc, trajectory, others
    return d, fd, trajectory, others
