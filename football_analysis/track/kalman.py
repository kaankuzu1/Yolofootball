"""A small linear Kalman filter with a Rauch-Tung-Striebel smoother.

Written out rather than taken from ``filterpy`` because the ball filter needs
two things a generic library makes awkward: a time step that changes every
frame (timestamps come from the container, not a nominal fps), and a record of
every predicted and filtered covariance so the backward pass can run over any
sub-range of the clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ca_matrices", "KalmanStep", "rts_smooth"]


def ca_matrices(dt: float, jerk_density: float) -> tuple[np.ndarray, np.ndarray]:
    """Transition and process noise for a 2-D constant-acceleration model.

    State order is ``[x, y, vx, vy, ax, ay]``.  Noise is white jerk with the
    given spectral density, the standard discretisation.
    """
    f1 = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    q1 = jerk_density * np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ]
    )
    # Expand the per-axis blocks into the interleaved [x, y, vx, vy, ax, ay] order.
    F = np.kron(f1, np.eye(2))
    Q = np.kron(q1, np.eye(2))
    return F, Q


H_POS = np.hstack([np.eye(2), np.zeros((2, 4))])


@dataclass
class KalmanStep:
    """Everything the smoother needs about one frame of the forward pass."""

    x_pred: np.ndarray
    P_pred: np.ndarray
    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    """Transition that produced ``x_pred`` from the previous step's ``x``."""


def rts_smooth(steps: list[KalmanStep]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backward pass over a contiguous run of forward steps."""
    n = len(steps)
    xs = [s.x.copy() for s in steps]
    Ps = [s.P.copy() for s in steps]
    for k in range(n - 2, -1, -1):
        nxt = steps[k + 1]
        try:
            gain = steps[k].P @ nxt.F.T @ np.linalg.inv(nxt.P_pred)
        except np.linalg.LinAlgError:
            continue
        xs[k] = steps[k].x + gain @ (xs[k + 1] - nxt.x_pred)
        Ps[k] = steps[k].P + gain @ (Ps[k + 1] - nxt.P_pred) @ gain.T
    return xs, Ps
