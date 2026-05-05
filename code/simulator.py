import numpy as np
from typing import Optional, Tuple


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    softening: float = 0.1,
) -> np.ndarray:
    """
    Vectorized gravitational accelerations.
    diff[i,j] = r_i - r_j, so a[i] = -G * sum_j m_j * diff[i,j] / |diff[i,j]|^3
    Self-terms vanish because diff[i,i] = 0.
    """
    diff = positions[:, None, :] - positions[None, :, :]   # (N, N, 2)
    dist2 = np.sum(diff**2, axis=-1) + softening**2        # (N, N)
    dist3 = dist2**1.5
    return -G * np.einsum('ijn,ij->in', diff, masses[None, :] / dist3)


def compute_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    softening: float = 0.1,
) -> float:
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=-1))
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + softening**2)
    upper = np.triu(np.ones(len(masses), dtype=bool), k=1)
    pe = -G * np.sum(np.outer(masses, masses)[upper] / dist[upper])
    return ke + pe


def simulate(
    n_bodies: int,
    n_steps: int,
    dt: float,
    masses: Optional[np.ndarray] = None,
    init_pos: Optional[np.ndarray] = None,
    init_vel: Optional[np.ndarray] = None,
    G: float = 1.0,
    softening: float = 0.1,
    integrator: str = 'verlet',
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    N-body simulation. Returns (positions, velocities, accelerations, masses),
    each of shape (n_steps+1, n_bodies, 2) except masses which is (n_bodies,).
    """
    rng = np.random.default_rng(seed)

    if masses is None:
        masses = rng.uniform(0.5, 2.0, n_bodies)
    if init_pos is None:
        init_pos = rng.standard_normal((n_bodies, 2)) * 3.0
    if init_vel is None:
        init_vel = rng.standard_normal((n_bodies, 2)) * 0.5

    # Work in center-of-mass frame (zero total momentum)
    init_vel -= np.sum(masses[:, None] * init_vel, axis=0) / masses.sum()

    pos = np.empty((n_steps + 1, n_bodies, 2))
    vel = np.empty((n_steps + 1, n_bodies, 2))
    acc = np.empty((n_steps + 1, n_bodies, 2))

    pos[0], vel[0] = init_pos.copy(), init_vel.copy()
    acc[0] = compute_accelerations(pos[0], masses, G, softening)

    for t in range(n_steps):
        if integrator == 'verlet':
            pos[t + 1] = pos[t] + vel[t] * dt + 0.5 * acc[t] * dt**2
            acc[t + 1] = compute_accelerations(pos[t + 1], masses, G, softening)
            vel[t + 1] = vel[t] + 0.5 * (acc[t] + acc[t + 1]) * dt
        else:  # forward Euler
            pos[t + 1] = pos[t] + vel[t] * dt
            vel[t + 1] = vel[t] + acc[t] * dt
            acc[t + 1] = compute_accelerations(pos[t + 1], masses, G, softening)

    return pos, vel, acc, masses
