import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from simulator import simulate
from model import build_graph


class NBodyDataset(Dataset):
    def __init__(
        self,
        n_sims: int,
        n_bodies: int,
        n_steps: int,
        dt: float,
        G: float = 1.0,
        softening: float = 0.1,
        integrator: str = 'verlet',
        seed: int = 0,
    ):
        self.data = []
        rng = np.random.default_rng(seed)

        for _ in tqdm(range(n_sims), desc=f'Generating {n_bodies}-body sims', leave=False):
            sim_seed = int(rng.integers(0, 2**31))
            pos, vel, acc, masses = simulate(
                n_bodies, n_steps, dt,
                G=G, softening=softening, integrator=integrator, seed=sim_seed,
            )
            masses_t = torch.tensor(masses, dtype=torch.float32)

            for t in range(n_steps + 1):
                graph = build_graph(
                    torch.tensor(pos[t], dtype=torch.float32),
                    torch.tensor(vel[t], dtype=torch.float32),
                    masses_t,
                )
                graph.y = torch.tensor(acc[t], dtype=torch.float32)
                self.data.append(graph)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def make_rollout_batch(
    n_bodies: int,
    n_rollout_steps: int,
    dt: float,
    G: float = 1.0,
    softening: float = 0.1,
    seed: int = 999,
):
    pos, vel, acc, masses = simulate(
        n_bodies, n_rollout_steps, dt,
        G=G, softening=softening, integrator='verlet', seed=seed,
    )
    return (
        torch.tensor(pos[0], dtype=torch.float32),
        torch.tensor(vel[0], dtype=torch.float32),
        torch.tensor(masses,  dtype=torch.float32),
        pos,
        vel,
        masses,
    )
