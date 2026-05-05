import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from simulator import compute_energy
from dataset import make_rollout_batch
from model import InteractionNetwork, rollout

NODE_DIM = 5
EDGE_DIM = 1


def constant_velocity_rollout(init_pos: torch.Tensor, init_vel: torch.Tensor, n_steps: int, dt: float) -> np.ndarray:
    pos = init_pos.clone()
    positions = [pos]
    for _ in range(n_steps):
        pos = pos + init_vel * dt
        positions.append(pos.clone())
    return torch.stack(positions).numpy()


def load_model(checkpoint_path: str) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    saved = ckpt.get('cfg') or ckpt.get('args')
    model = InteractionNetwork(
        NODE_DIM, EDGE_DIM,
        saved['edge_hidden_dim'], saved['node_hidden_dim'],
        saved['effect_dim'], saved['n_edge_layers'],
    )
    model.load_state_dict(ckpt['model_state'])
    return model, saved


def evaluate(args):
    os.makedirs(os.path.dirname(args.output_fig) or '.', exist_ok=True)

    model, saved_args = load_model(args.checkpoint)
    dt = saved_args['dt']

    init_pos, init_vel, masses_t, gt_pos, gt_vel, masses_np = make_rollout_batch(
        n_bodies=args.n_test_bodies,
        n_rollout_steps=args.n_rollout_steps,
        dt=dt,
        seed=args.seed,
    )

    pred_pos, pred_vel = rollout(model, init_pos, init_vel, masses_t, args.n_rollout_steps, dt)
    pred_pos_np = pred_pos.numpy()   # (T+1, N, 2)
    pred_vel_np = pred_vel.numpy()

    # Constant-velocity baseline
    cv_pos_np = constant_velocity_rollout(init_pos, init_vel, args.n_rollout_steps, dt)

    # Metrics
    mse_model    = np.mean((pred_pos_np - gt_pos)**2, axis=(1, 2))   # (T+1,)
    mse_baseline = np.mean((cv_pos_np   - gt_pos)**2, axis=(1, 2))

    energies = np.array([
        compute_energy(pred_pos_np[t], pred_vel_np[t], masses_np)
        for t in range(args.n_rollout_steps + 1)
    ])

    t_arr = np.arange(args.n_rollout_steps + 1) * dt

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, args.n_test_bodies))

    # 1. Trajectories
    ax = axes[0]
    for i, c in enumerate(colors):
        ax.plot(gt_pos[:, i, 0],       gt_pos[:, i, 1],       '-',  color=c, alpha=0.8, label=f'GT {i}')
        ax.plot(pred_pos_np[:, i, 0],  pred_pos_np[:, i, 1],  '--', color=c, alpha=0.8, label=f'Pred {i}')
        ax.plot(gt_pos[0, i, 0],       gt_pos[0, i, 1],       'o',  color=c, ms=6)
    ax.set_title(f'Trajectories ({args.n_test_bodies} bodies, trained on {saved_args["n_bodies"]})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, ncol=2)

    # 2. Rollout MSE over time (log scale)
    ax = axes[1]
    ax.semilogy(t_arr, mse_model,    label='InteractionNet')
    ax.semilogy(t_arr, mse_baseline, label='Constant velocity')
    ax.set_title('Rollout MSE over time')
    ax.set_xlabel('time')
    ax.set_ylabel('MSE (log scale)')
    ax.legend()

    # 3. Energy conservation
    ax = axes[2]
    ax.plot(t_arr, energies, label='Total energy')
    ax.axhline(energies[0], color='gray', linestyle='--', label='Initial energy')
    ax.set_title('Energy conservation (predicted rollout)')
    ax.set_xlabel('time')
    ax.set_ylabel('Energy')
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output_fig, dpi=150, bbox_inches='tight')
    print(f'Figure saved: {args.output_fig}')

    # Summary
    rel_drift = abs(energies[-1] - energies[0]) / (abs(energies[0]) + 1e-10)
    print(f'Final MSE — model: {mse_model[-1]:.4e}  |  baseline: {mse_baseline[-1]:.4e}')
    print(f'Energy drift over rollout: {rel_drift:.2%}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',      type=str,   default='../runs/best_model.pt')
    p.add_argument('--n-test-bodies',   type=int,   default=5)
    p.add_argument('--n-rollout-steps', type=int,   default=200)
    p.add_argument('--output-fig',      type=str,   default='../runs/evaluation.png')
    p.add_argument('--seed',            type=int,   default=999)
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
