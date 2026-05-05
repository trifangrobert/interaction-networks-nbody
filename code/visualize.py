import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simulator import compute_energy
from dataset import make_rollout_batch
from model import InteractionNetwork, rollout

NODE_DIM = 5
EDGE_DIM = 1


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    saved = ckpt.get('cfg') or ckpt.get('args')
    model = InteractionNetwork(
        NODE_DIM, EDGE_DIM,
        saved['edge_hidden_dim'], saved['node_hidden_dim'],
        saved['effect_dim'], saved['n_edge_layers'],
    )
    model.load_state_dict(ckpt['model_state'])
    return model, saved


def make_video(args):
    model, saved = load_model(args.checkpoint)
    dt = saved['dt']

    init_pos, init_vel, masses_t, gt_pos, gt_vel, masses_np = make_rollout_batch(
        n_bodies=args.n_bodies,
        n_rollout_steps=args.n_rollout_steps,
        dt=dt,
        seed=args.seed,
    )

    pred_pos, pred_vel = rollout(model, init_pos, init_vel, masses_t, args.n_rollout_steps, dt)
    pred_pos_np = pred_pos.numpy()
    pred_vel_np = pred_vel.numpy()

    energies = np.array([
        compute_energy(pred_pos_np[t], pred_vel_np[t], masses_np)
        for t in range(args.n_rollout_steps + 1)
    ])
    t_arr = np.arange(args.n_rollout_steps + 1) * dt

    colors = plt.cm.tab10(np.linspace(0, 1, args.n_bodies))

    fig, (ax_traj, ax_energy) = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectory axis limits
    all_pos = np.concatenate([gt_pos, pred_pos_np], axis=0)
    margin = 0.5
    ax_traj.set_xlim(all_pos[:, :, 0].min() - margin, all_pos[:, :, 0].max() + margin)
    ax_traj.set_ylim(all_pos[:, :, 1].min() - margin, all_pos[:, :, 1].max() + margin)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')

    # Legend: one entry for GT and one for Pred
    ax_traj.plot([], [], '-', color='gray', label='Ground truth')
    ax_traj.plot([], [], '--', color='gray', label='Predicted')
    ax_traj.legend(fontsize=8, loc='upper right')

    # Energy axis limits
    e_min, e_max = energies.min(), energies.max()
    e_pad = max((e_max - e_min) * 0.15, 0.1)
    ax_energy.set_xlim(0, t_arr[-1])
    ax_energy.set_ylim(e_min - e_pad, e_max + e_pad)
    ax_energy.axhline(energies[0], color='gray', linestyle='--', alpha=0.7, label='Initial energy')
    ax_energy.set_xlabel('time')
    ax_energy.set_ylabel('Energy')
    ax_energy.set_title('Energy conservation (predicted rollout)')

    gt_lines, pred_lines, gt_dots, pred_dots = [], [], [], []
    for i, c in enumerate(colors):
        gl, = ax_traj.plot([], [], '-',  color=c, alpha=0.9, lw=1.5)
        pl, = ax_traj.plot([], [], '--', color=c, alpha=0.9, lw=1.5)
        gd, = ax_traj.plot([], [], 'o',  color=c, ms=7)
        pd, = ax_traj.plot([], [], 'o',  color=c, ms=5, markerfacecolor='white', markeredgewidth=1.5)
        gt_lines.append(gl)
        pred_lines.append(pl)
        gt_dots.append(gd)
        pred_dots.append(pd)

    energy_line, = ax_energy.plot([], [], color='steelblue', lw=1.5, label='Total energy')
    ax_energy.legend(fontsize=8)

    title = ax_traj.set_title('')
    plt.tight_layout()

    all_artists = gt_lines + pred_lines + gt_dots + pred_dots + [energy_line, title]

    def init():
        for i in range(args.n_bodies):
            gt_lines[i].set_data([], [])
            pred_lines[i].set_data([], [])
            gt_dots[i].set_data([], [])
            pred_dots[i].set_data([], [])
        energy_line.set_data([], [])
        title.set_text('')
        return all_artists

    def update(frame):
        for i in range(args.n_bodies):
            gt_lines[i].set_data(gt_pos[:frame + 1, i, 0],      gt_pos[:frame + 1, i, 1])
            pred_lines[i].set_data(pred_pos_np[:frame + 1, i, 0], pred_pos_np[:frame + 1, i, 1])
            gt_dots[i].set_data([gt_pos[frame, i, 0]],       [gt_pos[frame, i, 1]])
            pred_dots[i].set_data([pred_pos_np[frame, i, 0]], [pred_pos_np[frame, i, 1]])
        energy_line.set_data(t_arr[:frame + 1], energies[:frame + 1])
        title.set_text(
            f'Trajectories ({args.n_bodies} bodies, trained on {saved["n_bodies"]})  |  '
            f't = {t_arr[frame]:.2f}'
        )
        return all_artists

    step = max(1, (args.n_rollout_steps + 1) // args.max_frames)
    frames = list(range(0, args.n_rollout_steps + 1, step))

    anim = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=1000 // args.fps,
    )

    if args.output.endswith('.gif'):
        anim.save(args.output, writer='pillow', fps=args.fps)
    else:
        anim.save(args.output, writer=animation.FFMpegWriter(fps=args.fps))

    print(f'Video saved: {args.output}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',      type=str, default='../runs/best_model.pt')
    p.add_argument('--n-bodies',        type=int, default=3)
    p.add_argument('--n-rollout-steps', type=int, default=1000)
    p.add_argument('--seed',            type=int, default=999)
    p.add_argument('--fps',             type=int, default=30)
    p.add_argument('--max-frames',      type=int, default=300)
    p.add_argument('--output',          type=str, default='../runs/rollout.mp4')
    args = p.parse_args()
    make_video(args)


if __name__ == '__main__':
    main()
