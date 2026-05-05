import argparse
import os
import shutil
import yaml
import time
import torch
import torch.nn as nn
import wandb
from datetime import datetime
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import NBodyDataset
from model import InteractionNetwork

NODE_DIM = 5  # pos_x, pos_y, vel_x, vel_y, mass
EDGE_DIM = 1  # trivial ones


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_graphs = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(pred, batch.y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

    return total_loss / total_graphs


def load_hparams(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_metrics(path: str, metrics: dict):
    with open(path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)


def make_exp_dir(base_dir: str, cfg: dict) -> str:
    name = (f"n{cfg['n_bodies']}_h{cfg['edge_hidden_dim']}_l{cfg['n_edge_layers']}_ef{cfg['effect_dim']}"
            f"_sims{cfg['n_train_sims']}_lr{cfg['lr']}_ep{cfg['epochs']}_seed{cfg.get('torch_seed', 0)}")
    exp_dir = os.path.join(base_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def train(cfg: dict, hparams_path: str):
    exp_dir = make_exp_dir(cfg['output_dir'], cfg)
    print(f'Experiment dir: {exp_dir}')

    shutil.copy(hparams_path, os.path.join(exp_dir, 'hparams.yml'))

    device = get_device()
    print(f'Device: {device}')

    torch.manual_seed(cfg.get('torch_seed', 0))
    wandb.init(project='nbody-gnn', config=cfg, dir=exp_dir)

    print('Generating data...')
    train_ds = NBodyDataset(cfg['n_train_sims'], cfg['n_bodies'], cfg['n_steps'], cfg['dt'], seed=42)
    val_ds   = NBodyDataset(cfg['n_val_sims'],   cfg['n_bodies'], cfg['n_steps'], cfg['dt'], seed=123)
    print(f'Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}')

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    model = InteractionNetwork(
        NODE_DIM, EDGE_DIM,
        cfg['edge_hidden_dim'], cfg['node_hidden_dim'],
        cfg['effect_dim'], cfg['n_edge_layers'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')
    wandb.summary['n_params'] = n_params

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.MSELoss()

    metrics = {
        'best_val_loss': float('inf'),
        'best_epoch': -1,
        'n_params': n_params,
    }

    t_start = time.time()

    for epoch in tqdm(range(1, cfg['epochs'] + 1), desc='Epochs'):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        tqdm.write(f'Epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f}')

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})

        if val_loss < metrics['best_val_loss']:
            metrics['best_val_loss'] = round(val_loss, 8)
            metrics['best_epoch'] = epoch
            torch.save(
                {'model_state': model.state_dict(), 'cfg': cfg},
                os.path.join(exp_dir, 'best_model.pt'),
            )

    metrics['training_time_seconds'] = round(time.time() - t_start, 1)
    wandb.summary['best_val_loss'] = metrics['best_val_loss']
    wandb.summary['best_epoch']    = metrics['best_epoch']
    wandb.finish()

    save_metrics(os.path.join(exp_dir, 'metrics.yml'), metrics)

    print(f'\nDone. Best val loss: {metrics["best_val_loss"]:.6f} at epoch {metrics["best_epoch"]}')
    print(f'Results saved to: {exp_dir}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hparams', type=str, default='../hparams/default.yml')
    args = p.parse_args()
    cfg = load_hparams(args.hparams)
    train(cfg, args.hparams)


if __name__ == '__main__':
    main()
