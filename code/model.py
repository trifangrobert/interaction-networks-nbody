import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Tuple


def _build_mlp(input_dim: int, hidden_dim: int, n_hidden: int, output_dim: int) -> nn.Sequential:
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_hidden_dim: int,
        node_hidden_dim: int,
        effect_dim: int,
        n_edge_layers: int = 1,
    ):
        super().__init__(aggr='add')
        self.edge_mlp = _build_mlp(2 * node_dim + edge_dim, edge_hidden_dim, n_edge_layers, effect_dim)
        self.node_mlp = _build_mlp(node_dim + effect_dim, node_hidden_dim, 1, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


def build_graph(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
) -> Data:
    n = positions.shape[0]
    idx = torch.arange(n)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    x = torch.cat([positions, velocities, masses.unsqueeze(-1)], dim=-1)
    edge_attr = torch.ones(edge_index.shape[1], 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@torch.no_grad()
def rollout(
    model: InteractionNetwork,
    init_pos: torch.Tensor,
    init_vel: torch.Tensor,
    masses: torch.Tensor,
    n_steps: int,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    pos, vel = init_pos.clone(), init_vel.clone()
    positions, velocities = [pos], [vel]

    for _ in range(n_steps):
        graph = build_graph(pos, vel, masses)
        acc = model(graph.x, graph.edge_index, graph.edge_attr)
        vel = vel + acc * dt
        pos = pos + vel * dt
        positions.append(pos.clone())
        velocities.append(vel.clone())

    return torch.stack(positions), torch.stack(velocities)
