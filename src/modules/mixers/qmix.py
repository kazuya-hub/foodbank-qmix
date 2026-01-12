import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from my_utils import array_logger
from my_utils import shared


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(
                self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, wandb_log=False, t_env=None):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim) # shape: (バッチサイズ * (最大)エピソード長, 状態の次元数)
        agent_qs = agent_qs.view(-1, 1, self.n_agents) # shape: (バッチサイズ * (最大)エピソード長, 1, エージェント数)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim) # shape: (バッチサイズ * (最大)エピソード長, エージェント数, embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim) # shape: (バッチサイズ * (最大)エピソード長, 1, embed_dim)

        hidden_p = th.bmm(agent_qs, w1) + b1
        hidden = F.elu(hidden_p) # shape: (バッチサイズ * (最大)エピソード長, 1, embed_dim)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1) # shape: (バッチサイズ * (最大)エピソード長, embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1) # shape: (バッチサイズ * (最大)エピソード長, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v # shape: (バッチサイズ * (最大)エピソード長, 1, 1)
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # shape: (バッチサイズ, (最大)エピソード長, 1)

        if wandb_log:
            # fig, ax = plt.subplots()
            # data = w1.view(bs, -1, self.n_agents * self.embed_dim)[0].detach().numpy()
            # sns.heatmap(
            #     data,
            #     cmap="Reds",
            #     vmin=0,
            #     ax=ax,
            # )
            # ax.set_ylabel("Timestep")
            # ax.set_xticks(range(0, self.n_agents * self.embed_dim, self.embed_dim))
            # for x in range(0, self.n_agents * self.embed_dim, self.embed_dim):
            #     ax.axvline(x, color='black', linewidth=0.5)
            # wandb.log({"w1_n_agents>embed_dim": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots()
            data = w1.permute(0, 2, 1).reshape(bs, -1, self.n_agents * self.embed_dim)[0].detach().numpy()
            columns = ["weight_{}".format(i) for i in range(self.n_agents * self.embed_dim)]
            wandb.log({"w1_table_embed_dim>n_agents": wandb.Table(data=data, columns=columns)}, step=t_env)
            # sns.heatmap(
            #     data,
            #     cmap="Reds",
            #     vmin=0,
            #     ax=ax,
            # )
            # ax.set_ylabel("Timestep")
            # ax.set_xticks(range(0, self.n_agents * self.embed_dim, self.n_agents))
            # for x in range(0, self.n_agents * self.embed_dim, self.n_agents):
            #     ax.axvline(x, color='black', linewidth=0.5)
            # wandb.log({"w1_embed_dim>n_agents": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots()
            data = b1.view(bs, -1, self.embed_dim)[0].detach().numpy()
            columns = ["embed_node_{}".format(i) for i in range(self.embed_dim)]
            wandb.log({"b1_table": wandb.Table(data=data, columns=columns)}, step=t_env)
            # sns.heatmap(
            #     data,
            #     cmap="coolwarm",
            #     center=0,
            #     ax=ax,
            # )
            # ax.set_ylabel("Timestep")
            # wandb.log({"b1": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots()
            data = hidden_p.view(bs, -1, self.embed_dim)[0].detach().numpy()
            columns = ["embed_node_{}".format(i) for i in range(self.embed_dim)]
            wandb.log({"hidden_p_table": wandb.Table(data=data, columns=columns)}, step=t_env)
            # sns.heatmap(
            #     data,
            #     cmap="coolwarm",
            #     center=0,
            #     ax=ax,
            # )
            # ax.set_ylabel("Timestep")
            # wandb.log({"hidden_p": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots()
            data = hidden.view(bs, -1, self.embed_dim)[0].detach().numpy()
            columns = ["embed_node_{}".format(i) for i in range(self.embed_dim)]
            wandb.log({"hidden_table": wandb.Table(data=data, columns=columns)}, step=t_env)
            # sns.heatmap(
            #     data,
            #     cmap="coolwarm",
            #     center=0,
            #     ax=ax,
            # )
            # ax.set_ylabel("Timestep")
            # wandb.log({"hidden": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots()
            data = w_final.view(bs, -1, self.embed_dim)[0].detach().numpy()
            columns = ["embed_node_{}".format(i) for i in range(self.embed_dim)]
            wandb.log({"w_final_table": wandb.Table(data=data, columns=columns)}, step=t_env)
            # sns.heatmap(
            #     data,
            #     cmap="Reds",
            #     vmin=0,
            #     ax=ax,
            # )
            # ax.set_ylabel("Timestep")
            # wandb.log({"w_final": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots()
            data = v.view(bs, -1, 1)[0].detach().numpy()
            columns = ["v"]
            wandb.log({"v_table": wandb.Table(data=data, columns=columns)}, step=t_env)
            # sns.heatmap(
            #     data,
            #     cmap="coolwarm",
            #     center=0,
            #     ax=ax,
            # )
            # ax.set_ylabel("Timestep")
            # wandb.log({"v": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止
        return q_tot
