import wandb
import numpy as np

num_episodes = 5
steps_per_episode = 10

with wandb.init(project="test") as run:
    for episode in range(num_episodes):
        print("Episode:", episode)
        for step in range(steps_per_episode):
            print(" Step:", step)
            state = np.random.rand(10, 10)  # ダミーの状態データ
            np.save("state.npy", state)
            
            # 同じ名前でOK - 自動的にバージョンが増える
            artifact = wandb.Artifact(
                name="rl-states",  # 毎回同じ名前
                type="state",
                metadata={"episode": episode, "step": step}
            )
            artifact.add_file("state.npy")
            run.log_artifact(artifact)