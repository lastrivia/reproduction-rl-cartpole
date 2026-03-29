import time
import torch
from torch.optim import Adam
from tqdm import tqdm

from physics import CartPoleStateBatched, step_batched, fps
from model import CartPoleModel

if __name__ == "__main__":

    torch.manual_seed(42)

    model = CartPoleModel(hidden=128)
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    model.to(device)

    batch_size = 1024
    lr = 1e-2
    optimizer = Adam(model.parameters(), lr=lr)

    solved_lifetime = 120 * fps

    reward_gamma = 0.75 ** (1.0 / fps)

    epoch = 0
    max_epoch = 200
    while True:
        batch = CartPoleStateBatched(batch_size=batch_size, device=device)
        log_probs = []
        rewards = []
        masks = []
        n_solved = 0

        for t in tqdm(range(solved_lifetime)):
            logits = model(batch.states)
            probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

            log_prob = dist.log_prob(actions)
            log_probs.append(log_prob)

            with torch.no_grad():
                masks.append(batch.is_alive * 1.0)
                batch = step_batched(batch, force=actions * 2.0 - 1.0)  # [-1.0, 1.0]
                if t == solved_lifetime - 1:
                    rewards.append(batch.is_alive * (1.0 / (1.0 - reward_gamma)))
                    n_solved = batch.count_alive()
                else:
                    rewards.append(batch.is_alive * 1.0)
                    if batch.count_alive() == 0:
                        break

        log_probs = torch.stack(log_probs, dim=-1)
        rewards = torch.stack(rewards, dim=-1)
        masks = torch.stack(masks, dim=-1)
        # [batch_size, max_lifetime]

        max_lifetime = masks.shape[1]
        sum_lifetime = masks.sum().item()

        print(f" epoch: {epoch},"
              f" avg lifetime: {sum_lifetime / batch_size:.1f},"
              f" {n_solved / batch_size * 100.0:.1f}% solved")

        if round(n_solved) == batch_size or epoch == max_epoch:
            status = "finished" if round(n_solved) == batch_size else "terminated"
            timestamp = time.strftime("%m%d-%H%M")
            filename = f"cartpole-{timestamp}-{status}.pt"
            torch.save(model.state_dict(), filename)
            print(f"Training {status}. Saved model to {filename}")
            break

        returns = torch.zeros_like(rewards)
        r = torch.zeros(batch_size, device=device)
        for t in reversed(range(max_lifetime)):
            r = rewards[:, t] + reward_gamma * r
            returns[:, t] = r
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(log_probs * returns * masks).sum() / masks.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch = epoch + 1
