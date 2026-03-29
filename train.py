import time
import torch
from torch.optim import Adam

from physics import CartPoleState, step
from model import CartPoleModel

if __name__ == "__main__":

    torch.manual_seed(42)

    model = CartPoleModel(hidden=128)
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    model.to(device)

    batch_size = 64
    lr = 1e-3
    optimizer = Adam(model.parameters(), lr=lr)

    solved_lifetime = 3600

    epoch = 0
    verbose = 1

    sum_lifetime = 0
    sum_iterations = 0
    start_time = time.time()
    while True:
        if epoch % verbose == 0 and epoch != 0:
            avg_lifetime = sum_lifetime / (verbose * batch_size)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            avg_it_ps = sum_iterations / elapsed_time
            print(f"epoch: {epoch}, avg lifetime: {avg_lifetime:.1f}, {avg_it_ps:.1f} it/s")
            sum_lifetime = 0
            sum_iterations = 0
            pass

        states = [CartPoleState() for _ in range(batch_size)]
        log_probs = []
        rewards = [[] for _ in range(batch_size)]
        masks = [[] for _ in range(batch_size)]

        for _ in range(solved_lifetime):
            state_tensor = torch.stack([state.to_tensor(device=device) for state in states])
            logits = model(state_tensor)
            probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

            log_prob = dist.log_prob(actions)
            log_probs.append(log_prob)

            actions_list = actions.to(torch.device("cpu")).tolist()

            n_alive = 0
            for i in range(batch_size):
                masks[i].append(1.0 if states[i].is_alive else 0.0)

                force = -1.0 if actions_list[i] == 0 else 1.0
                states[i] = step(states[i], force)

                if states[i].is_alive:
                    n_alive += 1
                    rewards[i].append(1.0)
                else:
                    rewards[i].append(0.0)

            if n_alive == 0:
                break

        lifetime = [sum(i) for i in masks]
        sum_lifetime += sum(lifetime)
        sum_iterations += max(lifetime)

        returns = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            return_value = 0
            gamma = 0.99
            for r in reversed(rewards[i]):
                return_value = r + gamma * return_value
                returns[i].insert(0, return_value)

        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs, dim=-1)
        masks = torch.tensor(masks, device=device)

        loss = -(log_probs * returns * masks).sum() / masks.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch = epoch + 1
