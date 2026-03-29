import torch
from torch.optim import Adam

from physics import CartPoleState, step
from model import CartPoleModel

if __name__ == "__main__":

    model = CartPoleModel(hidden=128)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    lr = 1e-3
    optimizer = Adam(model.parameters(), lr=lr)

    verbose = 100

    epoch = 0
    sum_lifetime = 0
    while True:
        if epoch % verbose == 0 and epoch != 0:
            print(f"epoch: {epoch}, avg lifetime: {sum_lifetime / verbose}")
            sum_loss = 0
            sum_lifetime = 0
            pass

        state = CartPoleState()
        log_probs = []
        rewards = []

        while True:
            state_tensor = state.to_tensor(device=device)
            logits = model(state_tensor)
            probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)

            force = -1.0 if action.item() == 0 else 1.0
            state = step(state, force)

            log_probs.append(log_prob)
            if state.is_alive():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                break

        returns = []
        return_value = 0
        gamma = 0.99

        for r in reversed(rewards):
            return_value = r + gamma * return_value
            returns.insert(0, return_value)

        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs)

        loss = -(log_probs * returns).mean()

        lifetime = len(rewards)
        sum_lifetime += lifetime

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch = epoch + 1
