import torch

from physics import CartPoleState, step, fps
from model import CartPoleModel
from graphics import CartPoleGraphics

if __name__ == '__main__':

    window = CartPoleGraphics(width=800, height=600, scale=50)
    state = CartPoleState()
    model = CartPoleModel(hidden=128)
    model.load_state_dict(torch.load("cartpole-0329-2336-finished.pt"))

    fast_forward = 4

    while True:
        logits = model(state.to_tensor())
        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        force = -1.0 if action == 0 else 1.0

        state = step(state, force)
        window.drawcall(state)
        window.tick(int(fps * fast_forward))

