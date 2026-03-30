import time

import torch
import pygame

from physics import CartPoleState, step, fps
from model import CartPoleModel
from graphics import CartPoleGraphics

if __name__ == '__main__':

    window = CartPoleGraphics(width=1280, height=960, scale=100, aa_level=1.5)
    state = CartPoleState()
    model = CartPoleModel(hidden=128)
    model.load_state_dict(torch.load("cartpole-0329-2336-finished.pt"))
    model.eval()

    fast_forward = 1
    use_deterministic = True

    do_respawn = True
    respawn_latency = 1.0
    respawn_counter = 0

    with torch.no_grad():
        while True:
            if do_respawn and not state.is_alive:
                if respawn_counter == 0:
                    respawn_counter = time.time()
                if time.time() > respawn_counter + respawn_latency:
                    state = CartPoleState()
                    respawn_counter = 0

            logits = model(state.to_tensor())
            probs = torch.softmax(logits, dim=-1)

            if use_deterministic:
                action = torch.argmax(probs).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            force = -1.0 if action == 0 else 1.0

            keys = pygame.key.get_pressed()

            if keys[pygame.K_a]:
                force += -1.0
            elif keys[pygame.K_d]:
                force += 1.0

            state = step(state, force)
            window.drawcall(state, show_fps=True)
            window.tick(int(fps * fast_forward))

