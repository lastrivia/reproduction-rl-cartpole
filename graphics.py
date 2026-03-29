import pygame

from math import sin, cos

from physics import CartPoleState, step, l_pole, fps as default_fps


class CartPoleGraphics:
    def __init__(self, width: int, height: int, scale: int):
        self.width = width
        self.height = height
        self.scale = scale

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def drawcall(self, state: CartPoleState):
        self.screen.fill((255, 255, 255))

        cart_x_px = int(self.width / 2 + state.x_cart * self.scale)
        cart_y_px = int(self.height / 2)

        pole_x_px = int(cart_x_px + l_pole * 2.0 * self.scale * sin(state.theta_pole))
        pole_y_px = int(cart_y_px - l_pole * 2.0 * self.scale * cos(state.theta_pole))

        cart_size = 1.0
        cart_size_px = int(cart_size * self.scale)

        pygame.draw.rect(
            surface=self.screen,
            color=(127, 127, 127),
            rect=(cart_x_px - cart_size_px, cart_y_px, cart_size_px * 2, cart_size_px)
        )
        pygame.draw.rect(
            surface=self.screen,
            color=(0, 0, 0),
            rect=(cart_x_px - cart_size_px, cart_y_px, cart_size_px * 2, cart_size_px),
            width=3
        )

        pygame.draw.line(
            surface=self.screen,
            color=(0, 0, 255),
            start_pos=(cart_x_px, cart_y_px),
            end_pos=(pole_x_px, pole_y_px),
            width=3
        )

        pygame.display.flip()

    def tick(self, fps: int):
        self.clock.tick(fps)


if __name__ == "__main__":

    # demo (keyboard control, not RL)
    force = 0.0
    window = CartPoleGraphics(width=800, height=600, scale=50)
    state = CartPoleState(0, 0, 0, 0)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            force = -1.0
        elif keys[pygame.K_d]:
            force = 1.0

        state = step(state, force)
        window.drawcall(state)
        window.tick(default_fps)
