import pygame

from math import sin, cos

from physics import CartPoleState, step, l_pole, fps


class CartPoleGraphics:
    def __init__(self, width: int, height: int, scale: int, aa_level: float = 1.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.aa_level = aa_level

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def drawcall(self, state: CartPoleState):
        canvas = pygame.Surface((self.width * self.aa_level, self.height * self.aa_level))

        canvas.fill((255, 255, 255))

        cart_x_px = int((self.width / 2.0 + state.x_cart * self.scale) * self.aa_level)
        cart_y_px = int((self.height / 2.0) * self.aa_level)

        pole_x_px = int(cart_x_px + (l_pole * 2.0 * self.scale * sin(state.theta_pole)) * self.aa_level)
        pole_y_px = int(cart_y_px - (l_pole * 2.0 * self.scale * cos(state.theta_pole)) * self.aa_level)

        line_width = 1.0 / 20.0
        line_width_px = max(1, int(line_width * self.scale * self.aa_level))

        cart_size = 1.0
        cart_size_px = int(cart_size * self.scale * self.aa_level)

        pygame.draw.rect(
            surface=canvas,
            color=(127, 127, 127),
            rect=(cart_x_px - cart_size_px, cart_y_px, cart_size_px * 2, cart_size_px)
        )
        pygame.draw.rect(
            surface=canvas,
            color=(0, 0, 0),
            rect=(cart_x_px - cart_size_px, cart_y_px, cart_size_px * 2, cart_size_px),
            width=line_width_px
        )

        pygame.draw.line(
            surface=canvas,
            color=(0, 0, 255),
            start_pos=(cart_x_px, cart_y_px),
            end_pos=(pole_x_px, pole_y_px),
            width=line_width_px
        )

        final = pygame.transform.smoothscale(canvas, (self.width, self.height))
        self.screen.blit(final, (0, 0))

        pygame.display.flip()

    def tick(self, fps: int):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.clock.tick(fps)


if __name__ == "__main__":

    # demo (keyboard control, not RL)
    force = 0.0
    window = CartPoleGraphics(width=1280, height=960, scale=100, aa_level=1.5)
    state = CartPoleState(0, 0, 0, 0)

    while True:
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            force = -1.0
        elif keys[pygame.K_d]:
            force = 1.0

        state = step(state, force)
        window.drawcall(state)
        window.tick(fps)
