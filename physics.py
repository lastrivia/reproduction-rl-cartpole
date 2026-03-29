import torch
from math import sin, cos, pi
from dataclasses import dataclass

m_cart = 1.0
m_pole = 0.1
m_total = m_pole + m_cart
gravity = 9.81
l_pole = 1.0

fps = 60
dt = 1.0 / fps

threshold_theta = pi / 15.0
threshold_x = 2.4

@dataclass
class CartPoleState:
    x_cart: float
    v_cart: float
    theta_pole: float
    omega_pole: float
    is_alive: bool

    def __init__(self, x_cart = 0.0, v_cart = 0.0, theta_pole = 0.0, omega_pole = 0.0):
        self.x_cart = x_cart
        self.v_cart = v_cart
        self.theta_pole = theta_pole
        self.omega_pole = omega_pole
        self.is_alive = self._is_alive()

    def to_tensor(self, dtype=torch.float32, device=None) -> torch.Tensor:
        return torch.tensor(
            data=[self.x_cart, self.v_cart, self.theta_pole, self.omega_pole],
            dtype=dtype, device=device
        )

    def _is_alive(self) -> bool:
        return abs(self.theta_pole) < threshold_theta and abs(self.x_cart) < threshold_x


def step(now: CartPoleState, force: float) -> CartPoleState:
    x_cart = now.x_cart
    v_cart = now.v_cart
    theta_pole = now.theta_pole
    omega_pole = now.omega_pole

    sin_theta = sin(theta_pole)
    cos_theta = cos(theta_pole)

    beta_pole = (
        gravity * sin_theta -
        (force + m_pole * l_pole * (omega_pole ** 2.0) * sin_theta) / m_total * cos_theta
    ) / (
        l_pole * (4.0 / 3.0 - m_pole * (cos_theta ** 2.0) / m_total)
    )

    a_cart = (
        force + m_pole * l_pole * ((omega_pole ** 2.0) * sin_theta - beta_pole * cos_theta)
    ) / m_total

    v_cart += a_cart * dt
    x_cart += v_cart * dt

    omega_pole += beta_pole * dt
    theta_pole += omega_pole * dt

    next_state = CartPoleState(x_cart, v_cart, theta_pole, omega_pole)
    if not now.is_alive:
        next_state.is_alive = False

    return next_state

class CartPoleStateBatched:
    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")):
        self.batch_size = batch_size
        self.device = device
        self.states = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        self.is_alive = torch.ones((batch_size,), dtype=torch.bool, device=device)

    def count_alive(self):
        return self.is_alive.sum().item()

def step_batched(batch: CartPoleStateBatched, force: torch.Tensor) -> CartPoleStateBatched:
    x_cart = batch.states[:, 0]
    v_cart = batch.states[:, 1]
    theta_pole = batch.states[:, 2]
    omega_pole = batch.states[:, 3]

    sin_theta = torch.sin(theta_pole)
    cos_theta = torch.cos(theta_pole)

    beta_pole = (
        gravity * sin_theta -
        (force + m_pole * l_pole * (omega_pole ** 2.0) * sin_theta) / m_total * cos_theta
    ) / (
        l_pole * (4.0 / 3.0 - m_pole * (cos_theta ** 2.0) / m_total)
    )

    a_cart = (
        force + m_pole * l_pole * ((omega_pole ** 2.0) * sin_theta - beta_pole * cos_theta)
    ) / m_total

    batch_next = CartPoleStateBatched(batch_size=batch.batch_size, device=batch.device)

    batch_next.states[:, 1] = v_cart + a_cart * dt
    batch_next.states[:, 0] = x_cart + batch_next.states[:, 1] * dt

    batch_next.states[:, 3] = omega_pole + beta_pole * dt
    batch_next.states[:, 2] = theta_pole + batch.states[:, 3] * dt

    theta = batch_next.states[:, 2]
    x = batch_next.states[:, 0]

    alive_mask = (theta.abs() < threshold_theta) & (x.abs() < threshold_x)
    batch_next.is_alive = batch.is_alive & alive_mask

    return batch_next
