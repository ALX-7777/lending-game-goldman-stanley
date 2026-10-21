"""Game-theorist lens: the net war (Bertrand with a 2% loyalty bonus) has no pure equilibrium, so the team plays a
mixed strategy: each band member gets an independent net drawn uniformly on [0.60, 0.99] (cannot be sniped
everywhere by a rival that guesses a round number). Deterministic draw."""
import numpy as np

NAME = "GT mixed-strategy net U(0.60, 0.99)"
REGIME = "net"
ALSO_VS_BASE = True


def rates(p, ctx):
    rng = np.random.default_rng(12345)
    net = rng.uniform(0.60, 0.99, ctx["n"])
    return np.where(p <= ctx["tau"], ctx["below"](p), net)
