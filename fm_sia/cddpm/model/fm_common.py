"""Fixed-step ODE updates with simultaneous updates for all state components."""


def masked_mse(pred, target, mask=None):
    error = (pred - target).square()
    if mask is None:
        return error.flatten(1).mean(1).mean()
    mask = mask.to(error).expand_as(error)
    count = mask.flatten(1).sum(1)
    if (count == 0).any():
        raise ValueError("At least one image has no valid pixels")
    return ((error * mask).flatten(1).sum(1) / count).mean()


def masked_mae(pred, target, mask=None):
    """Per-case spatial mean followed by batch mean; preserve operators' L1 loss."""
    error = (pred - target).abs()
    if mask is None:
        return error.flatten(1).mean(1).mean()
    mask = mask.to(error).expand_as(error)
    count = mask.flatten(1).sum(1)
    if (count == 0).any():
        raise ValueError("At least one image has no valid pixels")
    return ((error * mask).flatten(1).sum(1) / count).mean()


def ode_step(field, state, t, dt, method):
    def add(y, k, scale):
        return tuple(a + scale*b for a, b in zip(y, k))
    k1 = field(state, t)
    if method == "euler":
        return add(state, k1, dt)
    if method == "heun":
        k2 = field(add(state, k1, dt), t + dt)
        return tuple(y + dt*(a+b)/2 for y, a, b in zip(state, k1, k2))
    if method == "rk4":
        k2 = field(add(state, k1, dt/2), t + dt/2)
        k3 = field(add(state, k2, dt/2), t + dt/2)
        k4 = field(add(state, k3, dt), t + dt)
        return tuple(y + dt*(a+2*b+2*c+d)/6 for y, a, b, c, d in zip(state, k1, k2, k3, k4))
    raise ValueError(f"Unsupported fixed-step solver: {method}")

