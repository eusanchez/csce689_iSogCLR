import torch

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state.setdefault(p, {})
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # momentum update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # parameter update
                update = torch.sign(exp_avg)
                p.add_(update, alpha=-lr)

        return loss
