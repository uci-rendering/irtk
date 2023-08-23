import torch
import largesteps
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential

class LargeStepsOptimizer(torch.optim.Optimizer):
    def __init__(self, V, F, lr=0.1, betas=(0.9, 0.999), lmbda=20, device='cuda'):
        self.device = device
        largesteps.geometry.device = self.device
        self.V = V
        self.F = F.to(self.device)

        self.M = compute_matrix(self.V.detach().clone().to(self.device), self.F, lmbda)
        self.u = to_differential(
            self.M, self.V.detach().clone().to(self.device)).clone().requires_grad_()
        defaults = dict(F=self.F, lr=lr, betas=betas)

        self.optimizer = AdamUniform([self.u], lr=lr, betas=betas)
        super(LargeStepsOptimizer, self).__init__([V], defaults)

    def step(self):
        # build compute graph from u to V
        V = from_differential(self.M, self.u, 'Cholesky')
        # propagate gradients from V to u
        V.backward(self.V.grad.to(self.device))
        # step u
        self.optimizer.step()
        # update param
        self.V.data.copy_(from_differential(
            self.M, self.u, 'Cholesky').to(self.V.device))

    def zero_grad(self):
        super(LargeStepsOptimizer, self).zero_grad()
        self.optimizer.zero_grad()