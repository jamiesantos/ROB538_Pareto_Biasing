import copy
import torch

from Policy import Policy

class Utils:
    def SBX(self, x1: Policy, x2: Policy, eta=15):
        """
        Perform Simulated Binary Crossover (SBX) on two Policy instances.

        Parameters:
        - x1 (Policy): First parent Policy instance.
        - x2 (Policy): Second parent Policy instance.
        - eta (int, optional): Distribution index controlling the spread of offspring. Default is 15.

        Returns:
        - y1 (Policy): First offspring Policy instance.
        - y2 (Policy): Second offspring Policy instance.
        """
        y1 = copy.deepcopy(x1)
        y2 = copy.deepcopy(x2)

        with torch.no_grad():
            for param_x1, param_x2, param_y1, param_y2 in zip(
                x1.parameters(), x2.parameters(), y1.parameters(), y2.parameters()
            ):
                p1 = param_x1.data
                p2 = param_x2.data

                u = torch.rand_like(p1)
                beta_q = torch.where(
                    u <= 0.5,
                    (2 * u) ** (1.0 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1.0 / (eta + 1)),
                )

                child1 = 0.5 * ((1 + beta_q) * p1 + (1 - beta_q) * p2)
                child2 = 0.5 * ((1 - beta_q) * p1 + (1 + beta_q) * p2)

                param_y1.data.copy_(child1)
                param_y2.data.copy_(child2)

        return y1, y2
