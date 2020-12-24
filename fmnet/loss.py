# 3p
import torch
import torch.nn as nn


class SoftErrorLoss(nn.Module):
    """
    Calculate soft error loss as defined is FMNet paper.
    """
    def __init__(self):
        super().__init__()

    def forward(self, P, geodesic_dist):
        """Compute soft error loss

        Arguments:
            P {torch.Tensor} -- soft correspondence matrix. Shape: batch_size x num_vertices x num_vertices.
            geodesic_dist {torch.Tensor} -- geodesic distances on Y. Shape: batch_size x num_vertices x num_vertices.

        Returns:
            float -- total loss
        """
        loss = torch.sqrt(((P * geodesic_dist) ** 2).sum((1, 2)))
        return torch.mean(loss)
