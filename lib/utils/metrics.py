from torch import Tensor
import torch


def single_label_count_accurate(pred: Tensor, correct: Tensor) -> int:
    """
    Calculates number of correct predictions from a
    batch_size * categories prediction matrix
    returns: int
    """

    bool_vector = pred.argmax(dim=1) == correct

    return bool_vector.sum().item()


def prec_count_at_k(pred: Tensor, correct: Tensor, k: int) -> int:
    """
    Calculates number of accurate predictions up to top k
    recommendations

    Inputs
    pred: tensor of size (BATCH_SIZE, n_classes)
    correct: one hot encoded tensor of size (BATCH_SIZE, n_classes)
    k: int

    returns: the total number of correct in the batch
    """

    topK, indices = pred.topk(k)

    new_pred = torch.zeros_like(pred)

    new_pred = new_pred.scatter(1, indices, topK)

    new_pred[new_pred != 0] = 1

    # set to True, the predictions in new_pred == 1 and where it is correct
    count = (new_pred > 0) & (new_pred == correct)

    return count.sum().item()


class PrecisionCounter:

    def __init__(self, k: list = [1, 3, 5]):
        """
        Keeps track of multiple prec@k metrics
        """

        self.precision_tracker = {key: 0 for key in k}

    def update(self, pred: Tensor, correct: Tensor):
        """
        Updates the precision tracker
        """

        for k in self.precision_tracker:
            self.precision_tracker[k] += prec_count_at_k(pred, correct, k)

    def dict(self):

        return self.precision_tracker

    def finalise(self, loader_length, batch_size):

        """
        Takes all the counts in precision tracker and performs macro averaging
        """
        for k in self.precision_tracker:
            self.precision_tracker[k] = self.precision_tracker[k] / (loader_length * batch_size * k)

    def __str__(self):

        output_str = ""

        for k in self.precision_tracker:

            output_str += "{:d}: {:.3f}. ".format(k, self.precision_tracker[k])

        return output_str


