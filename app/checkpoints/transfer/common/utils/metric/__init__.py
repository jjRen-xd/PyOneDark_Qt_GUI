import torch
import prettytable

__all__ = ['keypoint_detection']

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

def accuracy(output, target, topk=(1,0)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) #
        pred = pred.t()
        #correct = pred.eq(target[None])
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k * (100.0 / batch_size))        
        return res


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, iu = self.compute()

        table = prettytable.PrettyTable(["class", "acc", "iou"])
        for i, class_name, per_acc, per_iu in zip(range(len(classes)), classes, (acc * 100).tolist(), (iu * 100).tolist()):
            table.add_row([class_name, per_acc, per_iu])

        return 'global correct: {:.1f}\nmean correct:{:.1f}\nmean IoU: {:.1f}\n{}'.format(
            acc_global.item() * 100, acc.mean().item() * 100, iu.mean().item() * 100, table.get_string())

