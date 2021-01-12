import torch


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def miou(output, target, threshold=0.5, eps=1e-6):
    thresholded_output = torch.sigmoid(output) >= threshold
    target = target == 1.0
    intersection = torch.sum((thresholded_output & target), dim=[1, 2, 3])
    union = torch.sum((thresholded_output | target), dim=[1, 2, 3])
    iou = intersection / (union + eps)
    return iou.mean()


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg_value(self):
        return self.avg

    def __str__(self):
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricTracker:
    def __init__(self):
        self.average_meters = {}
        for metric in self.get_metric_names():
            self.average_meters[metric] = AverageMeter(metric, self.get_fmt(metric))

    def get_fmt(self, metric):
        return ":.4f" if metric == "Loss" else ":6.2f"

    def get_metric_names(self):
        raise NotImplementedError

    def calculate_metrics(self, output, target, loss):
        raise NotImplementedError

    def update_metrics(self, output, target, loss, batch_size):
        output = output.detach()
        calculated_metrics = self.calculate_metrics(output, target, loss)
        for metric, value in calculated_metrics.items():
            self.update_metric(metric, value, batch_size)

    def update_metric(self, metric, value, n=1):
        self.average_meters[metric].update(value, n)

    def get_avg_values(self):
        return {name: meter.get_avg_value() for name, meter in self.average_meters.items()}

    def __str__(self):
        return " | ".join(str(average_value) for average_value in self.average_meters.values())


class ClassificationMetricTracker(MetricTracker):
    def get_metric_names(self):
        return ["Loss", "Acc@1", "Acc@5"]

    def calculate_metrics(self, output, target, loss):
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return {
            "Acc@1": acc1.item(),
            "Acc@5": acc5.item(),
            "Loss": loss.item(),
        }


class SemanticSegmentationMetricTracker(MetricTracker):
    def get_metric_names(self):
        return ["Loss", "mIOU"]

    def calculate_metrics(self, output, target, loss):
        iou = miou(output, target)
        return {
            "mIOU": iou.item(),
            "Loss": loss.item(),
        }


class BestResultTracker:
    def __init__(self, task):
        self.best_result = None
        self.metric_name = self.get_tracked_metric_name(task)

    @staticmethod
    def get_tracked_metric_name(task):
        if task == "classification":
            return "Acc@1"
        elif task == "semantic_segmentation":
            return "mIOU"
        else:
            raise ValueError(f"Unsupported task {task}")

    def is_best(self, results):
        result = results[self.metric_name]
        if self.best_result is None or self.best_result < result:
            self.best_result = result
            return True
        return False

    def get_best(self):
        return self.best_result


def get_metric_tracker(task):
    if task == "classification":
        return ClassificationMetricTracker()
    elif task == "semantic_segmentation":
        return SemanticSegmentationMetricTracker()
    else:
        raise ValueError(f"Unsupported task {task}")
