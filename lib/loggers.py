import csv


class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = ["Epoch"]
        for mode in ("Train", "Val"):
            for metric in self.get_metric_names():
                self.header.append(f"{mode} {metric}")

        self.write_header()

    def get_metric_names(self):
        raise NotImplementedError

    def write_header(self):
        if self.filepath is None:
            return
        with open(self.filepath, "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()

    def write_metrics(self, epoch, train_metrics, val_metrics):
        if self.filepath is None:
            return
        row = {"Epoch": epoch}
        for metric, value in train_metrics.items():
            row[f"Train {metric}"] = value
        for metric, value in val_metrics.items():
            row[f"Val {metric}"] = value

        with open(self.filepath, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(row)


class ClassificationCSVLogger(CSVLogger):
    def get_metric_names(self):
        return ["Loss", "Acc@1", "Acc@5"]


class SemanticSegmentationCSVLogger(CSVLogger):
    def get_metric_names(self):
        return ["Loss", "mIOU"]


def get_csv_logger(task, filepath):
    if task == "classification":
        return ClassificationCSVLogger(filepath)
    elif task == "semantic_segmentation":
        return SemanticSegmentationCSVLogger(filepath)
    else:
        raise ValueError(f"Unsupported task {task}")
