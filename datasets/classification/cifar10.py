import torchvision

import cv2


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
