#!/usr/bin/env python

from typing import Tuple, List
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

class GlimpseSensor:
    def __init__(self, patch_width: int, num_scales: int)->None:
        """
        patch_width: int width of patch
        num_scales: int number of glimpses for img
        """
        self.patch_width = patch_width
        self.num_scales = num_scales


    @staticmethod
    def _get_ctr(w: int, h: int)->Tuple[int, int]:
        return int(w - w * 0.5), int(h - h * 0.5)

    def get_coords(self, img_size:Tuple[int, int], location: Tuple[int, int])->List[Tuple[float, float, float, float]]:
        height, width = img_size
        w = self.patch_width
        coords = []
        for _ in range(self.num_scales):
            x, y = location  # assume the location is between(-1, -1) to (+1, +1)
            x_ctr, y_ctr = self._get_ctr(height, width)
            x = x + x_ctr
            y = y + y_ctr
            half_width = w/2
            x0 = max(0, x - half_width)
            y0 = max(0, y - half_width)
            x1 = min(width, x + half_width)
            y1 = min(height, y + half_width)
            coords.append((x0, y0, x1, y1))
            w *= 4

        return coords


    def glimpse(self, image: torch.tensor, location: torch.tensor)-> torch.tensor:
        image = transforms.ToPILImage()(image)
        location = tuple(location.detach().numpy())
        coords = self.get_coords(image.size, location)
        crops = []
        for coord in coords:
            crop = image.crop(coord)
            crop = crop.resize((self.patch_width, self.patch_width))
            crops.append(crop)

        crops = [transforms.ToTensor()(x) for x in crops]
        crops = torch.stack(crops, dim=0)
        return crops


    def forward(self, images: torch.tensor, location: torch.tensor)->torch.tensor:
        crops_batch = []
        for idx in range(0, images.shape[0]):
            # import ipdb; ipdb.set_trace()
            img = images[idx].squeeze(0)
            loc = location[idx].squeeze(0)
            assert loc[0] >= -1.0 and loc[0] <= 1.0
            assert loc[1] >= -1.0 and loc[1] <= 1.0
            crops = self.glimpse(img, loc)
            crops_batch.append(crops)
        crops_batch = torch.stack(crops_batch, dim=0)
        return crops_batch


def visualize_patch(img: Image, loc: Tuple[float, float], sensor: GlimpseSensor)-> None:
    coords = sensor.get_coords(img.size, loc)
    canvas = np.array(img)
    # plt.axis('tight')
    # plt.axis('off')
    for coord in coords:
        x0, y0, x1, y1 = list(map(lambda x: int(x), coord))
        print("Coords", x0, y0, x1, y1)
        canvas = cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 0), 1)
    plt.imshow(canvas)
    plt.show()


def main():
    sensor = GlimpseSensor(64, 2)
    path = './test_data/cat2.jpg'
    img = torch.unsqueeze(transforms.functional.to_tensor(Image.open(path).resize((256, 256))), 0)
    loc = torch.zeros((1, 2))
    batch_crops = sensor.forward(img, loc)
    print('Shape of batch_crops', batch_crops.shape)
    for crops in batch_crops:
        for crop in crops:
            print('Crop shape', crop.shape)
            plt.imshow(transforms.functional.to_pil_image(crop))
            plt.show()
    # visualize_patch(img, (14, 100), sensor)


if __name__ == "__main__":
    main()
