from __future__ import print_function
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class RandomSampler:
    def __init__(self, number_of_samples):
        self.number_of_samples = number_of_samples
        self.range_multiplier = torch.arange(0.0, number_of_samples + 1, 1).view(number_of_samples + 1, 1, 1)

    def sample(self, min_offset_x, max_offset_x):
        noise = torch.rand(min_offset_x.repeat(1, self.number_of_samples + 1, 1, 1).size())
        offset_x = min_offset_x + ((max_offset_x - min_offset_x) / (self.number_of_samples + 1)) * \
            (self.range_multiplier + noise)
        offset_x = offset_x.unsqueeze_(1).expand(-1, offset_x.size()[1], -1, -1, -1)
        offset_x = offset_x.contiguous().view(offset_x.size()[0],
                                              offset_x.size()[1] * offset_x.size()[2],
                                              offset_x.size()[3],
                                              offset_x.size()[4])
        return offset_x


class Evaluate:
    def __init__(self, left_features, filter_size):
        self.filter_size = filter_size

        self.left_x_coordinate = torch.arange(0.0, left_features.size()[3]).repeat(
            left_features.size()[2]).view(left_features.size()[2], left_features.size()[3])

        self.left_x_coordinate = torch.clamp(self.left_x_coordinate, min=0, max=left_features.size()[3] - 1)
        self.left_x_coordinate = self.left_x_coordinate.expand(left_features.size()[0], -1, -1).unsqueeze(1)

        self.left_y_coordinate = torch.arange(0.0, left_features.size()[2]).unsqueeze(1).repeat(
            1, left_features.size()[3]).view(left_features.size()[2], left_features.size()[3])

        self.left_y_coordinate = torch.clamp(self.left_y_coordinate, min=0, max=left_features.size()[3] - 1)
        self.left_y_coordinate = self.left_y_coordinate.expand(left_features.size()[0], -1, -1).unsqueeze(1)

    def evaluate(self, left_features, right_features, offset_x):
        right_x_coordinate = torch.clamp(self.left_x_coordinate - offset_x, min=0, max=left_features.size()[3] - 1)
        offset_y = torch.zeros_like(offset_x)
        right_y_coordinate = torch.clamp(self.left_y_coordinate - offset_y, min=0, max=left_features.size()[2] - 1)

        right_x_coordinate -= right_x_coordinate.size()[3] / 2
        right_x_coordinate /= (right_x_coordinate.size()[3] / 2)
        right_y_coordinate -= right_y_coordinate.size()[2] / 2
        right_y_coordinate /= (right_y_coordinate.size()[2] / 2)

        samples = torch.cat((right_x_coordinate.unsqueeze(4), right_y_coordinate.unsqueeze(4)), dim=4)
        samples = samples.view(samples.size()[0] * samples.size()[1],
                               samples.size()[2],
                               samples.size()[3],
                               samples.size()[4])

        offset_strength = torch.mean(-1.0 * (torch.abs(left_features.expand(
            offset_x.size()[1], -1, -1, -1) - F.grid_sample(right_features.expand(
                offset_x.size()[1], -1, -1, -1), samples))), dim=1) * 1000000000000

        offset_strength = offset_strength.view(left_features.size()[0],
                                               offset_strength.size()[0] // left_features.size()[0],
                                               offset_strength.size()[1],
                                               offset_strength.size()[2])

        offset_strength = torch.softmax(offset_strength, dim=1)
        offset_x = torch.sum(offset_x * offset_strength, dim=1).unsqueeze(1)

        return offset_x


class Propagation:
    def __init__(self, filter_size):
        self.filter_size = filter_size
        label = torch.arange(0, self.filter_size).repeat(self.filter_size).view(
            self.filter_size, 1, 1, 1, self.filter_size)

        self.one_hot_filter_h = torch.zeros_like(label).scatter_(0, label, 1).float()

        label = torch.arange(0, self.filter_size).repeat(self.filter_size).view(
            self.filter_size, 1, 1, self.filter_size, 1).long()

        self.one_hot_filter_v = torch.zeros_like(label).scatter_(0, label, 1).float()

    def propagate(self, offset_x, propagation_type="horizontal"):
        offset_x = offset_x.view(offset_x.size()[0], 1, offset_x.size()[1], offset_x.size()[2], offset_x.size()[3])

        if propagation_type is "horizontal":
            aggregated_offset_x = F.conv3d(offset_x, self.one_hot_filter_h, padding=(0, 0, self.filter_size // 2))

        else:
            aggregated_offset_x = F.conv3d(offset_x, self.one_hot_filter_v, padding=(0, self.filter_size // 2, 0))

        aggregated_offset_x = aggregated_offset_x.permute([0, 2, 1, 3, 4])
        aggregated_offset_x = aggregated_offset_x.contiguous().view(
            aggregated_offset_x.size()[0],
            aggregated_offset_x.size()[1] * aggregated_offset_x.size()[2],
            aggregated_offset_x.size()[3],
            aggregated_offset_x.size()[4])

        return aggregated_offset_x


class PatchMatch:
    def __init__(self, left_path, right_path, outfile, disparity_range):
        left_image = np.asarray(Image.open(left_path).convert('RGB'))
        right_image = np.asarray(Image.open(right_path).convert('RGB'))
        left_tensor = transforms.ToTensor()(left_image).unsqueeze(0).float().requires_grad_(True)
        right_tensor = transforms.ToTensor()(right_image).unsqueeze(0).float().requires_grad_(True)
        filter_size = 7
        self.number_of_samples = 0
        self.propagation_filter_size = 3
        self.window_size = disparity_range
        self.propagation = Propagation(self.propagation_filter_size)
        self.uniform_sampler = RandomSampler(self.number_of_samples)
        self.evaluate = None
        self.left_features, self.right_features = self.extract_features(left_tensor, right_tensor, filter_size)
        self.outfile = outfile

    def extract_features(self, left_input, right_input, filter_size):
        label = torch.arange(0, filter_size * filter_size).repeat(
            filter_size * filter_size).view(filter_size * filter_size, 1, 1, filter_size, filter_size)

        one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()

        left_features = F.conv3d(left_input.unsqueeze(1), one_hot_filter,
                                 padding=(0, filter_size // 2, filter_size // 2))
        right_features = F.conv3d(right_input.unsqueeze(1), one_hot_filter,
                                  padding=(0, filter_size // 2, filter_size // 2))

        left_features = left_features.view(left_features.size()[0],
                                           left_features.size()[1] * left_features.size()[2],
                                           left_features.size()[3],
                                           left_features.size()[4])

        right_features = right_features.view(right_features.size()[0],
                                             right_features.size()[1] * right_features.size()[2],
                                             right_features.size()[3],
                                             right_features.size()[4])

        return left_features, right_features

    def train(self, iterations):
        self.evaluate = Evaluate(self.left_features, self.propagation_filter_size)

        min_offset_x = torch.zeros((self.left_features.size()[0], 1, self.left_features.size()[2],
                                    self.left_features.size()[3]))
        max_offset_x = torch.zeros((self.left_features.size()[0], 1, self.left_features.size()[2],
                                    self.left_features.size()[3])) + self.window_size
        self.offsets = torch.zeros((self.left_features.size()[0], 2, self.left_features.size()[2],
                                self.left_features.size()[3]))

        for prop_iter in range(1, iterations+1):
            self.offsets = self.uniform_sampler.sample(min_offset_x, max_offset_x)
            self.offsets = self.propagation.propagate(self.offsets, "horizontal")
            self.offsets = self.evaluate.evaluate(self.left_features, self.right_features, self.offsets)
            self.offsets = self.propagation.propagate(self.offsets, "vertical")
            self.offsets = self.evaluate.evaluate(self.left_features, self.right_features, self.offsets)
            min_offset_x = torch.clamp(self.offsets - self.window_size // 2, min=0, max=100)
            max_offset_x = torch.clamp(self.offsets + self.window_size // 2, min=0, max=100)

        self.offsets = np.asarray(self.offsets.squeeze().detach()).astype('uint16')[:, 100:]

    def visualize(self):
        plt.imshow(self.offsets, cmap="inferno")
        plt.savefig(self.outfile)


def main(left_path, right_path, outfile, disparity_range, iterations):
    patch_match = PatchMatch(left_path=left_path,
                             right_path=right_path,
                             outfile=outfile,
                             disparity_range=disparity_range)
    patch_match.train(iterations)
    patch_match.visualize()


if __name__ == '__main__':
    tic = time.process_time()
    main(left_path="source/flying_objects/left/1001.png",
         right_path="source/flying_objects/right/1001.png",
         outfile="output/flying_objects/diffpatchmatch2/1001.png",
         disparity_range=100, iterations=2)
    toc = time.process_time()
    print("Differential PatchMatch Runtime ( 2 iter) ( flying_objects ): " + str(toc-tic))
