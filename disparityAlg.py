import matplotlib.pyplot as plt
import numpy as np

INTMIN = -99999999


class DisparityAlg:
    def __init__(self, left_path='left.png', right_path='right.png', outfile="output.png"):
        self.outfile = outfile
        self.disparity_range = 55

        self.left = plt.imread(left_path)
        self.right = plt.imread(right_path)

        self.left_patches, self.right_patches = self.get_patches()
        self.height = self.right_patches.shape[0]
        self.width = self.right_patches.shape[1]-self.disparity_range

    def get_patches(self):
        height, width, depth = self.left.shape
        left_patches = np.zeros((height - 6, width - 6, 7, 7, depth))
        right_patches = np.zeros((height - 6, width - 6, 7, 7, depth))
        for x in range(3, height - 3):
            for y in range(3, width - 3):
                left_patches[x - 3][y - 3] = self.left[x - 3:x + 4, y - 3:y + 4, :]
                right_patches[x - 3][y - 3] = self.right[x - 3:x + 4, y - 3:y + 4, :]
        return left_patches, right_patches

    def patch_distance_error(self, x, y, offset):
        if y+offset >= self.right_patches.shape[1]:
            return INTMIN
        right_patch = self.right_patches[x][y]
        left_patch = self.left_patches[x][y+offset]
        distance_error = -np.log(np.sum(np.square(right_patch-left_patch)))
        return distance_error

    def visualize(self):
        plt.imshow(self.offsets, cmap="inferno")
        plt.savefig(self.outfile)
