import matplotlib.pyplot as plt
import numpy as np
import time

INTMIN = -99999999


class PatchMatch:
    def __init__(self, left_path='left.png', right_path='right.png'):
        self.disparity_range = 55

        self.left = plt.imread(left_path)
        self.right = plt.imread(right_path)

        self.left_patches, self.right_patches = self.get_patches()
        self.imgHeight, self.imgWidth, _, _, _ = self.left_patches.shape
        self.offsets = self.initialize_offsets()

    def get_patches(self):
        height, width, depth = self.left.shape
        left_patches = np.zeros((height - 6, width - 6, 7, 7, depth))
        right_patches = np.zeros((height - 6, width - 6, 7, 7, depth))
        for x in range(3, height - 3):
            for y in range(3, width - 3):
                left_patches[x - 3][y - 3] = self.left[x - 3:x + 4, y - 3:y + 4, :]
                right_patches[x - 3][y - 3] = self.right[x - 3:x + 4, y - 3:y + 4, :]
        return left_patches, right_patches

    def initialize_offsets(self):
        offsets = np.zeros((self.imgHeight, self.imgWidth), dtype=int)
        return offsets

    def patch_distance_error(self, x, y, offset):
        if y+offset >= self.imgWidth:
            return INTMIN
        right_patch = self.right_patches[x][y]
        left_patch = self.left_patches[x][y+offset]
        distance_error = -np.log(np.sum(np.abs(right_patch-left_patch)))
        return distance_error

    def visualize(self, outfile='classicStereo2.png'):
        plt.imshow(self.offsets[:, :-self.disparity_range], cmap="inferno")
        plt.savefig(outfile)

    def train(self):
        for x, row in enumerate(self.offsets):
            for y, _ in enumerate(row[:-self.disparity_range]):
                distances = np.zeros(self.disparity_range)
                for offset, _ in enumerate(distances):
                    distances[offset] = self.patch_distance_error(x, y, offset)
                self.offsets[x][y] = np.argmax(distances)
            if x % 10 == 0:
                print("  Image row {0:d} / {1:d}".format(x, self.imgHeight))


if __name__ == "__main__":
    # Start a timer
    tic = time.process_time()

    # Calculate Map
    patch_match = PatchMatch()
    patch_match.train()

    # Display compute time.
    toc = time.process_time()

    patch_match.visualize()
    elapsed = toc - tic
    print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))
