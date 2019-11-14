import numpy as np
import time
import matplotlib.pyplot as plt

INTMIN = -99999999


class ClassicStereo:
    def __init__(self, left_path, right_path, outfile, disparity_range):
        self.outfile = outfile
        self.disparity_range = disparity_range

        self.left = plt.imread(left_path)
        self.right = plt.imread(right_path)

        self.left_patches, self.right_patches = self.get_patches()
        self.height = self.right_patches.shape[0]
        self.width = self.right_patches.shape[1]-self.disparity_range
        self.offsets = self.initialize_offsets()

    def initialize_offsets(self):
        offsets = np.zeros((self.height, self.width), dtype=int)
        return offsets

    def train(self):
        for x in range(self.height):
            for y in range(self.width):
                distances = np.zeros(self.disparity_range)
                for offset in range(self.disparity_range):
                    distances[offset] = self.patch_distance_error(x, y, offset)
                self.offsets[x][y] = np.argmax(distances)

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
        distance_error = -np.sum(np.square(right_patch-left_patch))
        return distance_error

    def visualize(self):
        plt.imshow(self.offsets, cmap="inferno")
        plt.savefig(self.outfile)


def main(left_path, right_path, outfile, disparity_range):
    classic_stereo = ClassicStereo(left_path=left_path, right_path=right_path,
                                   outfile=outfile, disparity_range=disparity_range)
    classic_stereo.train()
    classic_stereo.visualize()


if __name__ == "__main__":
    tic = time.process_time()
    main(left_path="source/flying_objects/left/1001.png",
         right_path="source/flying_objects/right/1001.png",
         outfile="output/flying_objects/classic/1001.png",
         disparity_range=100)
    toc = time.process_time()
    print("Classic Stereo Runtime ( flying_objects ): " + str(toc-tic))
