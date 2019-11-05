import matplotlib.pyplot as plt
import numpy as np
import time

INTMAX = 99999999


def get_patches(image):
    height, width = image.shape
    patches = np.zeros((height-6, width-6, 7, 7))
    for x in range(3, height-3):
        for y in range(3, width-3):
            patches[x-3][y-3] = image[x-3:x+4, y-3:y+4]
    return patches


class PatchMatch:
    def __init__(self, left_path='left.png', right_path='right.png'):
        self.left_img = plt.imread(left_path)
        self.right_img = plt.imread(right_path)

        self.left_grey = np.mean(self.left_img, axis=2)
        self.right_grey = np.mean(self.right_img, axis=2)
        self.imgHeight, self.imgWidth = np.shape(self.right_grey)

        self.left_patches = get_patches(self.left_grey)
        self.right_patches = get_patches(self.right_grey)
        self.offsets = self.initialize_offsets()
        self.new_offsets = self.offsets.copy()

    def initialize_offsets(self):
        offsets = np.zeros((self.right_patches.shape[0], self.right_patches.shape[1]), dtype=int).T
        for x in range(125, 251):
            for y in range(200, 301):
                offsets[x][y] += np.random.randint(0, high=min(self.right_patches.shape[1]-x-1, 50))
        return offsets.T

    def patch_distance_error(self, x, y, offset):
        if y+offset >= self.right_patches.shape[1]:
            return INTMAX
        right_patch = self.right_patches[x][y]
        left_patch = self.left_patches[x][y+offset]
        distance_error = np.sum(np.square(right_patch-left_patch))
        return distance_error

    def propagate_patch(self, x, y):
        offset_args = [self.offsets[x][y],
                       self.offsets[x-1][y],
                       self.offsets[x][y-1]]
        distance_errors = [self.patch_distance_error(x, y, offset_arg) for offset_arg in offset_args]
        best_offset = offset_args[int(np.argmin(distance_errors))]
        self.offsets[x][y] = best_offset

    def propagate(self):
        for x in np.arange(201, 301):
            for y in np.arange(126, 250):
                self.propagate_patch(x, y)

    def random_search(self):
        radius = 50
        while radius > 0:
            for x in np.arange(200, 301):
                for y in np.arange(125, 251):
                    new_offset = np.random.randint(max(0, self.offsets[x][y]-radius),
                                                   high=min(self.right_patches.shape[1]-y-1, 50,
                                                            self.offsets[x][y]+radius))
                    if self.patch_distance_error(x, y, self.offsets[x][y]) > \
                            self.patch_distance_error(x, y, new_offset):
                        self.offsets[x][y] = new_offset
            radius = int(radius/4)

    def visualize(self, outfile='patchMatch2_sample.png'):
        plt.imshow(self.offsets, cmap="inferno")
        plt.savefig(outfile)

    def train(self, iterations):
        for i in range(1, iterations+1):
            print("Loop:", i)
            print("Propagating...")
            self.propagate()
            print("Searching...")
            self.random_search()


if __name__ == "__main__":
    # Start a timer
    tic = time.process_time()

    # Calculate Map
    patch_match = PatchMatch()
    patch_match.train(4)
    patch_match.visualize()

    # Display compute time.
    toc = time.process_time()
    elapsed = toc - tic
    print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))
