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
    def __init__(self, left_path='left.png', right_path='right.png', output_path='patchMatch.png'):
        self.output_path = output_path

        self.left_img = plt.imread(left_path)
        self.right_img = plt.imread(right_path)

        self.left_grey = np.mean(self.left_img, axis=2)
        self.right_grey = np.mean(self.right_img, axis=2)
        self.imgHeight, self.imgWidth = np.shape(self.right_grey)

        self.left_patches = get_patches(self.left_grey)
        self.right_patches = get_patches(self.right_grey)
        self.offsets = self.initialize_offsets()
        self.visualize()
        self.new_offsets = self.offsets.copy()

    def initialize_offsets(self):
        height, width, _, _ = self.right_patches.shape
        offsets = np.zeros((height, width, 2), dtype=int)
        for x in range(200, 301):
            for y in range(125, 251):
                offsets[x][y][0] += np.random.randint(0, high=height-1)-x
                offsets[x][y][1] += np.random.randint(0, high=width-1)-y
        return offsets

    def patch_distance_error(self, x, y, offset):
        x_offset, y_offset = offset
        if x+x_offset < 0 or \
                x+x_offset >= self.right_patches.shape[0] or \
                y+y_offset < 0 or \
                y+y_offset >= self.right_patches.shape[1]:
            return INTMAX
        right_patch = self.right_patches[x][y]
        left_patch = self.left_patches[x+x_offset][y+y_offset]
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
        for x in np.arange(1, np.shape(self.right_patches)[0]):
            for y in np.arange(1, np.shape(self.right_patches)[1]):
                self.propagate_patch(x, y)

    def random_search(self):
        height, width, _, _ = self.right_patches.shape
        radius = [np.floor_divide(height, 2), np.floor_divide(width, 2)]
        while radius[0] > 0 and radius[1] > 0:
            for x, row in enumerate(self.offsets):
                x_low = max(0, x-radius[0])
                x_high = min(height, x+radius[0])
                for y, offset in enumerate(row[:-1]):
                    y_low = max(0, x-radius[1])
                    y_high = min(width, x+radius[1])
                    prob_arg = np.subtract(np.array([np.random.randint(x_low, high=x_high),
                                                     np.random.randint(y_low, high=y_high)]), np.array([x, y]))
                    if self.patch_distance_error(x, y, offset) > self.patch_distance_error(x, y, prob_arg):
                        self.offsets[x][y] = prob_arg
            radius = np.floor_divide(radius, 4)

    def visualize(self, outfile='patchMatch2.png'):
        plt.imshow(self.offsets[:, :, 1], cmap="inferno")
        plt.savefig(outfile)

    def train(self, iterations):
        for i in range(1, iterations+1):
            print("Loop:", i)
            print("Propagating...")
            self.propagate()
            self.visualize()
            print("Searching...")
            self.random_search()
            self.visualize()


if __name__ == "__main__":
    # Start a timer
    tic = time.process_time()

    # Calculate Map
    patch_match = PatchMatch()
    patch_match.train(100)
    patch_match.visualize()

    # Display compute time.
    toc = time.process_time()
    elapsed = toc - tic
    print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))
