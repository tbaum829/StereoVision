import matplotlib.pyplot as plt
import numpy as np
import time

INTMAX = 99999999


def get_patches(image):
    height, width = image.shape
    patches = np.zeros((height-2, width-2, 5, 5))
    for x in range(2, height-2):
        for y in range(2, width-2):
            patches[x-1][y-1] = image[x-2:x+3, y-2:y+3]
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
        self.new_offsets = self.offsets.copy()

    def initialize_offsets(self):
        offsets = np.zeros((self.right_patches.shape[0], self.right_patches.shape[1]), dtype=int).T
        for i, row in enumerate(offsets[:-1]):
            row += np.random.randint(0, high=min(self.right_patches.shape[1]-i-1, 50), size=row.shape)
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
                       self.offsets[x][y-1],
                       self.offsets[x+1][y],
                       self.offsets[x][y+1]]
        distance_errors = [self.patch_distance_error(x, y, offset_arg) for offset_arg in offset_args]
        best_offset = offset_args[int(np.argmin(distance_errors))]
        self.new_offsets[x][y] = best_offset

    def propagate(self):
        for x in np.arange(1, np.shape(self.right_patches)[0]-1):
            for y in np.arange(1, np.shape(self.right_patches)[1]-1):
                self.propagate_patch(x, y)
        self.offsets = self.new_offsets.copy()

    def random_search(self):
        radius = 50
        while radius > 0:
            for x, row in enumerate(self.offsets):
                for y, offset in enumerate(row[:-1]):
                    new_offset = np.random.randint(max(0, offset-radius),
                                                   high=min(self.right_patches.shape[1]-y-1, 50, offset+radius))
                    if self.patch_distance_error(x, y, offset) > self.patch_distance_error(x, y, new_offset):
                        self.offsets[x][y] = new_offset
            radius = int(radius/4)

    def visualize(self, outfile='patchMatch.png'):
        plt.imshow(np.abs(self.offsets), cmap="inferno")
        plt.savefig('patchMatch.png')

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
