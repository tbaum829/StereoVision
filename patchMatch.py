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
        self.best_distances = self.initialize_distances()

    def initialize_offsets(self):
        offsets = np.zeros((self.right_patches.shape[0], self.right_patches.shape[1]), dtype=int).T
        for i, row in enumerate(offsets[:-70]):
            random_array = np.random.randint(0, high=min(self.right_patches.shape[1]-i-1, 50), size=row.shape)
            row += random_array
        return offsets.T

    def initialize_distances(self):
        best_distances = np.zeros((self.offsets.shape[0], self.offsets.shape[1]))
        for x, row in enumerate(self.offsets):
            for y, offset in enumerate(row):
                best_distances[x][y] = self.patch_distance_error(x, y, offset)
        return best_distances

    def patch_distance_error(self, x, y, offset):
        if y+offset >= self.right_patches.shape[1]:
            return INTMAX
        right_patch = self.right_patches[x][y]
        left_patch = self.left_patches[x][y+offset]
        distance_error = np.sum(np.square(right_patch-left_patch))
        return distance_error

    def propagate_patch(self, x, y):
        current_offset = self.offsets[x][y]
        current_distance = self.best_distances[x][y]
        above_offset = self.offsets[x-1][y]
        above_distance = self.patch_distance_error(x, y, above_offset)
        left_offset = self.offsets[x][y-1]
        left_distance = self.patch_distance_error(x, y, left_offset)
        distance_errors = [current_distance, above_distance, left_distance]
        offset_args = [current_offset, above_offset, left_offset]
        best_distance = min(distance_errors)
        best_offset = offset_args[int(np.argmin(distance_errors))]
        self.offsets[x][y] = best_offset
        self.best_distances[x][y] = best_distance

    def propagate(self):
        for x in np.arange(1, np.shape(self.right_patches)[0]):
            for y in np.arange(1, np.shape(self.right_patches)[1]-70):
                self.propagate_patch(x, y)

    def random_search(self):
        radius = 50
        while radius > 0:
            for x, row in enumerate(self.offsets):
                for y, offset in enumerate(row[:-70]):
                    current_distance = self.best_distances[x][y]
                    if current_distance < 0.2:
                        continue
                    new_offset = np.random.randint(max(0, offset-radius),
                                                   high=min(self.right_patches.shape[1]-y-1, 50, offset+radius))
                    new_distance = self.patch_distance_error(x, y, new_offset)
                    if current_distance > new_distance:
                        self.offsets[x][y] = new_offset
                        self.best_distances[x][y] = new_distance
            radius = int(radius/4)

    def visualize(self, outfile='patchMatch.png'):
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
    patch_match.train(1)

    # Display compute time.
    toc = time.process_time()

    patch_match.visualize()
    elapsed = toc - tic
    print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))
