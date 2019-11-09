import matplotlib.pyplot as plt
import numpy as np
import time

INTMAX = 99999999


def get_patches(image, patch_width):
    height, width, depth = image.shape
    patches = np.zeros((height-patch_width*2, width-patch_width*2, patch_width*2+1, patch_width*2+1, depth))
    for x in range(patch_width, height-patch_width):
        for y in range(patch_width, width-patch_width):
            patches[x-patch_width][y-patch_width] = image[x-patch_width:x+patch_width+1, y-patch_width:y+patch_width+1]
    return patches


class PatchMatch:
    def __init__(self, left_path='left.png', right_path='right.png'):
        self.patch_width = 3

        self.left_img = plt.imread(left_path)
        self.right_img = plt.imread(right_path)

        self.left_grey = np.mean(self.left_img, axis=2)
        self.right_grey = np.mean(self.right_img, axis=2)
        self.imgHeight, self.imgWidth = np.shape(self.right_grey)

        self.left_patches = get_patches(self.left_img, self.patch_width)
        self.right_patches = get_patches(self.right_img, self.patch_width)
        self.offsets = self.initialize_offsets()
        self.best_distances = self.initialize_distances()

    def initialize_offsets(self):
        offsets = np.zeros((self.right_patches.shape[0], self.right_patches.shape[1]), dtype=int).T
        for i, row in enumerate(offsets[:-70]):
            random_array = np.random.randint(0, high=self.right_patches.shape[1]-i-1, size=row.shape)
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
            return -INTMAX
        right_patch = self.right_patches[x][y]
        left_patch = self.left_patches[x][y+offset]
        distance_error = -np.log(np.sum(np.square(right_patch-left_patch)))
        return distance_error

    def propagate_down(self, x, y):
        current_offset = self.offsets[x][y]
        current_distance = self.best_distances[x][y]
        above_offset = self.offsets[x-1][y]
        above_distance = self.patch_distance_error(x, y, above_offset)  # - 0.05
        left_offset = self.offsets[x][y-1]
        left_distance = self.patch_distance_error(x, y, left_offset)  # - 0.05
        distance_errors = [current_distance, above_distance, left_distance]
        offset_args = [current_offset, above_offset, left_offset]
        best_distance = min(distance_errors)
        best_offset = offset_args[int(np.argmax(distance_errors))]
        self.offsets[x][y] = best_offset
        self.best_distances[x][y] = best_distance

    def propagate_up(self, x, y):
        current_offset = self.offsets[x][y]
        current_distance = self.best_distances[x][y]
        below_offset = self.offsets[x+1][y]
        below_distance = self.patch_distance_error(x, y, below_offset)
        right_offset = self.offsets[x][y+1]
        right_distance = self.patch_distance_error(x, y, right_offset)
        distance_errors = [current_distance, below_distance, right_distance]
        offset_args = [current_offset, below_offset, right_offset]
        best_distance = min(distance_errors)
        best_offset = offset_args[int(np.argmax(distance_errors))]
        self.offsets[x][y] = best_offset
        self.best_distances[x][y] = best_distance

    def random_search(self, x, y, radius):
        current_distance = self.best_distances[x][y]
        new_offset = np.random.randint(max(0, self.offsets[x][y]-radius),
                                       high=min(self.right_patches.shape[1]-y-1, self.offsets[x][y]+radius))
        new_distance = self.patch_distance_error(x, y, new_offset)
        if current_distance < new_distance:
            self.offsets[x][y] = new_offset
            self.best_distances[x][y] = new_distance

    def visualize(self, outfile='patchMatch.png'):
        for x, row in enumerate(self.offsets):
            for y, offset in enumerate(row):
                if offset > 50:
                    self.offsets[x][y] = 50
        plt.imshow(self.offsets, cmap="inferno")
        plt.savefig(outfile)

    def train(self, iterations):
        radius = 50
        for i in range(1, iterations+1):
            print("Iteration", i)
            if i % 2 == 1:
                for x, row in enumerate(self.offsets):
                    for y, offset in enumerate(row[:-70]):
                        self.propagate_down(x, y)
                        self.random_search(x, y, 100)
                        self.random_search(x, y, 50)
                        self.random_search(x, y, 20)
                    if x % 10 == 0:
                        print("  Image row {0:d} / {1:d} {2:.2f}%".format(x, self.imgHeight,
                                                                          (x / self.imgHeight) * 100))
            else:
                for x, row in enumerate(self.offsets):
                    for y, offset in enumerate(row[:-70]):
                        self.propagate_up(self.offsets.shape[0]-2-x, self.offsets.shape[1]-71-y)
                        self.random_search(self.offsets.shape[0]-2-x, self.offsets.shape[1]-71-y, 100)
                        self.random_search(self.offsets.shape[0]-2-x, self.offsets.shape[1]-71-y, 50)
                        self.random_search(self.offsets.shape[0]-2-x, self.offsets.shape[1]-71-y, 20)
                    if x % 10 == 0:
                        print("  Image row {0:d} / {1:d} {2:.2f}%".format(x, self.imgHeight,
                                                                          (x / self.imgHeight) * 100))
            radius = np.floor_divide(radius, 2)
            if radius <= 0:
                break


if __name__ == "__main__":
    # Start a timer
    tic = time.process_time()

    # Calculate Map
    patch_match = PatchMatch()
    # patch_match = PatchMatch('leftCar.png', 'rightCar.png')
    patch_match.train(2)

    # Display compute time.
    toc = time.process_time()

    patch_match.visualize()
    # patch_match.visualize('carOut.png')
    elapsed = toc - tic
    print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))
