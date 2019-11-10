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
        self.best_distances = self.initialize_distances()

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
        offsets = np.random.randint(0, high=self.disparity_range, size=(self.imgHeight, self.imgWidth), dtype=int)
        return offsets

    def initialize_distances(self):
        best_distances = np.zeros((self.imgHeight, self.imgWidth))
        for x, row in enumerate(self.offsets):
            for y, offset in enumerate(row):
                best_distances[x][y] = self.patch_distance_error(x, y, offset)
        return best_distances

    def patch_distance_error(self, x, y, offset):
        if y+offset >= self.imgWidth:
            return INTMIN
        right_patch = self.right_patches[x][y]
        left_patch = self.left_patches[x][y+offset]
        distance_error = -np.log(np.sum(np.square(right_patch-left_patch)))
        return distance_error

    def propagate_down(self, x, y):
        current_offset = self.offsets[x][y]
        current_distance = self.best_distances[x][y]
        above_offset = self.offsets[x-1][y]
        above_distance = self.patch_distance_error(x, y, above_offset)
        left_offset = self.offsets[x][y-1]
        left_distance = self.patch_distance_error(x, y, left_offset)
        distance_errors = [current_distance, above_distance, left_distance]
        offset_args = [current_offset, above_offset, left_offset]
        best_index = int(np.argmax(distance_errors))
        self.offsets[x][y] = offset_args[best_index]
        self.best_distances[x][y] = distance_errors[best_index]

    def propagate_up(self, x, y):
        current_offset = self.offsets[x][y]
        current_distance = self.best_distances[x][y]
        below_offset = self.offsets[x+1][y]
        below_distance = self.patch_distance_error(x, y, below_offset)
        right_offset = self.offsets[x][y+1]
        right_distance = self.patch_distance_error(x, y, right_offset)
        distance_errors = [current_distance, below_distance, right_distance]
        offset_args = [current_offset, below_offset, right_offset]
        best_index = int(np.argmax(distance_errors))
        self.offsets[x][y] = offset_args[best_index]
        self.best_distances[x][y] = distance_errors[best_index]

    def random_search(self, x, y):
        current_distance = self.best_distances[x][y]
        new_offset = np.random.randint(0, self.disparity_range)
        new_distance = self.patch_distance_error(x, y, new_offset)
        if current_distance < new_distance:
            self.offsets[x][y] = new_offset
            self.best_distances[x][y] = new_distance

    def visualize(self, outfile='patchMatch.png'):
        plt.imshow(self.offsets[:, :-self.disparity_range], cmap="inferno")
        plt.savefig(outfile)

    def train(self, iterations):
        for i in range(1, iterations+1):
            print("Iteration", i)
            if i % 2 == 1:
                x_range = range(1, self.imgHeight, 1)
                y_range = range(1, self.imgWidth-self.disparity_range, 1)
                propagate_function = self.propagate_down
            else:
                x_range = range(self.imgHeight-2, 0, -1)
                y_range = range(self.imgWidth-self.disparity_range-2, 0, -1)
                propagate_function = self.propagate_up
            for x in x_range:
                for y in y_range:
                    propagate_function(x, y)
                    self.random_search(x, y)
                if x % 10 == 0:
                    print("  Image row {0:d} / {1:d}".format(x, self.imgHeight))


if __name__ == "__main__":
    # Start a timer
    tic = time.process_time()

    # Calculate Map
    patch_match = PatchMatch()
    patch_match.train(2)

    # Display compute time.
    toc = time.process_time()

    patch_match.visualize()
    elapsed = toc - tic
    print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))
