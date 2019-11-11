import numpy as np
import time
from disparityAlg import DisparityAlg

INTMIN = -99999999


class PatchMatch(DisparityAlg):
    def __init__(self, left_path, right_path, outfile='output/patchMatch.png'):
        super().__init__(left_path=left_path, right_path=right_path, outfile=outfile)
        self.offsets = self.initialize_offsets()
        self.best_distances = self.initialize_distances()

    def initialize_offsets(self):
        offsets = np.random.randint(0, high=self.disparity_range, size=(self.height, self.width), dtype=int)
        return offsets

    def initialize_distances(self):
        best_distances = np.zeros((self.height, self.width))
        for x, row in enumerate(self.offsets):
            for y, offset in enumerate(row):
                best_distances[x][y] = self.patch_distance_error(x, y, offset)
        return best_distances

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

    def random_search(self, x, y, offset):
        current_distance = self.best_distances[x][y]
        new_distance = self.patch_distance_error(x, y, offset)
        if current_distance < new_distance:
            self.offsets[x][y] = offset
            self.best_distances[x][y] = new_distance

    def train(self, iterations):
        for i in range(1, iterations+1):
            if i % 2 == 1:
                x_range = range(1, self.height, 1)
                y_range = range(1, self.width, 1)
                propagate_function = self.propagate_down
            else:
                x_range = range(self.height-2, 0, -1)
                y_range = range(self.width-2, 0, -1)
                propagate_function = self.propagate_up
            random_offsets = self.initialize_offsets()
            for x in x_range:
                for y in y_range:
                    propagate_function(x, y)
                    self.random_search(x, y, random_offsets[x][y])


if __name__ == "__main__":
    tic = time.process_time()

    patch_match = PatchMatch(left_path="source/left/floating.png", right_path="source/right/floating.png")
    patch_match.train(5)

    toc = time.process_time()

    patch_match.visualize()
    print("Runtime:", toc-tic)
