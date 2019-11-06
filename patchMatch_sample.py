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
        offsets = np.zeros((self.right_patches.shape[0], self.right_patches.shape[1], 2), dtype=int)
        for x in range(200, 301):
            for y in range(125, 251):
                x_offset = np.random.randint(0, high=self.right_patches.shape[0]-1)-x
                y_offset = np.random.randint(0, high=self.right_patches.shape[1]-1)-y
                offsets[x][y] += [x_offset, y_offset]
        return offsets

    def initialize_distances(self):
        best_distances = np.zeros((self.offsets.shape[0], self.offsets.shape[1]))
        for x in range(200, 301):
            for y in range(125, 251):
                offset = self.offsets[x][y]
                best_distances[x][y] = self.patch_distance_error(x, y, offset)
        return best_distances

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
        for x in np.arange(201, 301):
            for y in np.arange(126, 250):
                self.propagate_patch(x, y)

    def random_search(self):
        radius = np.floor_divide(np.array([self.right_patches.shape[0], self.right_patches.shape[1]]), 2)
        while radius[0] > 0 and radius[1] > 0:
            for x in np.arange(200, 301):
                for y in np.arange(125, 251):
                    for i in range(3):
                        x_offset = np.random.randint(max(0, x-radius[0]),
                                                     high=min(self.right_patches.shape[0]-1, x+radius[0]))-x
                        y_offset = np.random.randint(max(0, y-radius[1]),
                                                     high=min(self.right_patches.shape[1]-1, y+radius[1]))-y
                        new_offset = np.array([x_offset, y_offset])
                        if self.patch_distance_error(x, y, self.offsets[x][y]) > \
                                self.patch_distance_error(x, y, new_offset):
                            self.offsets[x][y] = new_offset
            radius = np.floor_divide(radius, 2)

    def visualize(self, outfile='patchMatch_sample.png'):
        plt.imshow(self.offsets[:,:,1], cmap="inferno")
        plt.savefig(outfile)

    def train(self, iterations):
        for i in range(1, iterations+1):
            print("Loop:", i)
            print("Propagating...")
            self.propagate()
            print("Searching...")
            self.random_search()
            self.visualize()


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
