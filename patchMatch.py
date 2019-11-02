import matplotlib.pyplot as plt
import numpy as np

INTMAX = 99999999


def get_patches(image):
    height, width, depth = image.shape
    patches = np.zeros((height-2, width-2, 3, 3, depth))
    for x in range(1, height-1):
        for y in range(1, width-1):
            patches[x-1][y-1] = image[x-1:x+2, y-1:y+2, :]
    return patches


class PatchMatch:
    def __init__(self, left_path='left.png', right_path='left.png', output_path='patchMatch.png'):
        self.output_path = output_path

        self.left_img = plt.imread(left_path)
        self.right_img = plt.imread(right_path)

        # self.left_grey = np.mean(self.left_img, axis=2)
        # self.right_grey = np.mean(self.right_img, axis=2)
        self.imgHeight, self.imgWidth, self.imgDepth = np.shape(self.left_img)

        self.left_patches = get_patches(self.left_img)
        self.right_patches = get_patches(self.right_img)
        self.offsets = self.initialize_offsets()
        self.new_offsets = np.empty_like(self.offsets)

    def initialize_offsets(self):
        offsets = np.zeros((self.imgHeight-2, self.imgWidth-2, 2), dtype=int)
        for x in range(self.imgHeight-2):
            for y in range(self.imgWidth-2):
                x_rand = np.random.randint(0, self.imgHeight-2)
                y_rand = np.random.randint(0, self.imgWidth-2)
                offsets[x][y] = np.array([x_rand-x, y_rand-y])
        return offsets

    def patch_distance_error(self, x, y, offset):
        x_offset, y_offset = offset
        if x+x_offset < 0 or \
                x+x_offset >= self.right_patches.shape[0] or \
                y+y_offset < 0 or \
                y+y_offset >= self.right_patches.shape[1]:
            return INTMAX
        left_patch = self.left_patches[x][y]
        right_patch = self.right_patches[x+x_offset][y+y_offset]
        distance_error = np.sum(np.abs(left_patch-right_patch))
        return distance_error

    def propagate_patch(self, x, y):
        offset_args = [self.offsets[x][y], self.offsets[x-1][y], self.offsets[x][y-1]]
        distance_errors = [self.patch_distance_error(x, y, offset_arg) for offset_arg in offset_args]
        best_offset = offset_args[int(np.argmin(distance_errors))]
        self.new_offsets[x][y] = best_offset

    def propagate(self):
        for x in np.arange(1, np.shape(self.left_patches)[0]):
            for y in np.arange(1, np.shape(self.left_patches)[1]):
                self.propagate_patch(x, y)
        self.offsets = self.new_offsets

    def random_search(self):
        x_radius, y_radius = int((self.imgHeight-2)/2), int((self.imgWidth-2)/2)
        while x_radius > 0 and y_radius > 0:
            for x, row in enumerate(self.offsets):
                for y, offset in enumerate(row):
                    curr_x, curr_y = offset[0], offset[1]
                    new_x = np.random.randint(max(0, x+curr_x-x_radius), min(self.imgHeight-2, x+curr_x+x_radius))-x
                    new_y = np.random.randint(max(0, y+curr_y-y_radius), min(self.imgWidth-2, y+curr_y+y_radius))-y
                    if self.patch_distance_error(x, y, offset) > self.patch_distance_error(x, y, [new_x, new_y]):
                        offset[0], offset[1] = new_x, new_y
            x_radius, y_radius = int(x_radius/2), int(y_radius/2)

    def visualize(self):
        print("Displaying disparity map...")
        plt.imshow(self.offsets[:, :, 1], cmap="inferno")
        plt.savefig('patchMatch.png')

    def train(self, iterations):
        for i in range(1, iterations+1):
            print("Loop:", i)
            print("Propagating...")
            self.propagate()
            print("Searching...")
            self.random_search()


if __name__ == "__main__":
    patch_match = PatchMatch()
    patch_match.train(1)
    patch_match.visualize()
