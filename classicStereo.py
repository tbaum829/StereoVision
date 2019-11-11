import numpy as np
import time
from disparityAlg import DisparityAlg

INTMIN = -99999999


class ClassicStereo(DisparityAlg):
    def __init__(self, left_path, right_path, outfile="output/classicStereo.png"):
        super().__init__(left_path=left_path, right_path=right_path, outfile=outfile)
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


if __name__ == "__main__":
    tic = time.process_time()
    classic_stereo = ClassicStereo(left_path="source/left/floating.png", right_path="source/right/floating.png")
    classic_stereo.train()
    toc = time.process_time()
    classic_stereo.visualize()
    print("Runtime:", toc-tic)
