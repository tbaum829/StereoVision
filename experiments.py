import time
from classicStereo import ClassicStereo
from patchMatch import PatchMatch
import os


def test_classic(left_path, right_path, outfile, disparity_range):
    tic = time.process_time()
    classic_Stereo = ClassicStereo(left_path=left_path, right_path=right_path,
                                   outfile=outfile, disparity_range=disparity_range)
    classic_Stereo.train()
    toc = time.process_time()
    classic_Stereo.visualize()
    scene = outfile.split("/")[1]
    print("Classic Stereo Runtime ( " + scene + " ): " + str(toc-tic))


def test_patch_match(left_path, right_path, outfile, disparity_range, iterations):
    tic = time.process_time()
    patch_match = PatchMatch(left_path=left_path, right_path=right_path,
                             outfile=outfile, disparity_range=disparity_range)
    patch_match.train(iterations)
    toc = time.process_time()
    patch_match.visualize()
    scene = outfile.split("/")[1]
    print("PatchMatch Runtime ( " + str(iterations) + " iter) ( " + scene + " ): " + str(toc-tic))


if __name__ == "__main__":
    source_path = "source/"
    for scene in os.listdir(source_path):
        disparity_range = 200 if scene == "driving" else 100
        left_dir = source_path + scene + "/left/"
        right_dir = source_path + scene + "/right/"
        for left, right in zip(os.listdir(left_dir), os.listdir(right_dir)):
            left_path = left_dir + left
            right_path = right_dir + right
            outfile = "output/" + scene + "/patchmatch1/" + left
            test_patch_match(left_path=left_path, right_path=right_path,
                             outfile=outfile, disparity_range=disparity_range, iterations=1)
            outfile = "output/" + scene + "/patchmatch2/" + left
            test_patch_match(left_path=left_path, right_path=right_path,
                             outfile=outfile, disparity_range=disparity_range, iterations=2)
            outfile = "output/" + scene + "/patchmatch5/" + left
            test_patch_match(left_path=left_path, right_path=right_path,
                             outfile=outfile, disparity_range=disparity_range, iterations=5)
            outfile = "output/" + scene + "/classic/" + left
            test_classic(left_path=left_path, right_path=right_path,
                         outfile=outfile, disparity_range=disparity_range)
