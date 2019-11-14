import time
import classicStereo
import patchMatch
import diffPatchMatch
import os


def test_classic(left_path, right_path, outfile, disparity_range):
    tic = time.process_time()
    classicStereo.main(left_path=left_path, right_path=right_path,
                       outfile=outfile, disparity_range=disparity_range)
    toc = time.process_time()
    file = outfile.split("/")[-1]
    print("Classic Stereo Runtime (" + file + "): " + str(toc-tic))


def test_patch_match(left_path, right_path, outfile, disparity_range, iterations):
    tic = time.process_time()
    patchMatch.main(left_path=left_path, right_path=right_path,
                    outfile=outfile, disparity_range=disparity_range,
                    iterations=iterations)
    toc = time.process_time()
    file = outfile.split("/")[-1]
    print("PatchMatch Runtime (" + str(iterations) + " iter) (" + file + "): " + str(toc-tic))


def test_diff_patch_match(left_path, right_path, outfile, disparity_range, iterations):
    tic = time.process_time()
    diffPatchMatch.main(left_path=left_path, right_path=right_path,
                        outfile=outfile, disparity_range=disparity_range,
                        iterations=iterations)
    toc = time.process_time()
    file = outfile.split("/")[-1]
    print("Differential PatchMatch Runtime (" + str(iterations) + " iter) (" + file + "): " + str(toc-tic))


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
            outfile = "output/" + scene + "/diffpatchmatch1/" + left
            test_diff_patch_match(left_path=left_path, right_path=right_path,
                                  outfile=outfile, disparity_range=disparity_range, iterations=1)
            outfile = "output/" + scene + "/diffpatchmatch2/" + left
            test_diff_patch_match(left_path=left_path, right_path=right_path,
                                  outfile=outfile, disparity_range=disparity_range, iterations=2)
            outfile = "output/" + scene + "/diffpatchmatch5/" + left
            test_diff_patch_match(left_path=left_path, right_path=right_path,
                                  outfile=outfile, disparity_range=disparity_range, iterations=5)
            outfile = "output/" + scene + "/classic/" + left
            test_classic(left_path=left_path, right_path=right_path,
                         outfile=outfile, disparity_range=disparity_range)
