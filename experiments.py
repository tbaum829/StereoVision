import time
from classicStereo import ClassicStereo
from patchMatch import PatchMatch


def test_classic():
    tic = time.process_time()
    classic_Stereo = ClassicStereo(left_path="source/left/floating.png", right_path="source/right/floating.png")
    classic_Stereo.train()
    toc = time.process_time()
    classic_Stereo.visualize()
    print("Classic Stereo Runtime:", toc-tic)


def test_patch_match(iterations):
    tic = time.process_time()
    patch_match = PatchMatch(left_path="source/left/floating.png", right_path="source/right/floating.png",
                             outfile="output/patchMatch" + str(iterations) + ".png")
    patch_match.train(iterations)
    toc = time.process_time()
    patch_match.visualize()
    print("PatchMatch Runtime ({} iter):".format(iterations), toc-tic)


if __name__ == "__main__":
    test_patch_match(1)
    test_patch_match(2)
    test_patch_match(5)
    test_classic()
