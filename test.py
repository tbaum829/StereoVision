import time
from classicStereo import ClassicStereo
from patchMatch import PatchMatch


def test_classic():
    tic = time.process_time()
    classic_Stereo = ClassicStereo()
    classic_Stereo.train()
    toc = time.process_time()
    classic_Stereo.visualize()
    print("Classic Stereo Runtime:", toc-tic)


def test_patch_match(iterations):
    tic = time.process_time()
    patch_match = PatchMatch(outfile="patchMatch" + str(iterations) + ".png")
    patch_match.train(iterations)
    toc = time.process_time()
    patch_match.visualize()
    print("PatchMatch Runtime ({} iter):".format(iterations), toc-tic)


if __name__ == "__main__":
    test_classic()
    test_patch_match(1)
    test_patch_match(2)
    test_patch_match(5)
