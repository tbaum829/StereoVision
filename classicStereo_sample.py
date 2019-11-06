import matplotlib.pyplot as plt
import numpy as np
import time

# Start a timer
tic = time.process_time()

left = plt.imread('left.png')
right = plt.imread('right.png')
leftI = np.mean(left, axis=2)
rightI = np.mean(right, axis=2)
imgHeight, imgWidth = np.shape(leftI)
DbasicSubpixel = np.zeros((imgHeight-6, imgWidth-6), dtype='int')
halfBlockSize = 3
blockSize = 2 * halfBlockSize + 1

for m in range(201, 301):
    for n in range(125, 251):
        right_block = rightI[m-3:m+3, n-3:n+3]
        blockDiffs = np.zeros((imgHeight-6, imgWidth-6), dtype='int')
        for x in range(3, imgHeight-3):
            for y in range(3, imgWidth-3):
                left_block = leftI[x-3:x+3, y-3:y+3]
                blockDiffs[x-3][y-3] = np.sum(np.abs((right_block - left_block)))
        DbasicSubpixel[m-3, n-3] = int(np.argmin(blockDiffs) / imgWidth)
        print(m, n)

# Display compute time.
toc = time.process_time()
elapsed = toc - tic
print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))

# Display the Disparity Map
plt.imshow(DbasicSubpixel, cmap="inferno")
plt.savefig('classicStereo_sample.png', )
