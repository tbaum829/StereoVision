import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import stats

# Read the two images
left = plt.imread('left.png')
right = plt.imread('right.png')


# ====================================
#        Basic Block Matching
# ====================================

# Start a timer
tic = time.process_time()

leftI = np.mean(left, axis=2)
rightI = np.mean(right, axis=2)

imgHeight, imgWidth = np.shape(leftI)

DbasicSubpixel = np.zeros((imgHeight, imgWidth, 3), dtype='float32')

halfBlockSize = 6
wing = 25
corners = np.zeros((5, 2, 2), dtype="int")

penalty = 0.05

# For each row 'm' of pixels in the image...
for m in range(halfBlockSize + 1, imgHeight - 2 - halfBlockSize):

    corners[0][0][0] = m - halfBlockSize
    corners[0][0][1] = m + halfBlockSize
    corners[1][0][0] = m - halfBlockSize + 3
    corners[1][0][1] = m + halfBlockSize - 3
    corners[2][0][0] = m - halfBlockSize - 3
    corners[2][0][1] = m + halfBlockSize + 3

    # For each column 'n' of pixels in the image...
    for n in range(halfBlockSize + 1, imgWidth - 2 - 51 - halfBlockSize):

        corners[0][1][0] = n - halfBlockSize
        corners[0][1][1] = n + halfBlockSize
        corners[1][1][0] = n - halfBlockSize
        corners[1][1][1] = n + halfBlockSize
        corners[2][1][0] = n - halfBlockSize
        corners[2][1][1] = n + halfBlockSize

        # Select the block from the right image to use as the template.
        template0 = rightI[corners[0][0][0]:corners[0][0][1], corners[0][1][0]:corners[0][1][1]]
        template1 = rightI[corners[1][0][0]:corners[1][0][1], corners[1][1][0]:corners[1][1][1]]
        template2 = rightI[corners[2][0][0]:corners[2][0][1], corners[2][1][0]:corners[2][1][1]]

        for x in range(3):
            wing = 25
            bound = 2*wing + 1
            centers = np.zeros(3)
            costs = np.zeros(3)

            range0 = np.arange(0, bound)
            range1 = np.arange(0, bound)
            range2 = np.arange(0, bound)
            ranges = [range0, range1, range2]

            while wing > 1:
                for i, rangeVal in enumerate(ranges):
                    centers[i] = np.random.choice(rangeVal)

                block0 = leftI[corners[0][0][0]:corners[0][0][1], int(corners[0][1][0] + centers[0]):int(corners[0][1][1] + centers[0])]
                block1 = leftI[corners[1][0][0]:corners[1][0][1], int(corners[1][1][0] + centers[1]):int(corners[1][1][1] + centers[1])]
                block2 = leftI[corners[2][0][0]:corners[2][0][1], int(corners[2][1][0] + centers[2]):int(corners[2][1][1] + centers[2])]

                costs[0] = np.sum(np.square(template0 - block0))
                costs[1] = np.sum(np.square(template1 - block1))
                costs[2] = np.sum(np.square(template2 - block2))

                bestCenter = centers[np.argmin(costs)]

                newBlock0 = leftI[corners[0][0][0]:corners[0][0][1], int(corners[0][1][0] + bestCenter):int(corners[0][1][1] + bestCenter)]
                newBlock1 = leftI[corners[1][0][0]:corners[1][0][1], int(corners[1][1][0] + bestCenter):int(corners[1][1][1] + bestCenter)]
                newBlock2 = leftI[corners[2][0][0]:corners[2][0][1], int(corners[2][1][0] + bestCenter):int(corners[2][1][1] + bestCenter)]

                if np.sum(np.square(template0 - newBlock0)) < costs[0] - penalty:
                    centers[0] = bestCenter
                if np.sum(np.square(template1 - newBlock1)) < costs[1] - penalty:
                    centers[1] = bestCenter
                if np.sum(np.square(template2 - newBlock2)) < costs[2] - penalty:
                    centers[2] = bestCenter

                wing = int(wing/2)
                if wing <= 1:
                    break

                for i in range(3):
                    ranges[i] = np.arange(max(0, centers[i]-wing), min(bound, centers[i] + wing + 1))

            DbasicSubpixel[m, n, x] = bestCenter

    # Update progress every 10th row.
    if m % 10 == 0:
        print("  Image row {0:d} / {1:d} {2:.2f}%".format(m, imgHeight, (m / imgHeight) * 100))

# Display compute time.
toc = time.process_time()
elapsed = toc - tic
print("Calculating disparity map took {0:.2f} min.\n".format(elapsed / 60.0))


# =========================================
#        Visualize Disparity Map
# =========================================

print("Displaying disparity map...")
plt.imshow(np.median(DbasicSubpixel, axis=2), cmap="inferno")
plt.savefig('notes.png')
