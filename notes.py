import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import stats

# Read the two images
left = plt.imread('left.png')
right = plt.imread('right.png')

# ===================================
#       Display Composite Image
# ===================================

# Create a composite image out of the two stereo images.
leftRed = left[:, :, :2]

# Take the green and blue color channels from the right image.
rightGreenBlue = right[:, :, 2:]

# Combine the above channels into a single composite image using the 'cat'
# function, which concatenates the matrices along dimension '3'.
composite = np.concatenate((leftRed, rightGreenBlue), axis=2)

# Show the composite image.
# plt.imshow(composite)
# plt.show()


# ====================================
#        Basic Block Matching
# ====================================

# Start a timer
tic = time.process_time()

# Convert the images from RGB to grayscale by
# averaging the three color channels.
leftI = np.mean(left, axis=2)
rightI = np.mean(right, axis=2)

# DbasicSubpixel will hold the result of the block matching.
# The values will be 'single' precision (32-bit) floating point.
DbasicSubpixel = np.zeros(np.shape(leftI), dtype='float32')

halfBlockSize = 3
blockSize = 2 * halfBlockSize + 1

# Get the image dimensions.
imgHeight, imgWidth = np.shape(leftI)

# For each row 'm' of pixels in the image...
for m in range(imgHeight):

    corners = np.zeros((5, 2, 2), dtype="int")

    corners[0][0][0] = corners[1][0][0] = max(0, m - halfBlockSize + 1)
    corners[0][0][1] = corners[1][0][1] = min(imgHeight, m + 2)
    corners[2][0][0] = corners[3][0][0] = max(0, m - 1)
    corners[2][0][1] = corners[3][0][1] = min(imgHeight, m + halfBlockSize)
    corners[2][0][0] = max(0, m - 2)
    corners[2][0][1] = min(imgHeight, m + halfBlockSize)

    # For each column 'n' of pixels in the image...
    for n in range(imgWidth - halfBlockSize):

        corners[0][1][0] = corners[2][1][0] = max(0, n - halfBlockSize + 1)
        corners[0][1][1] = corners[2][1][1] = min(imgHeight, n + 2)
        corners[1][1][0] = corners[3][1][0] = max(0, n - 1)
        corners[1][1][1] = corners[3][1][1] = min(imgWidth, n + halfBlockSize)

        dwidth = min(65, imgWidth - halfBlockSize - n)
        orgWidth = dwidth
        wing = int((dwidth-1)/2)
        centers = np.zeros(4)
        ranges = np.tile(np.arange(0, dwidth), (4, 1))

        # Select the block from the right image to use as the template.
        template0 = rightI[corners[0][0][0]:corners[0][0][1], corners[0][1][0]:corners[0][1][1]]
        template1 = rightI[corners[1][0][0]:corners[1][0][1], corners[1][1][0]:corners[1][1][1]]
        template2 = rightI[corners[2][0][0]:corners[2][0][1], corners[2][1][0]:corners[2][1][1]]
        template3 = rightI[corners[3][0][0]:corners[3][0][1], corners[3][1][0]:corners[3][1][1]]

        costs = np.zeros(4)

        while dwidth > 1:
            for i in range(4):
                centers[i] = np.random.choice(ranges[i])
                while centers[i] in centers[:i] and dwidth > 4:
                    centers[i] = np.random.choice(ranges[i])

            block0 = leftI[corners[0][0][0]:corners[0][0][1],
                     int(corners[0][1][0] + centers[0]):int(corners[0][1][1] + centers[0])]
            block1 = leftI[corners[1][0][0]:corners[1][0][1],
                     int(corners[1][1][0] + centers[1]):int(corners[1][1][1] + centers[1])]
            block2 = leftI[corners[2][0][0]:corners[2][0][1],
                     int(corners[2][1][0] + centers[2]):int(corners[2][1][1] + centers[2])]
            block3 = leftI[corners[3][0][0]:corners[3][0][1],
                     int(corners[3][1][0] + centers[3]):int(corners[3][1][1] + centers[3])]

            costs[0] = np.sum(np.square(template0 - block0))
            costs[1] = np.sum(np.square(template1 - block1))
            costs[2] = np.sum(np.square(template2 - block2))
            costs[3] = np.sum(np.square(template3 - block3))

            bestCenter = centers[np.argmin(costs)]

            newBlock0 = leftI[corners[0][0][0]:corners[0][0][1],
                        int(corners[0][1][0] + bestCenter):int(corners[0][1][1] + bestCenter)]
            newBlock1 = leftI[corners[1][0][0]:corners[1][0][1],
                        int(corners[1][1][0] + bestCenter):int(corners[1][1][1] + bestCenter)]
            newBlock2 = leftI[corners[2][0][0]:corners[2][0][1],
                        int(corners[2][1][0] + bestCenter):int(corners[2][1][1] + bestCenter)]
            newBlock3 = leftI[corners[3][0][0]:corners[3][0][1],
                        int(corners[3][1][0] + bestCenter):int(corners[3][1][1] + bestCenter)]

            if np.sum(np.square(template0 - newBlock0)) < costs[0]:
                centers[0] = bestCenter
            if np.sum(np.square(template1 - newBlock1)) < costs[1]:
                centers[1] = bestCenter
            if np.sum(np.square(template2 - newBlock2)) < costs[2]:
                centers[2] = bestCenter
            if np.sum(np.square(template3 - newBlock3)) < costs[3]:
                centers[3] = bestCenter

            dwidth = wing + 1
            if dwidth <= 0 or np.all(centers == centers[0]):
                break
            wing = int((dwidth-1)/2)

            newRanges = np.zeros((4, int(dwidth)))
            if centers[0] - wing < 0:
                newRanges[0] = np.arange(0, dwidth)
            elif centers[0] - wing + dwidth > orgWidth:
                newRanges[0] = np.arange(orgWidth-dwidth, orgWidth)
            else:
                newRanges[0] = np.arange(centers[0] - wing, centers[0] - wing + dwidth)

            if centers[1] - wing < 0:
                newRanges[1] = np.arange(0, dwidth)
            elif centers[1] - wing + dwidth > orgWidth:
                newRanges[1] = np.arange(orgWidth-dwidth, orgWidth)
            else:
                newRanges[1] = np.arange(centers[1] - wing, centers[1] - wing + dwidth)

            if centers[2] - wing < 0:
                newRanges[2] = np.arange(0, dwidth)
            elif centers[2] - wing + dwidth > orgWidth:
                newRanges[2] = np.arange(orgWidth-dwidth, orgWidth)
            else:
                newRanges[2] = np.arange(centers[2] - wing, centers[2] - wing + dwidth)

            if centers[3] - wing < 0:
                newRanges[3] = np.arange(0, dwidth)
            elif centers[3] - wing + dwidth > orgWidth:
                newRanges[3] = np.arange(orgWidth-dwidth, orgWidth)
            else:
                newRanges[3] = np.arange(centers[3] - wing, centers[3] - wing + dwidth)

            ranges = newRanges

        DbasicSubpixel[m, n] = stats.mode(centers).mode[0]

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

# Display the disparity map.
# Passing an empty matrix as the second argument tells imshow to take the
# minimum and maximum values of the data and map the data range to the
# display colors.
plt.imshow(DbasicSubpixel, cmap="inferno")

# Set the title to display.
plt.title("Basic block matching, Sub-px acc., Search right, Block size = " + str(blockSize))
plt.show()
