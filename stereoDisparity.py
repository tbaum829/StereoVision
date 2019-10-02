import matplotlib.pyplot as plt
import numpy as np
import time

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

# The disparity range defines how many pixels away from the block's location
# in the first image to search for a matching block in the other image.
# 50 appears to be a good value for the 450x375 images from the "Cones"
# dataset.
disparityRange = 50

# Define the size of the blocks for block matching.
halfBlockSize = 3
blockSize = 2 * halfBlockSize + 1

# Get the image dimensions.
imgHeight, imgWidth = np.shape(leftI)

# For each row 'm' of pixels in the image...
for m in range(imgHeight):

    # Set min / max row bounds for the template and blocks.
    # e.g., for the first row, minr = 0 and maxr = 3
    minr = max(0, m - halfBlockSize)
    maxr = min(imgHeight-1, m + halfBlockSize)

    # For each column 'n' of pixels in the image...
    for n in range(imgWidth):
        # Set the min / max column bounds for the template.
        # e.g., for the first column, minc = 1 and maxc = 4
        minc = max(0, n - halfBlockSize)
        maxc = min(imgWidth-1, n + halfBlockSize)

        # Define the search boundaries as offsets from the template location.
        # Limit the search so that we don't go outside of the image.
        # 'mind' is the the maximum number of pixels we can search to the left.
        # 'maxd' is the maximum number of pixels we can search to the right.
        #
        # In the "Cones" dataset, we only need to search to the right, so mind
        # is 0.
        #
        # For other images which require searching in both directions, set mind
        # as follows:
        #   mind = max(-disparityRange, 1 - minc);
        mind = 0
        maxd = min(disparityRange, imgWidth - maxc - 1)

        # Select the block from the right image to use as the template.
        template = rightI[minr:maxr+1, minc:maxc+1]

        # Get the number of blocks in this search.
        numBlocks = maxd - mind + 1

        # Create a vector to hold the block differences.
        blockDiffs = np.zeros(numBlocks)

        # Calculate the difference between the template and each of the blocks.
        for i, d in enumerate(range(mind, maxd+1)):

            # Select the block from the left image at the distance 'i'.
            # print(str(minr) + ":" + str(maxr+1), str(minc+d) + ":" + str(maxc+d+1))
            block = leftI[minr:maxr+1, minc+d:maxc+d+1]

            # Take the sum of absolute differences(SAD) between the template
            # and the block and store the resulting value.
            blockDiffs[i] = np.sum(np.square(template - block))

        # Sort the SAD values to find the closest match(smallest difference).
        # Discard the sorted vector(the "~" notation), we just want the list
        # of indices.
        #
        # Get the index of the closest - matching block.
        bestMatchIndex = np.argmin(blockDiffs)

        # Convert the index of this block back into an offset.
        # This is the final disparity value produced by basic block matching.
        d = bestMatchIndex + mind - 1

        # Calculate a sub - pixel estimate of the disparity by interpolating.
        # Sub - pixel estimation requires a block to the left and right, so we
        # skip it if the best matching block is at either edge of the search
        # window.
        if bestMatchIndex == 0 or bestMatchIndex == numBlocks-1:
            # Skip sub - pixel estimation and store the initial disparity value.
            DbasicSubpixel[m, n] = d
        else:
            # Grab the SAD values at the closest matching block(C2) and it's
            # immediate neighbors(C1 and C3).
            C1 = blockDiffs[bestMatchIndex - 1]
            C2 = blockDiffs[bestMatchIndex]
            C3 = blockDiffs[bestMatchIndex + 1]

            # Adjust the disparity by some fraction.
            # We're estimating the subpixel location of the true best match.
            DbasicSubpixel[m, n] = d - (0.5 * (C3 - C1) / (C1 - (2 * C2) + C3))

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
