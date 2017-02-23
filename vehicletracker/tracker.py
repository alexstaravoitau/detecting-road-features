import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.transform import resize

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
        visualise=vis, feature_vector=feature_vec)

# Define a function to compute binned color features
def bin_spatial(img, size=(16, 16)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=16, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
def extract_features(images, orient=10, pix_per_cell=8, cell_per_block=2):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in images:
        file_features = []
        #feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        feature_image = np.copy(image)

        spatial_features = bin_spatial(feature_image)
        file_features.append(spatial_features)

        hist_features = color_hist(feature_image)
        file_features.append(hist_features)

        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
        file_features.append(hog_features)

        # Append the new feature vector to the features list
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def get_regions(image, scale=1., k=64):
    (h, w, d) = image.shape
    scaled_image = resize(image, (h * scale, w * scale, d), preserve_range=True).astype(np.float32)
    (h, w, d) = scaled_image.shape
    regions = np.empty([0, k, k, d], dtype=np.float32)
    regions_coordinates = np.empty([0, 4], dtype=np.int)
    s = k // 2
    y_range, y_s = np.linspace(h / 2, h - k, (h + s) // (2 * s), retstep=True)
    x_range, x_s = np.linspace(0, w - k, (w + s) // s, retstep=True)
    for i in y_range.astype(np.int):
        for j in x_range.astype(np.int):
            regions = np.append(regions, [scaled_image[i:i+k, j:j+k, :]], axis=0)
            regions_coordinates = np.append(regions_coordinates, [[j, i, j+k, i+k]], axis=0)

    return regions.astype(np.float32), (regions_coordinates / scale).astype(np.int)