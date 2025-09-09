import cv2
import numpy as np

# ----------------------------------------------------------------------------#

def preprocess_line_image(line_image):
    """
    Comprehensive preprocessing with 
    noise removal and slant correction.
    """
    cv2.imwrite("/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output/0_original.png", line_image)

    # Converting to grayscale if needed.
    if len(line_image.shape) == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_image
    
    cv2.imwrite("/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output/1_grayscale.png", gray)

    # Binarization with adaptive thresholding.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up the image.
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Removing small connected components (noise).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Calculating median component area for adaptive thresholding.
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) > 0:
        median_area = np.median(areas)
        min_size = int(median_area * 0.1)  # 10% of median area.
    else:
        min_size = 10
    
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_size:
            cleaned[labels == i] = 255

    cv2.imwrite("/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output/2_cleaned.png", cleaned)
    
    # Slant correction using a Hough-based stroke detector.
    def estimate_slant_angle(image):
        """
        Estimate text slant angle (character shear) using Hough transform.
        """
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                # Only considering near-vertical lines (to measure slant).
                angle = (theta - np.pi/2) * 180 / np.pi
                if -45 < angle < 45:  # Filtering extremes.
                    angles.append(angle)

        if len(angles) > 0:
            return np.median(angles)
        else:
            return 0
    
    slant_angle = estimate_slant_angle(cleaned)
    print("Slant angle:", slant_angle)
    
    # Applying slant correction.
    if abs(slant_angle) > 1:  # Only correct if significant slant.
        shear_matrix = np.float32([[1, -np.tan(slant_angle * np.pi / 180), 0],
                                 [0, 1, 0]])
        rows, cols = cleaned.shape

        # Applying the changes to both the cleaned binary image and the original image.
        corrected = cv2.warpAffine(cleaned, shear_matrix, (cols, rows))
    else:
        corrected = cleaned
    
    cv2.imwrite("/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output/3_preprocessed.png", corrected)
    return corrected, slant_angle
