import cv2
import numpy as np

# ----------------------------------------------------------------------------#

def preprocess_line_image(line_image):
    """
    Comprehensive preprocessing with 
    noise removal and slant correction.
    """
    # Converting to grayscale if needed.
    if len(line_image.shape) == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_image
    
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
    
    # Slant correction using projection profile analysis.
    def estimate_slant_angle(image):
        """
        Estimating the slant angle using 
        projection profile analysis.
        """
        height, width = image.shape
        angles = np.arange(-30, 31, 1)  # Testing angles from -30 to 30 degrees.
        
        best_angle = 0
        best_variance = 0
        
        for angle in angles:
            # Shearing the image.
            shear_matrix = np.float32([[1, -np.tan(angle * np.pi / 180), 0],
                                     [0, 1, 0]])
            sheared = cv2.warpAffine(image, shear_matrix, (width, height))
            
            # Calculating vertical projection variance.
            vertical_projection = np.sum(sheared, axis=1)
            variance = np.var(vertical_projection)
            
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
        
        return best_angle
    
    slant_angle = estimate_slant_angle(cleaned)
    
    # Applying slant correction.
    if abs(slant_angle) > 1:  # Only correct if significant slant.
        shear_matrix = np.float32([[1, -np.tan(slant_angle * np.pi / 180), 0],
                                 [0, 1, 0]])
        rows, cols = cleaned.shape
        corrected = cv2.warpAffine(cleaned, shear_matrix, (cols, rows))
    else:
        corrected = cleaned
    
    return corrected, slant_angle
