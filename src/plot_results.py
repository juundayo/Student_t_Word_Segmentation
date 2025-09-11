import os
import cv2
import numpy as np

# ----------------------------------------------------------------------------#

def create_color_coded_image(line_image, word_bboxes, output_path=None):
    '''Adds a different colour to each word found in the original image.'''
    if not word_bboxes:
        return None
    
    color_coded = line_image.copy()
    if len(color_coded.shape) == 2:
        color_coded = cv2.cvtColor(color_coded, cv2.COLOR_GRAY2BGR)
    
    colors = [
        (0, 0, 255),      # Red (BGR format)
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 128, 255),    # Orange
        (128, 0, 128),    # Purple
        (128, 128, 0),    # Teal
        (0, 0, 128),      # Dark Red
        (0, 128, 0),      # Dark Green
        (128, 0, 0),      # Dark Blue
    ]
    
    for i, (x, y, w, h) in enumerate(word_bboxes):
        color = colors[i % len(colors)]
        
        # Extracting the word region from original image.
        word_region = color_coded[y:y+h, x:x+w]
        
        # Creating mask for black text regions (text is dark).
        if len(word_region.shape) == 3:
            # For color images, text is dark in all channels.
            mask = np.all(word_region < 128, axis=2)
        else:
            # For grayscale images!
            mask = word_region < 128
        
        # Apply color to the text regions.
        if len(word_region.shape) == 3:
            # For each color channel.
            for c in range(3):
                word_region_channel = word_region[:, :, c]
                word_region_channel[mask] = color[c]
                word_region[:, :, c] = word_region_channel
        else:
            # Converting grayscale to color if needed.
            word_region_color = cv2.cvtColor(word_region, cv2.COLOR_GRAY2BGR)
            for c in range(3):
                word_region_color[:, :, c][mask] = color[c]
            color_coded[y:y+h, x:x+w] = word_region_color
    
    if output_path:
        out = os.path.join(output_path, "4_colour_coded.png")
        cv2.imwrite(out, color_coded)
    
    return color_coded

# ----------------------------------------------------------------------------#

def create_numeric_encoded_image(line_image, word_bboxes, output_path=None):
    """
    Creates a numeric encoded image where background 
    is 0 and each word is numbered 1 to n.
    """
    if not word_bboxes:
        return None
    
    # Creating a black background image with same dimensions as the original.
    numeric_encoded = np.zeros(line_image.shape[:2], dtype=np.uint8)
    
    for i, (x, y, w, h) in enumerate(word_bboxes):
        word_value = i + 1  # Words are numbered 1, 2, 3, ...
        
        # Extracting the word region from original image.
        word_region = line_image[y:y+h, x:x+w]
        
        # Creating a mask for text regions (text is dark/black).
        if len(word_region.shape) == 3:
            # For color images, text is dark in all channels.
            mask = np.all(word_region < 128, axis=2)
        else:
            # For grayscale images.
            mask = word_region < 128
        
        # Applying the word value to text regions.
        numeric_encoded[y:y+h, x:x+w][mask] = word_value
    
    if output_path:
        out = os.path.join(output_path, "5_numeric_encoding.png")
        cv2.imwrite(out, numeric_encoded)
    
    return numeric_encoded
