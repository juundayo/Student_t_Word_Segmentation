import os
import glob
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess_line_image
from distances import compute_distances
from student_t import StudentsTMixtureModel
from plot_results import create_color_coded_image, create_numeric_encoded_image

# ----------------------------------------------------------------------------#

EXPECTED_WORDS = 8
IMG_PATH = "/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/data/LineImages/029_4_L_03.tif"

# ----------------------------------------------------------------------------#

def segment_words(line_image, expected_word_count):
    """
    Complete word segmentation with expected word count integration.
    """
    # Line preprocessing.
    preprocessed, slant_angle = preprocess_line_image(line_image)
    
    # Distance computation.
    distances, ocs = compute_distances(preprocessed)
    
    if len(distances) == 0:
        # Single word case.
        return [line_image], distances, np.array([]), None, [(0, 0, line_image.shape[1], line_image.shape[0])]
    
    # Fitting Student's-t mixture model.
    stmm = StudentsTMixtureModel(n_components=2, max_iter=200, tol=1e-10)
    stmm.fit(distances)
    
    # Predicting gap types.
    gap_probs = stmm.predict_proba(distances)
    gap_predictions = stmm.predict(distances)
    
    # Determining which component corresponds to between-word gaps.
    if stmm.means[0] > stmm.means[1]:
        between_word_class = 0
        within_word_class = 1
    else:
        between_word_class = 1
        within_word_class = 0
    
    # Counting predicted between-word gaps.
    predicted_between_gaps = np.sum(gap_predictions == between_word_class)
    expected_between_gaps = expected_word_count - 1
    
    # Adjusting segmentation based on expected word count.
    if predicted_between_gaps != expected_between_gaps:
        # Using the expected number of gaps to guide segmentation.
        if expected_between_gaps > 0:
            # Getting the distances and sort them.
            sorted_indices = np.argsort(distances)[::-1]
            
            # Selecting top expected_between_gaps distances as between-word gaps.
            gap_predictions = np.full_like(gap_predictions, within_word_class)
            gap_predictions[sorted_indices[:expected_between_gaps]] = between_word_class
        else:
            # No between-word gaps expected (single word).
            gap_predictions = np.full_like(gap_predictions, within_word_class)
    
    # Segment the line into words.
    words = []
    current_word_components = [ocs[0]['components']]  # Start with first OC.
    
    for i, prediction in enumerate(gap_predictions):
        if prediction == between_word_class:
            # End current word, start new word.
            words.append(current_word_components)
            current_word_components = [ocs[i + 1]['components']]
        else:
            # Continue current word.
            current_word_components.extend(ocs[i + 1]['components'])
    
    words.append(current_word_components)
    
    # Extracting word images.
    word_images = []
    word_bboxes = []

    if abs(slant_angle) > 1:
        # Create reverse shear matrix
        reverse_shear_matrix = np.float32([[1, np.tan(slant_angle * np.pi / 180), 0],
                                         [0, 1, 0]])
        rows, cols = line_image.shape[:2]
        
        # Function to transform a point using the reverse shear
        def reverse_slant_point(x, y):
            point = np.array([x, y, 1])
            transformed = reverse_shear_matrix @ point
            return int(transformed[0]), int(transformed[1])
    
    for word_components in words:
        # Flattening all components in the word.
        all_components = []

        for comp_list in word_components:
            if isinstance(comp_list, list):
                all_components.extend(comp_list)
            elif isinstance(comp_list, dict):
                all_components.append(comp_list)
            else:
                print("Unexpected type in word_components:", type(comp_list))
                continue

        if not all_components:
            continue

        # Transforming component coordinates back to original image space!
        bboxes_original = []
        for comp in all_components:
            x, y, w, h = comp['bbox']
            if abs(slant_angle) > 1:
                # Transforming all four corners back to original coordinates.
                tl_x, tl_y = reverse_slant_point(x, y)
                tr_x, tr_y = reverse_slant_point(x + w, y)
                bl_x, bl_y = reverse_slant_point(x, y + h)
                br_x, br_y = reverse_slant_point(x + w, y + h)
                
                # Finding the bounding box of transformed points.
                min_x = min(tl_x, tr_x, bl_x, br_x)
                max_x = max(tl_x, tr_x, bl_x, br_x)
                min_y = min(tl_y, tr_y, bl_y, br_y)
                max_y = max(tl_y, tr_y, bl_y, br_y)
                
                bboxes_original.append((min_x, min_y, max_x - min_x, max_y - min_y))
            else:
                bboxes_original.append((x, y, w, h))
        
        # Finding the bounding box for all components in the word.
        min_x = min(b[0] for b in bboxes_original)
        max_x = max(b[0] + b[2] for b in bboxes_original)
        min_y = min(b[1] for b in bboxes_original)
        max_y = max(b[1] + b[3] for b in bboxes_original)

        padding = 2

        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(line_image.shape[1], max_x + padding)
        max_y = min(line_image.shape[0], max_y + padding)
        
        word_img = line_image[min_y:max_y, min_x:max_x]
        word_images.append(word_img)
        word_bboxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

        # Debug saving.
        debug_image = line_image.copy()
        cv2.rectangle(debug_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.putText(debug_image, f'Word {len(word_images)}', (min_x, min_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(f'/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output/debug_word_{len(word_images)}.png', debug_image)
    
    # Verifying if we have the expected number of words.
    if len(word_images) != expected_word_count:
        # Fallback: use the most confident predictions!
        between_word_probs = gap_probs[:, between_word_class]
        sorted_indices = np.argsort(between_word_probs)[::-1]
        
        # Take top expected_between_gaps as between-word gaps.
        gap_predictions = np.full_like(gap_predictions, within_word_class)
        gap_predictions[sorted_indices[:expected_between_gaps]] = between_word_class
        
        # Re-segment.
        return segment_words(line_image, expected_word_count)

    return word_images, distances, gap_predictions, stmm, word_bboxes

# ----------------------------------------------------------------------------#

def evaluate_segmentation(word_images, expected_word_count, ground_truth_words=None):
    """Comprehensive segmentation evaluation"""
    results = {
        'num_segmented_words': len(word_images),
        'expected_words': expected_word_count,
        'correct_count': len(word_images) == expected_word_count,
        'word_sizes': [img.shape[1] for img in word_images]  # Width of each word
    }
    
    if ground_truth_words is not None:
        # More detailed evaluation if ground truth is available.
        results['ground_truth_count'] = len(ground_truth_words)
        results['exact_match'] = (len(word_images) == len(ground_truth_words))
    
    return results

# ----------------------------------------------------------------------------#

def visualize_segmentation(line_image, word_images, distances, predictions, stmm):
    """Comprehensive visualization of segmentation results"""
    # --- Top row: original line and model fit ---
    _, axes_top = plt.subplots(1, 2, figsize=(15, 6))

    # Original line
    axes_top[0].imshow(line_image, cmap='gray')
    axes_top[0].set_title('Original Text Line')
    axes_top[0].axis('off')

    # Distance distribution with fitted model
    x_range = np.linspace(min(distances) * 0.8, max(distances) * 1.2, 1000)
    for k in range(stmm.n_components):
        pdf_vals = stmm.weights[k] * np.array([
            stmm._student_t_pdf(x, stmm.means[k], stmm.variances[k], stmm.dofs[k]) 
            for x in x_range
        ])
        axes_top[1].plot(x_range, pdf_vals, label=f'Component {k+1}')
    
    axes_top[1].hist(distances, bins=30, density=True, alpha=0.5, label='Distance Distribution')
    axes_top[1].set_xlabel('Distance')
    axes_top[1].set_ylabel('Density')
    axes_top[1].set_title("Student's-t Mixture Model Fit")
    axes_top[1].legend()
    axes_top[1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- Bottom grid: segmented words ---
    n_words = len(word_images)
    cols = min(5, n_words)
    rows = (n_words + cols - 1) // cols

    fig_words, axes_words = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # Normalize axes to 2D array
    axes_words = np.array(axes_words).reshape(-1)

    for i, word_img in enumerate(word_images):
        ax = axes_words[i]
        ax.imshow(word_img, cmap='gray')
        ax.set_title(f'Word {i+1}')
        ax.axis('off')

    # Hide unused subplots.
    for j in range(len(word_images), len(axes_words)):
        axes_words[j].axis('off')

    plt.tight_layout()
    plt.savefig("/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output/vis_segmentation.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Number of distances: {len(distances)}")
    print(f"Number of words segmented: {len(word_images)}")
    print(f"Model parameters:")
    for k in range(stmm.n_components):
        print(f"  Component {k+1}: weight={stmm.weights[k]:.3f}, "
              f"mean={stmm.means[k]:.3f}, variance={stmm.variances[k]:.3f}, "
              f"dof={stmm.dofs[k]:.3f}")
        
# ----------------------------------------------------------------------------#

def clean_folder(folder="/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output"):
    img_patterns = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff')
    for pattern in img_patterns:
        for img_path in glob.glob(os.path.join(folder, pattern)):
            try:
                os.remove(img_path)
            except OSError as e:
                print(f"Could not delete {img_path}: {e}")

# ----------------------------------------------------------------------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Word segmentation with expected word count")
    parser.add_argument("--img_path", type=str, required=False, 
                        default=IMG_PATH, help="Path to the input image")
    parser.add_argument("--expected_words", type=int, required=False, 
                        default=EXPECTED_WORDS, help="Expected number of words in the line")
    args = parser.parse_args()

    expected_words = args.expected_words
    path = args.img_path

    img = cv2.imread(path)
    if img is None:
        raise Exception("Image is empty or corrupted", path)
    
    clean_folder()

    word_images, distances, gap_predictions, stmm, word_bboxes = segment_words(
            line_image=img, 
            expected_word_count=expected_words
        )
    
    color_coded = create_color_coded_image(
        line_image=img,
        word_bboxes=word_bboxes,  
        output_path="/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/output/color_coded_words.png"
    )

    print("\n=== SEGMENTATION RESULTS ===")
    print(f"Number of distances between components: {len(distances)}")
    print(f"Number of words segmented: {len(word_images)}")
    print(f"Gap predictions: {gap_predictions}")

    evaluation = evaluate_segmentation(word_images, expected_word_count=expected_words)
    print("\n=== EVALUATION ===")
    print(f"Segmented words: {evaluation['num_segmented_words']}")
    print(f"Expected words: {evaluation['expected_words']}")
    print(f"Correct count: {evaluation['correct_count']}")
    print(f"Word widths: {evaluation['word_sizes']}")

    print("\nGenerating visualization...")
    visualize_segmentation(img, word_images, distances, gap_predictions, stmm)
