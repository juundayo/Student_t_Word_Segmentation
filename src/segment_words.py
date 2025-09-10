import os
import glob
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess_line_image
from distances import compute_distances
from student_t import StudentsTMixtureModel

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
        return [line_image], distances, np.array([]), None
    
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
        
        # Finding the bounding box for all components in the word.
        min_x = min(comp['bbox'][0] for comp in all_components)
        max_x = max(comp['bbox'][0] + comp['bbox'][2] for comp in all_components)
        min_y = min(comp['bbox'][1] for comp in all_components)
        max_y = max(comp['bbox'][1] + comp['bbox'][3] for comp in all_components)

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
    
    '''
    # Reattaching tonos to the nearest word. 
    for tonos in tonos_components:
        tx, ty, tw, th = tonos['bbox']
        tonos_center = (tx + tw/2, ty + th/2)

        best_word_idx = None
        best_dist = float('inf')
        for i, (wx, wy, ww, wh) in enumerate(word_bboxes):
            word_center = (wx + ww/2, wy + wh/2)
            dist = np.sqrt((word_center[0] - tonos_center[0])**2 + (word_center[1] - tonos_center[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_word_idx = i

        if best_word_idx is not None:
            wx, wy, ww, wh = word_bboxes[best_word_idx]
            min_x = min(wx, tx)
            min_y = min(wy, ty)
            max_x = max(wx + ww, tx + tw)
            max_y = max(wy + wh, ty + th)
            word_bboxes[best_word_idx] = (min_x, min_y, max_x - min_x, max_y - min_y)
    '''
    return word_images, distances, gap_predictions, stmm

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
    parser.add_argument("--img_path", type=str, required=True, 
                        default=IMG_PATH, help="Path to the input image")
    parser.add_argument("--expected_words", type=int, required=True, 
                        default=EXPECTED_WORDS, help="Expected number of words in the line")
    args = parser.parse_args()

    expected_words = args.expected_words
    path = args.img_path

    img = cv2.imread(path)
    if img is None:
        raise Exception("Image is empty or corrupted", path)
    
    clean_folder()

    word_images, distances, gap_predictions, stmm = segment_words(
            line_image=img, 
            expected_word_count=expected_words
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
