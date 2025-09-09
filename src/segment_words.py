import numpy as np
import cv2
from scipy.special import digamma, gamma
import matplotlib.pyplot as plt

from preprocessing import preprocess_line_image
from distances import compute_distances
from student_t import StudentsTMixtureModel

def segment_words(line_image, expected_word_count):
    """
    Complete word segmentation with expected word count integration
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
    
    # Extract word images.
    word_images = []
    word_bboxes = []
    
    for word_components in words:
        # Flatten all components in the word.
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
        
        # Addding some padding.
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(line_image.shape[1], max_x + padding)
        max_y = min(line_image.shape[0], max_y + padding)
        
        word_img = line_image[min_y:max_y, min_x:max_x]
        word_images.append(word_img)
        word_bboxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
    
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

    # Hide unused subplots
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

if __name__ == "__main__":
    expected_words = 4
    path = "/home/ml3/Desktop/Thesis/.venv/02_WordSegmentation/data/Screenshot_25.png"
    img = cv2.imread(path)

    if img is None:
        raise Exception("Image is empty or corrupted", path)
    
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
