import cv2
import numpy as np

# ----------------------------------------------------------------------------#

def compute_distances(preprocessed_image):
    """
    Comprehensive distance computation with proper overlapped component definition
    """
    # Finding connected components.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        preprocessed_image, connectivity=8
    )
    
    if num_labels <= 1:
        return np.array([]), [], []
    
    # Getting all components (excluding background).
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        bbox = (x, y, w, h)
        components.append({
            'label': i,
            'bbox': bbox,
            'centroid': centroids[i],
            'area': area
        })
    
    # Sorting components by their x-coordinate.
    components.sort(key=lambda c: c['bbox'][0])
    
    # Defining overlapped components (OCs) 
    # based on horizontal projection overlap.
    def components_overlap(comp1, comp2):
        """Checking if two components overlap horizontally."""
        x1, y1, w1, h1 = comp1['bbox']
        x2, y2, w2, h2 = comp2['bbox']
        return max(x1, x2) < min(x1 + w1, x2 + w2)
    
    def merge_components(comp_list):
        """Merging a list of components into one OC."""
        if not comp_list:
            return None
        
        x_min = min(comp['bbox'][0] for comp in comp_list)
        y_min = min(comp['bbox'][1] for comp in comp_list)
        x_max = max(comp['bbox'][0] + comp['bbox'][2] for comp in comp_list)
        y_max = max(comp['bbox'][1] + comp['bbox'][3] for comp in comp_list)
        
        return {
            'components': comp_list,
            'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
            'centroids': [comp['centroid'] for comp in comp_list]
        }
    
    # Grouping components into OCs using agglomerative clustering.
    ocs = []
    current_oc = [components[0]]
    
    for i in range(1, len(components)):
        current_comp = components[i]
        overlaps = False
        
        # Checking if the current component overlaps 
        # with any component in the current OC.
        for oc_comp in current_oc:
            if components_overlap(current_comp, oc_comp):
                overlaps = True
                break
        
        if overlaps:
            current_oc.append(current_comp)
        else:
            ocs.append(merge_components(current_oc))
            current_oc = [current_comp]
    
    if current_oc:
        ocs.append(merge_components(current_oc))
    
    # Computing the exact Euclidean distances between adjacent OCs.
    distances = []
    for i in range(len(ocs) - 1):
        oc1 = ocs[i]
        oc2 = ocs[i + 1]
        
        # Getting the rightmost points of OC1 and leftmost points of OC2.
        oc1_right = [(comp['bbox'][0] + comp['bbox'][2], comp['bbox'][1] + comp['bbox'][3] / 2) 
                    for comp in oc1['components']]
        oc2_left = [(comp['bbox'][0], comp['bbox'][1] + comp['bbox'][3] / 2) 
                   for comp in oc2['components']]
        
        # Finding the minimum Euclidean distance between all point pairs.
        min_distance = float('inf')
        for p1 in oc1_right:
            for p2 in oc2_left:
                distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                if distance < min_distance:
                    min_distance = distance
        
        distances.append(max(0, min_distance))
    
    return np.array(distances), ocs
