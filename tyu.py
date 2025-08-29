import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_prediction_edges(image_path, low_threshold=100, high_threshold=200, show=False):
    """
    Reads an image, converts it to grayscale, and returns the Canny edge prediction.

    Args:
        image_path (str): Path to the image file.
        low_threshold (int): Lower bound for hysteresis thresholding.
        high_threshold (int): Upper bound for hysteresis thresholding.
        show (bool): If True, displays the edge image.

    Returns:
        edges (np.ndarray): Binary edge map.
    """
    # Load the image - Fixed: Use the parameter instead of hardcoded path
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    if show:
        plt.figure(figsize=(12, 6))  # Increased figure size
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Edges")
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return edges

# Example usage:
if __name__ == "__main__":
    # Fixed: Use raw string or forward slashes for the path
    image_path = r"C:\Users\Sweety gupta\Desktop\Hack 2 skill\quicksand-in-the-indian-ocean-galu-beach-kenya-it-is-dangerous-to-walk-during-the-low-tide-up-to-100-meters-away-from-beach-as-areas-with-quicksan-PKX0K2.jpg"
    
    try:
        edge_map = get_prediction_edges(image_path, show=True)
        cv2.imwrite("example_edges.png", edge_map)
        print("✅ Edge detection completed successfully!")
        print("📁 Output saved as 'example_edges.png'")
    except Exception as e:
        print(f"❌ Error: {e}")