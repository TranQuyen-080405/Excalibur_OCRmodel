import cv2
import numpy as np
import os
from PIL import Image
def split_long_image_for_ocr(image_path, max_height=2000, overlap=100, output_dir="split_images"):
    """
    Split a long image into smaller chunks for better OCR processing.
    
    Args:
        image_path (str): Path to the input image
        max_height (int): Maximum height for each split image chunk
        overlap (int): Overlap between consecutive chunks to avoid cutting text
        output_dir (str): Directory to save split images
    
    Returns:
        list: List of paths to the split image files
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = img.shape[:2]
    print(f"Original image dimensions: {width}x{height}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # If image is not long, return the original
    if height <= max_height:
        print("Image is not long enough to split")
        return [image_path]
    
    split_files = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Calculate number of splits needed
    num_splits = (height - overlap) // (max_height - overlap) + 1
    print(f"Splitting into {num_splits} chunks...")
    
    for i in range(num_splits):
        # Calculate start and end positions
        start_y = i * (max_height - overlap)
        end_y = min(start_y + max_height, height)
        
        # Extract the chunk
        chunk = img[start_y:end_y, :]
        
        # Save the chunk
        output_filename = f"{base_name}_chunk_{i:03d}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, chunk)
        split_files.append(output_path)
        
        print(f"Saved chunk {i+1}/{num_splits}: {output_filename} ({chunk.shape[1]}x{chunk.shape[0]})")
        
        # Break if we've reached the end
        if end_y >= height:
            break
    
    print(f"Successfully split image into {len(split_files)} chunks")
    return split_files

# ==========================================
# EXPERIMENTAL PREPROCESSING METHODS
# ==========================================
def read_img(img_path):
    """Read image using Pillow and convert to OpenCV format."""
    try:
        pil_img = Image.open(img_path).convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"Error reading {img_path} with Pillow: {e}")
        return None

    return img


def experiment_gaussian_blur(image_path, output_path=None, kernel_size=5, sigma_x=0):
    """Gaussian blur for noise reduction."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    processed = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_x)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_median_blur(image_path, output_path=None, kernel_size=5):
    """Median blur for salt-and-pepper noise removal."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    processed = cv2.medianBlur(img, kernel_size)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_bilateral_filter(image_path, output_path=None, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter for edge-preserving noise reduction."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    processed = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_histogram_equalization(image_path, output_path=None):
    """Global histogram equalization."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    processed = cv2.equalizeHist(img)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_clahe(image_path, output_path=None, clip_limit=2.0, tile_grid_size=(8,8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    processed = clahe.apply(img)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_gamma_correction(image_path, output_path=None, gamma=1.0):
    """Gamma correction for brightness adjustment."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
        if img is None or img.size == 0:
            raise ValueError("Invalid input image array")
    
    # Validate image dimensions
    if len(img.shape) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Image has invalid dimensions")
    
    # Build lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    processed = cv2.LUT(img, table)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_unsharp_masking(image_path, output_path=None, sigma=1.0, strength=1.5):
    """Unsharp masking for text sharpening."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Unsharp mask
    processed = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_otsu_threshold(image_path, output_path=None):
    """Otsu's automatic thresholding."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
        if img is None or img.size == 0:
            raise ValueError("Invalid input image array")
    
    # Validate image dimensions
    if len(img.shape) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Image has invalid dimensions")
    
    _, processed = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Validate processed image before saving
    if processed is None or processed.size == 0:
        raise ValueError("Failed to process image with Otsu thresholding")
    
    if output_path:
        try:
            cv2.imwrite(output_path, processed)
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
    
    return processed

def experiment_adaptive_threshold(image_path, output_path=None, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 block_size=11, c_value=2):
    """Adaptive thresholding methods."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    processed = cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY, block_size, c_value)
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_morphology(image_path, output_path=None, operation='opening', kernel_size=3, iterations=1):
    """Morphological operations."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
        if img is None or img.size == 0:
            raise ValueError("Invalid input image array")
    
    # Validate image dimensions
    if len(img.shape) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Image has invalid dimensions")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if operation == 'opening':
        processed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'closing':
        processed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'erosion':
        processed = cv2.erode(img, kernel, iterations=iterations)
    elif operation == 'dilation':
        processed = cv2.dilate(img, kernel, iterations=iterations)
    elif operation == 'gradient':
        processed = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    else:
        processed = img
    
    # Validate processed image before saving
    if processed is None or processed.size == 0:
        raise ValueError(f"Failed to process image with morphological {operation}")
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_edge_enhancement(image_path, output_path=None, method='laplacian'):
    """Edge enhancement methods."""
    if isinstance(image_path, str):
        img = read_img(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = image_path.copy()
    
    if method == 'laplacian':
        edges = cv2.Laplacian(img, cv2.CV_64F)
        processed = np.uint8(np.absolute(edges))
    elif method == 'sobel_x':
        processed = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        processed = np.uint8(np.absolute(processed))
    elif method == 'sobel_y':
        processed = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        processed = np.uint8(np.absolute(processed))
    elif method == 'sobel_combined':
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        processed = np.uint8(np.sqrt(sobelx**2 + sobely**2))
    else:
        processed = img
    
    if output_path:
        cv2.imwrite(output_path, processed)
    
    return processed

def experiment_canny_edge(img_path, low=100, high=200):
    """Canny edge detection."""
    if isinstance(img_path, str):
        img = read_img(img_path)
        if img is None:
            raise ValueError(f"Could not load image from {img_path}")
    else:
        img = img_path.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges

def experiment_deskew(img_path):
    """Deskew image using rotation."""
    if isinstance(img_path, str):
        img = read_img(img_path)
        if img is None:
            raise ValueError(f"Could not load image from {img_path}")
    else:
        img = img_path.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


