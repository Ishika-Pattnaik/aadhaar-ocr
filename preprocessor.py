import cv2
import numpy as np
import logging
from config import GAUSSIAN_KERNEL, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        pass

    def load_image(self, image_path):
        """Loads image from path."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        return img

    def grayscale(self, image):
        """Converts to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def denoise(self, image):
        """Applies Gaussian blur to reduce noise."""
        return cv2.GaussianBlur(image, GAUSSIAN_KERNEL, 0)

    def adaptive_threshold(self, image):
        """Applies adaptive thresholding to binarize image."""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )

    def auto_scan_crop(self, image):
        """
        Attempts to detect the card document and crop it.
        Uses Canny Edge Detection + Find Contours.
        """
        try:
            # Resize for consistent processing
            h, w = image.shape[:2]
            scale = 1.0
            if w > 1920:
                scale = 1920 / w
                small = cv2.resize(image, None, fx=scale, fy=scale)
            else:
                small = image

            gray = self.grayscale(small)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge Detection
            edges = cv2.Canny(blurred, 50, 200)
            
            # Find Contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate minimum area threshold (at least 5% of image area)
            min_area = (w * h) * 0.05
            logger.debug(f"Looking for contours with area > {min_area:.0f}")
            
            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            card_contour = None
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    # Since sorted by area, all remaining contours will be smaller
                    break
                    
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                logger.debug(f"Contour area: {area:.0f}, points: {len(approx)}")
                
                # If it has 4 points, assume it's our card
                if len(approx) == 4:
                    card_contour = approx
                    logger.debug(f"Found card contour with area: {area:.0f}")
                    break
            
            if card_contour is not None:
                # Get perspective transform
                warped = self.four_point_transform(image, card_contour.reshape(4, 2) / scale)
                return warped
            else:
                logger.warning("No document contour found, using original image.")
                return image

        except Exception as e:
            logger.error(f"Auto crop failed: {e}")
            return image

    def four_point_transform(self, image, pts):
        """Perspective transform utility."""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width of new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute height of new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def order_points(self, pts):
        """Orders points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
        
    def preprocess_pipeline(self, image_path):
        """Full pipeline."""
        original = self.load_image(image_path)
        
        # 1. Auto Crop (Geometric)
        cropped = self.auto_scan_crop(original)
        
        # 2. Grayscale
        gray = self.grayscale(cropped)
        
        # 3. Denoise
        denoised = self.denoise(gray)
        
        # 4. Adaptive Thresholding (for OCR) - Optional, sometimes OCR engine prefers raw gray
        # But we return both for the engine to try
        binary = self.adaptive_threshold(denoised)
        
        # Convert grayscale to BGR for PaddleOCR compatibility
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Keep binary as-is (it's single channel)
        
        return gray_bgr, binary
