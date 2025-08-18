import cv2
def process_image(image_data):
    """Process raw image data into color and greyscale images"""
    # Handle different image formats
    if len(image_data.shape) == 3: # Color image
        color_image = image_data
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    elif len(image_data.shape) == 2: # Greyscale image
        gray_image = image_data
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)    
    return color_image, gray_image