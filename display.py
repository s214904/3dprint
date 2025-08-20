import cv2
import numpy as np

# Global display settings
DISPLAY_WIDTH = 600
STATS_WIDTH = 400

def create_display_canvas(color_image, methods, sharpness_values, frame_count, fps):
    """Create display canvas with image and statistics"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Scale image for display
    original_height, original_width = color_image.shape[:2]
    scale_factor = DISPLAY_WIDTH / original_width
    display_height = int(original_height * scale_factor)
    display_image = cv2.resize(color_image, (DISPLAY_WIDTH, display_height))
    
    # Create combined PIL image
    canvas_height = max(display_height, 600)
    pil_image = Image.new('RGB', (DISPLAY_WIDTH + STATS_WIDTH, canvas_height), color=(250, 250, 250))
    
    # Place camera image on left
    if len(display_image.shape) == 3:
        img_pil = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    else:
        img_pil = Image.fromarray(display_image)
    pil_image.paste(img_pil, (0, 0))
    
    # Draw separator line
    draw = ImageDraw.Draw(pil_image)
    draw.line([(DISPLAY_WIDTH, 0), (DISPLAY_WIDTH, canvas_height)], fill=(180, 180, 180), width=2)
    
    # Try to use a system font for better quality
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    x = DISPLAY_WIDTH + 20
    y = 30
    
    # Draw text with PIL - start with metrics (no main title)
    draw.text((x, y), 'Sharpness Metrics', fill=(0, 0, 0), font=font_large)
    y += 35
    
    for method in methods:
        draw.text((x, y), f'{method.title()}:', fill=(0, 0, 0), font=font_medium)
        y += 18
        draw.text((x + 10, y), f'{sharpness_values[method]:.2f}', fill=(0, 0, 100), font=font_medium)
        y += 27
    
    y += 20
    draw.text((x, y), 'System Controls', fill=(0, 0, 0), font=font_large)
    y += 35

    controls = ['Q - Quit', 'D - Save datapoint', 'P - Compute prediction distance']
    for control in controls:
        draw.text((x, y), control, fill=(0, 0, 0), font=font_small)
        y += 30
    
    # Draw FPS info in bottom left corner (FPS first, no double rendering)
    info_text = f'FPS: {fps:.1f}  |  Frame: {frame_count}'
    draw.text((10, canvas_height - 40), info_text, fill=(0, 0, 0), font=font_small)
    
    # Convert PIL image back to OpenCV format
    canvas = np.array(pil_image)
    return canvas

def init_display():
    """Initialize single display window for combined view"""
    cv2.namedWindow('Sharpness Analysis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sharpness Analysis', DISPLAY_WIDTH + STATS_WIDTH, 600)
    return

def show_frame(canvas):
    """Display single combined window and handle key presses"""
    # Convert RGB to BGR for OpenCV display
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    
    # Show the combined window
    cv2.imshow('Sharpness Analysis', canvas_bgr)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    return key

def cleanup_display():
    """Clean up display resources"""
    try:
        cv2.destroyAllWindows()
    except:
        pass