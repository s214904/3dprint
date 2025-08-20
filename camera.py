import display
import cv2
import time
from compute_sharpness import compute_sharpness
from get_frame import get_frame
from camera_feed import camera_feed
from cleanup_camera import cleanup_camera
from process_image import process_image
import numpy as np
import csv
from datetime import datetime

def camera(methods, frames, desired_step, min_range, live_sharpness, cam):
    """
    Alternative camera display using separate windows and enhanced text rendering
    for sharper, more readable text display.
    """
    if cam:
        USE_MOCK_CAMERA = False
    else:
        USE_MOCK_CAMERA = True

    # Initialize camera or mock camera
    if USE_MOCK_CAMERA:
        print("Using mock camera for testing...")
        cam, system, cam_list = None, None, None
        # Create a mock image (640x480 grayscale with some noise)
        mock_image_shape = (480, 640)
        print("Mock camera initialized")
    else:
        cam, system, cam_list = camera_feed()
        if cam is None:
            print("No FLIR camera available. Exiting...")
            return
    
    # Create CSV file for data collection
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'sharpness_data_{date_str}.csv'
    
    # Create CSV file with header
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Distance'] + [m.capitalize() for m in methods]
        writer.writerow(header)
    
    # Initialize display with separate windows
    display.init_display()
    
    # Initialize tracking variables
    frame_count = 0
    start_time = time.time()

    # Preallocate distance arrays
    d_clicks = 0
    base_unit = 0.00175          # mm - base movement unit, 1.75 microns

    # Compute optimal step_multiplier (closest integer multiple of base_unit)
    step_multiplier = round(desired_step / base_unit)
    actual_step = step_multiplier * base_unit

    # Compute minimum n_steps needed to cover the range
    min_n_steps = int(np.ceil(min_range / actual_step))
    n_steps = min_n_steps  # You can increase this if you want more range

    # Create the distance vector
    d_vec = np.linspace(-n_steps * actual_step, n_steps * actual_step, 2*n_steps + 1)
    print(f"Stepper motor step size: {base_unit}")
    print(f"Step size: {actual_step}")
    print(f"Number of points: {len(d_vec)}")
    print(f"d_vec range: [{d_vec[0]:.6f}, {d_vec[-1]:.6f}]")
    
    try:
        print("---------CONTROLS:---------")
        print("'q': Quit")
        print("'d': Save datapoint for prediction")
        print("'p': Compute prediction distance")
        print("---------------------------")
        
        while True:
            # Get frame from camera or create mock frame
            if USE_MOCK_CAMERA:
                # Create mock image data with some random noise
                image_data = np.random.randint(50, 200, mock_image_shape, dtype=np.uint8)
                # Add some patterns to make it more interesting
                center_y, center_x = mock_image_shape[0] // 2, mock_image_shape[1] // 2
                y, x = np.ogrid[:mock_image_shape[0], :mock_image_shape[1]]
                mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < 100**2
                image_data[mask] = 150  # Add a circle in the center
            else:
                image_data = get_frame(cam)
                if image_data is None:
                    break

            # Process image
            if USE_MOCK_CAMERA:
                # For mock camera, create both color and gray versions
                gray_image = image_data
                color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            else:
                color_image, gray_image = process_image(image_data)
                if color_image is None or gray_image is None:
                    print(f"Unexpected image shape: {image_data.shape}")
                    break

            # Calculate sharpness metrics for every frame
            sharpness_values = {}
            for method in methods:
                if live_sharpness:
                    sharpness_values[method] = compute_sharpness(gray_image, method)
                else:
                    sharpness_values[method] = 0 # Show 0 to increase speed

            # Calculate performance statistics
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            canvas = display.create_display_canvas(
                color_image, methods, sharpness_values, frame_count, fps)
            
            key = display.show_frame(canvas)

            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('d'):
                if len(d_vec) <= d_clicks:
                    print("No more data points to fit")
                    continue
                dist = np.round(d_vec[d_clicks], 3)
                print(f"Distance: {dist:.4f} mm")
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for i in range(frames):
                        # Get frame from camera or create mock frame
                        if USE_MOCK_CAMERA:
                            # Create mock image data with some random noise
                            image_data = np.random.randint(50, 200, mock_image_shape, dtype=np.uint8)
                            # Add some patterns to make it more interesting
                            center_y, center_x = mock_image_shape[0] // 2, mock_image_shape[1] // 2
                            y, x = np.ogrid[:mock_image_shape[0], :mock_image_shape[1]]
                            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < 100**2
                            image_data[mask] = 150  # Add a circle in the center
                            gray_image = image_data
                            color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                        else:
                            image_data = get_frame(cam)
                            if image_data is None:
                                print("No image data received. Exiting...")
                                break
                            # Process image
                            color_image, gray_image = process_image(image_data)
                            if color_image is None or gray_image is None:
                                print(f"Unexpected image shape: {image_data.shape}")
                                break
                        
                        # Calculate sharpness values
                        sharpness_values = {}
                        for method in methods:
                            sharpness_values[method] = compute_sharpness(gray_image, method)
                        
                        # Write data row to CSV
                        row = [dist] + [sharpness_values[method] for method in methods]
                        writer.writerow(row)
                d_clicks += 1
            elif key == ord('p'):   # Compute prediction distance
                if d_clicks < 3:
                    print(f"Need at least 3 data points for fitting. Currently have {d_clicks}. Press 'd' to add more.")
                    continue
                else:
                    from variance_analysis import variance_analysis_main
                    results = variance_analysis_main(methods, csv_filename)

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up with error suppression
        try:
            display.cleanup_display()
        except:
            pass  # Suppress display cleanup errors
        if not USE_MOCK_CAMERA:
            try:
                cleanup_camera(cam, system, cam_list)
            except:
                pass  # Suppress camera cleanup errors
        print("Script stopped")
        cv2.destroyAllWindows()
        if not USE_MOCK_CAMERA:
            cleanup_camera(cam, system, cam_list)
if __name__ == "__main__":
    methods = ['tenengrad', 'brenner', 'sobel_variance', 'laplacian']
    frames = 100            # Number of frames captured at each distance point
    desired_step = 0.050    # mm - desired step size (stepper motor)
    min_range = 0.5         # mm - minimum range (±min_range)
    live_sharpness = False  # If True: Displays live sharpness metrics
                            # If False: Displays 0 to increase speed
    cam = True              # If True: Use camera feed
                            # If False: Use simulated camera
    camera(methods, frames, desired_step, min_range, live_sharpness, cam)