def get_frame(cam):
    """Get a frame from the FLIR camera"""
    try:
        # Get the next image from FLIR camera
        image_result = cam.GetNextImage(1000)

        if image_result.IsIncomplete():
            print(f"Image incomplete with status: {image_result.GetImageStatus()}")
            image_result.Release()
            return None
        
        # Convert to numpy array
        image_data = image_result.GetNDArray()
        
        # Release the image
        image_result.Release()
        return image_data
        
    except Exception as e:
        print(f"Error getting frame: {e}")
        return None