def cleanup_camera(cam, system, cam_list):
    """Clean up camera resources with error handling"""
    try:
        if cam is not None:
            cam.EndAcquisition()
    except:
        pass  # Camera might already be stopped
        
    try:
        if cam is not None:
            cam.DeInit()
    except:
        pass  # Camera might already be deinitialized
        
    try:
        if cam is not None:
            del cam
    except:
        pass
        
    try:
        if cam_list is not None:
            cam_list.Clear()
    except:
        pass
        
    try:
        if system is not None:
            system.ReleaseInstance()
    except:
        pass  # System might already be released