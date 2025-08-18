def camera_feed():
    """Initialize FLIR camera"""
    try:
        import PySpin
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()

        if cam_list.GetSize() > 0:
            cam = cam_list[0]
            
            # Get camera info
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            node_device_model_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))
            if PySpin.IsAvailable(node_device_model_name) and PySpin.IsReadable(node_device_model_name):
                device_model_name = node_device_model_name.GetValue()
                print(f"Using FLIR camera: {device_model_name}")
            
            # Initialize camera
            cam.Init()
            
            # Set acquisition mode to continuous
            nodemap = cam.GetNodeMap()
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if PySpin.IsAvailable(node_acquisition_mode) and PySpin.IsWritable(node_acquisition_mode):
                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if PySpin.IsAvailable(node_acquisition_mode_continuous) and PySpin.IsReadable(node_acquisition_mode_continuous):
                    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            # Begin acquiring images
            cam.BeginAcquisition()
            return cam, system, cam_list
        else:
            # Clean up FLIR resources
            cam_list.Clear()
            system.ReleaseInstance()
            return None, None, None # Exit script - no camera found
            
    except Exception as e:
        print(f"FLIR camera not available: {e}")
        return None, None, None