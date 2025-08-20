# 3DPrint Autofocus for AMANDA

This repository contains Python code for performing **auto-focus** using a camera and various focus measures (e.g. Tenengrad, Brenner, Sobel Variance, Laplacian).  
It is designed for use with the **Teledyne FLIR Spinnaker SDK** and is intended to support experimental setups such as AMANDA.  

---

## Installation (Windows)

### 1. Install Python
Ensure that your Python version matches your Spinnaker SDK version.  
Recommended: [Python 3.10.0](https://www.python.org/downloads/release/python-3100/)

### 2. Install Spinnaker SDK (Python version)
Download and install the Python bindings of the Spinnaker SDK:  
[Spinnaker SDK Downloads](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK)

Follow the `README.md` included in the SDK `.zip` package.

### 3. Install required Python libraries
In your environment, install the following packages:

```bash
pip install numpy opencv-python matplotlib scipy pillow
```

The autofocus system supports multiple focus measures and can operate either with a live camera or in simulation mode.
Example usage in Python:
```python
methods = ['tenengrad', 'brenner', 'sobel_variance', 'laplacian']
frames = 100            # Number of frames captured at each distance point
desired_step = 0.050    # mm - desired step size (stepper motor)
min_range = 0.5         # mm - minimum range (±min_range)
live_sharpness = False  # If True: Displays live sharpness metrics
                        # If False: Displays 0 to increase speed
cam = True              # If True: Use camera feed
                        # If False: Use simulated camera

camera(methods, frames, desired_step, min_range, live_sharpness, cam)
```

### Disclaimer / Liability

This software is provided free of charge and may be used, modified, and distributed openly.
However, please note the following:
The software is provided “as is”, without any warranty of any kind.
The author(s) shall not be held liable for any damages, malfunctions, data loss, hardware damage, or unintended consequences resulting from the use of this code.
Users are fully responsible for verifying the suitability and safety of this software for their specific application.
By using this code, you agree that you do so entirely at your own risk.
