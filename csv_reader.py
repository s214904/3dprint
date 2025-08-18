import csv
import numpy as np
from collections import defaultdict
import os

def read_sharpness_csv(filepath, mode='mean', include_raw=False, min_distance=None, max_distance=None, step_skip=1):
    """
    Read CSV file with sharpness measurements and handle multiple measurements per distance.
    
    Args:
        filepath (str): Path to the CSV file
        mode (str): How to handle multiple measurements at same distance
                   'mean' - take average of all measurements at same distance
                   'all' - keep all measurements as lists
                   'median' - take median of all measurements at same distance
                   'max' - take maximum of all measurements at same distance
        include_raw (bool): If True, also return raw data alongside processed data
        min_distance (float, optional): Minimum distance to include (inclusive)
        max_distance (float, optional): Maximum distance to include (inclusive)
        step_skip (int): Skip factor for data points. 1 = all data, 2 = every 2nd point, 3 = every 3rd point, etc.
    
    Returns:
        dict: Processed data in format:
              - For mode='mean', 'median', 'max': {'distance': [values], 'method1': [values], ...}
              - For mode='all': {'distance': [values], 'method1': [[values_for_dist1], [values_for_dist2], ...], ...}
              - If include_raw=True: (processed_data, raw_data)
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Store raw data grouped by distance
    raw_data = defaultdict(lambda: defaultdict(list))
    methods = []
    
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Extract method names (skip 'Distance' column) and normalize to lowercase
        methods = [col.strip().lower() for col in header[1:]]
        
        # Read data
        for row in reader:
            if len(row) < len(header):
                continue  # Skip incomplete rows
            
            try:
                distance = float(row[0])
                
                # Store values for each method at this distance
                for i, method in enumerate(methods):
                    value = float(row[i + 1])
                    raw_data[distance][method].append(value)
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid row {row}: {e}")
                continue
    
    # Process data according to mode
    if mode == 'all':
        # For 'all' mode, flatten all measurements (each distance repeated for each measurement)
        processed_data = {'distance': [], **{method: [] for method in methods}}
        
        # Sort distances and apply filtering
        sorted_distances = sorted(raw_data.keys())
        
        # Apply distance range filtering with tolerance for floating-point precision
        tolerance = 1e-10  # Small tolerance to handle floating-point precision issues
        if min_distance is not None:
            sorted_distances = [d for d in sorted_distances if d >= min_distance - tolerance]
        if max_distance is not None:
            sorted_distances = [d for d in sorted_distances if d <= max_distance + tolerance]
        
        # Apply step skipping
        sorted_distances = sorted_distances[::step_skip]
        
        for distance in sorted_distances:
            methods_data = raw_data[distance]
            
            # Get the number of measurements at this distance (should be same for all methods)
            num_measurements = len(methods_data[methods[0]]) if methods else 0
            
            # Add this distance once for each measurement
            for _ in range(num_measurements):
                processed_data['distance'].append(distance)
            
            # Add all individual measurements for each method
            for method in methods:
                values = methods_data[method]
                processed_data[method].extend(values)
    else:
        # For aggregated modes, group by unique distances
        processed_data = {'distance': [], **{method: [] for method in methods}}
        
        # Sort distances and apply filtering
        sorted_distances = sorted(raw_data.keys())
        
        # Apply distance range filtering with tolerance for floating-point precision
        tolerance = 1e-10  # Small tolerance to handle floating-point precision issues
        if min_distance is not None:
            sorted_distances = [d for d in sorted_distances if d >= min_distance - tolerance]
        if max_distance is not None:
            sorted_distances = [d for d in sorted_distances if d <= max_distance + tolerance]
        
        # Apply step skipping
        sorted_distances = sorted_distances[::step_skip]
        
        for distance in sorted_distances:
            processed_data['distance'].append(distance)
            
            for method in methods:
                values = raw_data[distance][method]
                
                if mode == 'mean':
                    processed_data[method].append(np.mean(values) if values else 0.0)
                elif mode == 'median':
                    processed_data[method].append(np.median(values) if values else 0.0)
                elif mode == 'max':
                    processed_data[method].append(np.max(values) if values else 0.0)
                else:
                    raise ValueError(f"Unknown mode: {mode}. Use 'mean', 'median', 'max', or 'all'")
    
    if include_raw:
        # Convert defaultdict to regular dict for cleaner output
        raw_dict = {dist: dict(methods_dict) for dist, methods_dict in raw_data.items()}
        return processed_data, raw_dict
    
    return processed_data


# Example usage of the new parameters:
if __name__ == "__main__":
    # Example usage with new parameters
    print("Example usage of read_sharpness_csv with new parameters:")
    print()
    
    # Assuming you have a CSV file with sharpness data
    # csv_file = "your_sharpness_data.csv"
    
    # Example 1: Read data for distance range -0.5 to 0.5
    # data1 = read_sharpness_csv(csv_file, min_distance=-0.5, max_distance=0.5)
    # print(f"Distance range -0.5 to 0.5: {len(data1['distance'])} data points")
    
    # Example 2: Read data for skewed range -1.0 to 0.5
    # data2 = read_sharpness_csv(csv_file, min_distance=-1.0, max_distance=0.5)
    # print(f"Distance range -1.0 to 0.5: {len(data2['distance'])} data points")
    
    # Example 3: Skip every second measurement (step_skip=2)
    # data3 = read_sharpness_csv(csv_file, step_skip=2)
    # print(f"Every 2nd measurement: {len(data3['distance'])} data points")
    
    # Example 4: Combine range filtering and step skipping
    # data4 = read_sharpness_csv(csv_file, min_distance=-0.8, max_distance=0.8, step_skip=3)
    # print(f"Range -0.8 to 0.8, every 3rd measurement: {len(data4['distance'])} data points")
    
    print("Uncomment the examples above and provide a CSV file path to test the functionality.")