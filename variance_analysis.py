import numpy as np
import matplotlib.pyplot as plt
from csv_reader import read_sharpness_csv
from scipy import stats
from scipy.optimize import curve_fit


def gaussian(x, a, b, c, d):
    """Gaussian function for curve fitting."""
    return d + a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def fit_gaussian_to_sharpness(distances, sharpness_values, method_name):
    """Fit a Gaussian curve to sharpness vs distance data."""
    try:
        # Sort data by distance
        sorted_indices = np.argsort(distances)
        x_data = distances[sorted_indices]
        y_data = sharpness_values[sorted_indices]
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 3:
            return {'success': False, 'error': 'Insufficient data points'}
        
        # Initial parameter estimates
        a_init = np.max(y_data) - np.min(y_data)  # Amplitude
        b_init = x_data[np.argmax(y_data)]        # Optimal distance (peak location)
        c_init = (np.max(x_data) - np.min(x_data)) / 4  # Width estimate
        d_init = np.min(y_data)  # Offset estimate (baseline)
        
        # Perform curve fitting with initial parameter estimates
        popt, pcov = curve_fit(gaussian, x_data, y_data, 
                              p0=[a_init, b_init, c_init, d_init], 
                              maxfev=5000)
        
        # Extract fitted parameters
        amplitude, optimal_distance, width, offset = popt
        
        # Calculate parameter uncertainties (standard errors)
        parameter_errors = np.sqrt(np.diag(pcov))
        optimal_distance_err = parameter_errors[1]
        
        # Calculate degrees of freedom for t-distribution
        n_points = len(x_data)  # Number of distance points used in fitting
        n_params = 3  # Gaussian has 3 parameters (a, b, c)
        degrees_of_freedom = n_points - n_params
        
        # Use t-distribution for more accurate confidence intervals
        if degrees_of_freedom > 0:
            t_critical = stats.t.ppf(0.975, degrees_of_freedom)  # 97.5th percentile for 95% CI
        else:
            t_critical = 1.96  # Fallback to normal approximation if insufficient degrees of freedom
        
        # Calculate 95% confidence interval for optimal distance using t-distribution
        ci_half_width = t_critical * optimal_distance_err
        ci_lower = optimal_distance - ci_half_width
        ci_upper = optimal_distance + ci_half_width
        
        # Calculate goodness of fit (R²)
        y_pred = gaussian(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'success': True,
            'fitted_params': popt,
            'optimal_distance': optimal_distance,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_half_width': ci_half_width,
            'r_squared': r_squared,
            'n_points': n_points,
            'degrees_of_freedom': degrees_of_freedom,
            't_critical': t_critical
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def calculate_required_frames_for_distance_tolerance(variance_results, gaussian_fits, frames_per_distance, target_tolerance=0.010):
    """Calculate frames required to achieve target tolerance."""
    frame_requirements = {}
    
    for method in variance_results.keys():
        if method in gaussian_fits and gaussian_fits[method]['success']:
            current_ci_half_width = gaussian_fits[method]['ci_half_width']
            current_frames_per_distance = frames_per_distance
            n_distance_points = len(variance_results[method]['variance_data'])
            
            if current_ci_half_width > 0 and target_tolerance > 0:
                # The CI depends on both the number of distance points and frames per distance
                # Assuming we keep the same number of distance points, scale frames per distance
                frames_ratio = (current_ci_half_width / target_tolerance) ** 2
                required_frames_per_distance = current_frames_per_distance * frames_ratio
                total_required_frames = required_frames_per_distance * n_distance_points
                
                frame_requirements[method] = {
                    'current_ci_half_width': current_ci_half_width,
                    'current_frames_per_distance': current_frames_per_distance,
                    'required_frames_per_distance': required_frames_per_distance,
                    'n_distance_points': n_distance_points,
                    'total_required_frames': total_required_frames,
                    'scaling_assumption': 'frames_per_distance_only'
                }
        else:
            frame_requirements[method] = {'error': 'Gaussian fit failed'}
    
    return frame_requirements


def analyze_frame_variance(filename, methods, min_distance=None, max_distance=None, step_skip=1):
    """Analyze variance of sharpness metrics across multiple frames."""
    # Read raw data
    data = read_sharpness_csv(filename, mode='all', include_raw=True, 
                             min_distance=min_distance, max_distance=max_distance, 
                             step_skip=step_skip)
    
    processed_data, raw_data = data
    results = {}
    
    for method in methods:
        distances = np.array(processed_data['distance'])
        frames = np.sum(distances == distances[0])
        print(f"  {method}: Using {int(len(distances)/frames)} measurements")
        values = np.array(processed_data[method])
        
        # Apply standard normalization
        values_normalized = (values - np.mean(values)) / np.std(values, ddof=1)
        # Subtract min value to avoid negatives
        values_normalized = values_normalized - np.min(values_normalized)
        # Add small shift to avoid 0 values
        eps = 1e-8
        values_normalized = values_normalized + eps
        unique_distances = sorted(set(distances))
        variance_data = []
        
        for dist in unique_distances:
            # Get all measurements at this distance
            mask = distances == dist
            dist_values = values_normalized[mask]
            
            if len(dist_values) > 1:  # Need at least 2 measurements
                variance_info = {
                    'distance': dist,
                    'mean': np.mean(dist_values),
                    'sem': stats.sem(dist_values)
                }
                variance_data.append(variance_info)
        
        results[method] = {
            'variance_data': variance_data
        }
    
    return results, frames


def analyze_gaussian_fits(variance_results):
    """Perform Gaussian curve fitting analysis."""
    gaussian_fits = {}
    
    for method, method_results in variance_results.items():
        variance_data = method_results['variance_data']
        
        if len(variance_data) < 3:
            gaussian_fits[method] = {'success': False, 'error': 'Insufficient data'}
            continue
        
        # Extract distance and mean sharpness values
        distances = np.array([d['distance'] for d in variance_data])
        sharpness_means = np.array([d['mean'] for d in variance_data])
        
        # Perform Gaussian fitting
        fit_result = fit_gaussian_to_sharpness(distances, sharpness_means, method)
        gaussian_fits[method] = fit_result
    
    return gaussian_fits


def plot_variance_analysis(results, gaussian_fits=None):
    """Create 3 separate plot windows for variance analysis results."""
    methods = list(results.keys())
    
    # Plot 1: Standard Error of Mean vs Distance
    if 1 == 0:
        plt.figure(figsize=(8, 6))
        for method in methods:
            variance_data = results[method]['variance_data']
            distances = [d['distance'] for d in variance_data]
            sems = [d['sem'] for d in variance_data]
            plt.plot(distances, sems, 'o-', label=method.capitalize(), alpha=0.7)
        
        plt.xlabel('Distance (mm)')
        plt.ylabel('Standard Error of Mean')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    # Plot 2: Gaussian Fits
    if 1 == 1:
        plt.figure(figsize=(12, 8))
        if gaussian_fits:
            # Store line colors for each method to match with vertical lines
            method_colors = {}
            
            for method in methods:
                if method in gaussian_fits and gaussian_fits[method]['success']:
                    variance_data = results[method]['variance_data']
                    distances = np.array([d['distance'] for d in variance_data])
                    means = np.array([d['mean'] for d in variance_data])
                    
                    fit_result = gaussian_fits[method]
                    
                    # Plot data points (no label for legend)
                    scatter = plt.scatter(distances, means, alpha=0.6, s=50)
                    
                    # Plot Gaussian fit with combined label including R²
                    x_smooth = np.linspace(np.min(distances), np.max(distances), 100)
                    y_smooth = gaussian(x_smooth, *fit_result['fitted_params'])
                    line = plt.plot(x_smooth, y_smooth, '-', alpha=0.8, linewidth=3,
                            label=f'{method.capitalize()} (R²={fit_result["r_squared"]:.3f})')
                    
                    # Store the color for this method
                    method_colors[method.lower()] = line[0].get_color()
            
            # Add vertical lines and collect text for top left corner
            optimal_texts = []
            
            for method in methods:
                if method.lower() in ['brenner', 'tenengrad'] and method in gaussian_fits and gaussian_fits[method]['success']:
                    fit_result = gaussian_fits[method]
                    optimal_dist = fit_result['optimal_distance']
                    ci_half_width = fit_result['ci_half_width']
                    
                    # Add vertical line at optimal distance with matching color
                    method_color = method_colors.get(method.lower(), 'gray')
                    plt.axvline(optimal_dist, color=method_color, 
                               linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Collect text for top left corner with 95% CI notation
                    optimal_texts.append(f'{method.capitalize()}: {optimal_dist:.4f} mm (95% CI ± {ci_half_width:.4f})')
            
            # Add both texts in top left corner, two lines
            if optimal_texts:
                ax = plt.gca()
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_text = xlim[0] + (xlim[1] - xlim[0]) * 0.01  # 2% from left edge
                y_start = ylim[1] - (ylim[1] - ylim[0]) * 0.05  # 5% from top edge
                
                for i, text in enumerate(optimal_texts):
                    y_text = y_start - i * (ylim[1] - ylim[0]) * 0.08  # 8% spacing between lines
                    plt.text(x_text, y_text, text, ha='left', va='top', fontsize=18)
        
        plt.xlabel('Distance (mm)', fontsize=24)
        plt.ylabel('Normalized Sharpness', fontsize=24)
        plt.legend(loc='upper right', fontsize=22)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.tight_layout()
    
        # Save the Gaussian fit plot
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'gaussian_fits_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gaussian fit plot saved as: {filename}")

    # Plot 3: Confidence Intervals
    if 1 == 0:
        plt.figure(figsize=(8, 6))
        if gaussian_fits:
            method_positions = np.arange(len(methods))
            
            for i, method in enumerate(methods):
                if method in gaussian_fits and gaussian_fits[method]['success']:
                    fit_result = gaussian_fits[method]
                    optimal_dist = fit_result['optimal_distance']
                    ci_lower = fit_result['ci_lower']
                    ci_upper = fit_result['ci_upper']
                    
                    # Plot confidence interval as error bar
                    plt.errorbar(i, optimal_dist, 
                            yerr=[[optimal_dist - ci_lower], [ci_upper - optimal_dist]], 
                            fmt='o', capsize=5, capthick=2, 
                            label=f'{method.capitalize()}')
                    
                    # Add text showing CI width
                    ci_width = fit_result['ci_half_width']
                    plt.text(i, optimal_dist + ci_width + 0.001, f'±{ci_width:.4f}', 
                            ha='center', va='bottom', fontsize=9)
            
            plt.xticks(method_positions, [m.capitalize() for m in methods])
            plt.ylabel('Optimal Distance (mm)')
            plt.grid(True, alpha=0.3)
            
            # Add horizontal line at zero if within range
            ylim = plt.gca().get_ylim()
            if ylim[0] <= 0 <= ylim[1]:
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero distance')
                plt.legend()
        
        plt.tight_layout()
    plt.show()


def variance_analysis_main(methods, filename='Video/sharpness_20250815_145033.csv', 
                          min_distance=None, max_distance=None, step_skip=1, target_tolerance=0.001):
    """Main function to run complete variance analysis."""
    
    print("="*60)
    print("GAUSSIAN FOCUS ANALYSIS")
    print("="*60)
    
    # Perform variance analysis
    results, frames = analyze_frame_variance(filename, methods, min_distance, max_distance, step_skip)
    
    # Perform Gaussian fitting analysis
    gaussian_fits = analyze_gaussian_fits(results)
    
    # Calculate frame requirements for distance tolerance
    frame_requirements = calculate_required_frames_for_distance_tolerance(
        results, gaussian_fits, frames, target_tolerance)
    
    # Print only essential results
    print(f"\nTarget tolerance: ±{target_tolerance:.3f}")
    print("-" * 60)
    
    for method in methods:
        if method in gaussian_fits and gaussian_fits[method]['success']:
            fit_result = gaussian_fits[method]
            optimal_dist = fit_result['optimal_distance']
            
            # Calculate closest integer multiple of 0.00175
            base_unit = 0.00175  # mm
            closest_multiple = round(optimal_dist / base_unit) * base_unit
            multiple_index = round(optimal_dist / base_unit)
            
            print(f"{method.upper():12}: Optimal = {optimal_dist:.6f}, "
                  f"CI = [{fit_result['ci_lower']:.6f}, {fit_result['ci_upper']:.6f}], "
                  f"R² = {fit_result['r_squared']:.4f}")
            print(f"{'':12}  Closest 0.00175 multiple: {closest_multiple:.6f} ({multiple_index:+d} × 0.00175)")
            
            # Frame requirements for distance tolerance
            if method in frame_requirements and 'required_frames_per_distance' in frame_requirements[method]:
                req = frame_requirements[method]
                if req['current_ci_half_width'] < target_tolerance:
                    print(f"{'':12}  Current CI (±{req['current_ci_half_width']:.6f}) is already better than target (±{target_tolerance:.3f})")
                    print(f"{'':12}  Could achieve ±{target_tolerance:.3f} with {req['required_frames_per_distance']:.1f} frames per distance")
                    print(f"{'':12}  Total frames: {req['total_required_frames']:.0f} (across {req['n_distance_points']} distances)")
                else:
                    print(f"{'':12}  Need {req['required_frames_per_distance']:.1f} frames per distance to achieve ±{target_tolerance:.3f} tolerance")
                    print(f"{'':12}  Total frames needed: {req['total_required_frames']:.0f} (across {req['n_distance_points']} distances)")
                    print(f"{'':12}  Current: {req['current_frames_per_distance']} frames per distance, CI: ±{req['current_ci_half_width']:.6f}")
            elif method in frame_requirements and 'error' in frame_requirements[method]:
                print(f"{'':12}  Frame estimation failed: {frame_requirements[method]['error']}")
            else:
                print(f"{'':12}  No frame requirements calculated")
        else:
            print(f"{method.upper():12}: Gaussian fitting failed")
    
    # Create plots
    plot_variance_analysis(results, gaussian_fits)
    

    return results
