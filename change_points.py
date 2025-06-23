
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
sns.set(style="whitegrid", context="paper", font_scale=1.4)

# Configure Matplotlib to use LaTeX
plt.rcParams.update({
    'text.usetex': False,
    #'pgf.texsystem': 'pdflatex',
    'font.size': 22
})

def change_point_analysis(df_normalized):
    # Calculate the overall mean for each column (bin)
    overall_means = df_normalized.mean(axis=0)

    # Initialize lists to store results for each column (bin)
    cumulative_mean_s_all = []
    mean_ns_all = []
    mean_s_diff_squared_all = []
    mean_ns_diff_squared_all = []

    # Iterate over each column in the DataFrame
    for col in df_normalized.columns:
        # Calculate cumulative mean from 0 to s (forward mean)
        cumulative_mean_s = df_normalized[col].expanding().mean()
        
        # Calculate reverse cumulative sum and reverse cumulative mean from s to t
        reverse_cumsum_ns = df_normalized[col][::-1].cumsum()[::-1]
        reverse_counts = np.arange(len(df_normalized), 0, -1)
        mean_ns = reverse_cumsum_ns / reverse_counts

        # Calculate squared differences from the overall mean for s and ns sections
        mean_s_diff_squared_all.append((cumulative_mean_s - overall_means[col]) ** 2)
        mean_ns_diff_squared_all.append((mean_ns - overall_means[col]) ** 2)

    # Sum squared differences across all columns (bins)
    T1_s_sum = np.column_stack(mean_s_diff_squared_all).sum(axis=1)
    T2_s_sum = np.column_stack(mean_ns_diff_squared_all).sum(axis=1)

    # Create DataFrame with s-index and T-sums
    TS_df = pd.DataFrame({
        's': np.arange(1, len(df_normalized) + 1),  # Indices for time steps
        'T1_s': T1_s_sum,                          # Sum of squared differences for s-section
        'T2_s': T2_s_sum                           # Sum of squared differences for ns-section
    })
    
    # Calculate the minimum T value across both s and ns sections
    TS_df['Tmin(s)'] = TS_df[['T1_s', 'T2_s']].min(axis=1)
    
    # Find the maximum Tmin(s) value as the T-statistic
    T_statistic = TS_df['Tmin(s)'].max()

    return TS_df, T_statistic

def bootstrap_change_points(df_normalized, num_iter = 500, alpha=0.95):
    # Check if df_normalized is empty or has insufficient data
    if df_normalized.empty or df_normalized.shape[0] < 2:
        raise ValueError("DataFrame must have at least two rows for meaningful analysis.")

    T_values = []
    
    for i in tqdm(range(num_iter), desc="Bootsrapping"):
        # Bootstrap resampling
        df_resample = df_normalized.sample(frac=1, replace=True).reset_index(drop=True)
        
        # Run the change point analysis and collect the T statistic
        _, T_statistic = change_point_analysis(df_resample)
        T_values.append(T_statistic)
    
    # Convert T_values to a numpy array
    T_values = np.array(T_values)
    
    # Calculate the threshold z_alpha for the desired confidence level
    n = len(df_normalized)
    z_alpha = 2.5 * np.log(np.log(n)) * np.percentile(T_values, alpha * 100)

    return T_values, z_alpha

#--------------------------------------------------
#             Generalized program
#--------------------------------------------------
def recur_change_pts(df, num_iteration=1000, alpha=0.95):
    # Initialize a list to store all detected change points
    change_points = []
    subset_start = 0
    
    while True:
        # Run change point detection on the current subset
        T, z_alpha, s_hat = detect_change_point(df.iloc[subset_start:], num_iteration, alpha)
        
        if s_hat is None:
            # No further change points detected, break the loop
            print("No further change points detected.")
            break
        
        # Record the absolute position of the change point
        absolute_change_point = subset_start + s_hat
        change_points.append(absolute_change_point)
        
        print(f"Detected change point at t = {absolute_change_point}")
        
        # Update subset_start to start from the next segment after the detected change point
        subset_start = absolute_change_point
        
        # If the remaining subset is too small, stop
        if len(df) - subset_start < 2:
            print("Remaining subset too small to detect further change points.")
            break
    
    return change_points

def find_change_points(df, num_iteration=500, alpha=0.95, results=None):
    if results is None:
        results = {'T_values': [], 'z_alpha_values': [], 's_hat_values': []}

    # Perform the initial change point analysis
    TS_df, T = change_point_analysis(df)
    boot_df, z_alpha = bootstrap_change_points(df, num_iteration, alpha)
    
    n = len(df)
    min_s = int(0.05 * n)
    max_s = int(0.95 * n)
    
    # Determine if a change point exists
    if T > z_alpha:
        # Reject null hypothesis (change point exists)
        # Limit s to a valid range
        TS_df_filtered = TS_df[(TS_df['s'] >= min_s) & (TS_df['s'] <= max_s)]
        
        # Find the s_0 where Tmin(s) is maximized
        s_0 = TS_df_filtered.loc[TS_df_filtered['Tmin(s)'].idxmax(), 's']
        
        # Set s_hat as the index corresponding to s_0
        s_hat = df.index[0] + s_0  # Calculate the index time for s_0
        
        # Store the results
        results['T_values'].append(T)
        results['z_alpha_values'].append(z_alpha)
        results['s_hat_values'].append(s_hat)

        # Change point detected
        print(f"Subset range: from t = {df.index[0]} to t = {df.index[-1]}")
        print(f"Change-point exists! T ({T:.2e}) is greater than z_alpha = {z_alpha:.2e}.")
        print(f"Change point detected at s = {s_hat}.")
        
        # Split the data at the index of s_hat and recursively apply the function
        df_left = df[df.index <= s_hat]  # Slice the DataFrame for the left subset
        df_right = df[df.index > s_hat]  # Slice the DataFrame for the right subset
        
        # Recursively call find_change_points on the left and right subsets
        find_change_points(df_left, num_iteration, alpha, results)
        find_change_points(df_right, num_iteration, alpha, results)
        
    else:
        # No change point detected
        print(f"No change point detected. T = {T:.2e} is less than or equal to z_alpha = {z_alpha:.2e}.")
    return results


def detect_change_point(df, num_iteration=500, alpha=0.95):
    # Perform change point analysis and bootstrapping
    TS_df, T = change_point_analysis(df)
    boot_df, z_alpha = bootstrap_change_points(df, num_iteration, alpha)
    
    # Define the valid range for s using the length of the index
    n = len(df)  # Total number of time steps
    min_s = int(0.05 * n)  # 5% of total length
    max_s = int(0.95 * n)  # 95% of total length

    # Check if change point exists
    if T > z_alpha:
        # Filter the TS_df to include only valid s values
        TS_df_filtered = TS_df[(TS_df['s'] >= min_s) & (TS_df['s'] <= max_s)]
        
        # Find s_0 where Tmin(s) is maximized
        s_0 = TS_df_filtered.loc[TS_df_filtered['Tmin(s)'].idxmax(), 's']
        
        # Set s_hat as the index corresponding to s_0
        s_hat = df.index[0] + s_0  # Calculate the index time for s_0
        
        # Change point detected
        print(f"Subset range: from t = {df.index[0]} to t = {df.index[-1]}")
        print(f"Change-point exists! T ({T:.2e}) is greater than z_alpha = {z_alpha:.2e}.")
        print(f"Change point detected at s = {s_hat}.")
        return T, z_alpha, s_hat  # Return the index where the change point occurs
    else:
        # No change point detected
        print(f"No change-point detected. T ({T:.2e}) is less than or equal to z_alpha = {z_alpha:.2e}.")
        return T, z_alpha, None


# plot the detected change points
def plot_change_points(df, results_change, window_size=30):
    # Extract the change points from the results
    change_points = results_change['s_hat_values']
    df_rolling = df.rolling(window=window_size, min_periods=1).mean()  # Calculate moving average
        
    # Plot the time series data
    plt.figure(figsize=(12, 6))
    
    # Plot the moving averages for each column
    for col in df.columns:
        plt.plot(df.index, df_rolling[col], color='blue', linewidth=2, alpha=0.7)

    # Plot the observations
    plt.plot(df.index, df.values, color='gray', linewidth=1.5, alpha=0.4)
    plt.plot([], [], color='gray', linewidth=1.5, label='Observations')  # Dummy plot for legend
    plt.plot([], [], color='blue', linewidth=2, label='Moving Average')
    
    # Plot the change points
    for change_point in change_points:
        plt.axvline(x=change_point, color='r', linestyle='--', label=f'$\\hat{{s}}={change_point}$')

    # Labeling the plot
    plt.xlabel("Time $t$(s)")
    plt.ylabel("Concentration $C^{*}$")
    
    # Place the legend outside the plot (right side)
    plt.legend(loc='upper left', fontsize=14, frameon=True, shadow=True, title="Legend", title_fontsize=13, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Set grid and layout adjustments
    plt.grid(True)
    plt.tight_layout()  # Ensure the plot fits within the figure without overlap
    
    # Show the plot
    plt.savefig("change_pts.pdf", format="pdf", dpi=400)
    plt.show()
