import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Configure Matplotlib to use LaTeX
plt.rcParams.update({
    'text.usetex': False,
    #'pgf.texsystem': 'pdflatex',
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.titlesize': 18,
    'figure.autolayout': True
})
sns.set(style="whitegrid", context="paper", font_scale=1.4)

#----------------------------------------------------------------
# Function for scaled log transform
def u(x, mu, sigma, h):
    """
    defines the substitution variable u(x) to be used in the log-normal function.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    if np.any(x <= h):  # Check if any value in x violates the condition
        raise ValueError(f"Some values in x are invalid for the given h = {h}. Logarithm is undefined.")
    return (np.log(x - h) - mu) / sigma

#----------------------------------------------------------------
# Function to compute the integral for a bin
def bin_integral(xleft, xright, mu1, sigma1, h, mu2, sigma2, weight1=0.49, weight2=0.51):
    """
    Computes the normalized integral of a convex combination of two log-normal functions
    over the interval [xleft, xright].
    """
    try:
        # Parameter validation
        if weight1 + weight2 != 1.0:
            raise ValueError("Weights must sum to 1.")
        if xleft <= h or xright <= h:
            raise ValueError("xleft or xright is invalid for the logarithm with given h.")

        sqrt2 = np.sqrt(2)
        
        # Compute u-values for edges and normalization range
        u_left = u(xleft, mu1, sigma1, h)
        u_right = u(xright, mu1, sigma1, h)
        v_left = u(xleft, mu2, sigma2, h)
        v_right = u(xright, mu2, sigma2, h)
        
        u_min, u_max = u(135, mu1, sigma1, h), u(3615, mu1, sigma1, h)
        v_min, v_max = u(135, mu2, sigma2, h), u(3615, mu2, sigma2, h)
        
        # Compute denominators
        denom1 = weight1 * (erf(u_max / sqrt2) - erf(u_min / sqrt2))
        denom2 = weight2 * (erf(v_max / sqrt2) - erf(v_min / sqrt2))
        denominator = denom1 + denom2
        epsilon = 1e-10
        
        if abs(denominator) < epsilon:
            raise ValueError("Denominator is too close to zero.")
        
        # Compute numerators
        num1 = weight1 * (erf(u_right / sqrt2) - erf(u_left / sqrt2))
        num2 = weight2 * (erf(v_right / sqrt2) - erf(v_left / sqrt2))
        
        return (num1 + num2) / denominator

    except ValueError as e:
        print(f"ValueError in bin_integral: {e}")
        return np.nan
    except RuntimeWarning as e:
        print(f"RuntimeWarning in bin_integral: {e}")
        return np.nan

#----------------------------------------------------------------
# Objective function: minimize sum of squared residuals
def objective(params, bin_edges, window_means):
    """
    Objective function to minimize the sum of squared residuals between
    observed window means and predicted bin integrals.
    """
    mu1, sigma1, h, mu2, sigma2 = params

    # Penalize invalid parameters
    if sigma1 <= 0 or sigma2 <= 0 or h >= 135:
        return 1e10  # Large penalty for invalid parameters

    total_cost = 0.0
    for i, (left, right) in enumerate(bin_edges):
        predicted = bin_integral(left, right, mu1, sigma1, h, mu2, sigma2)
        if np.isnan(predicted):  # Handle invalid predictions
            return 1e10
        
        observation = window_means[i]
        residual = observation - predicted
        total_cost += residual ** 2

    return total_cost

#----------------------------------------------------------------
# Compute residuals and create a DataFrame for analysis
def compute_residuals(best_params, bin_edges, window_means):
    """
    Parameters:
        best_params (list): Optimal parameters [mu1, sigma1, h, mu2, sigma2].
        bin_edges (list of tuples): [(xleft, xright), ...], bin edges.
        window_means (list): Observed moving average values for each bin.
    """
    mu1, sigma1, h, mu2, sigma2 = best_params
    results = []
    
    for i, (left, right) in enumerate(bin_edges):
        predicted = bin_integral(left, right, mu1, sigma1, h, mu2, sigma2)
        observation = window_means[i]
        residual = observation - predicted
        
        # Avoid division by zero for residual ratio
        residual_ratio = residual / observation if np.abs(observation) > 1e-10 else np.nan

        results.append({
            'Bin': i,
            'Observation': np.round(observation, 4),
            'Predicted': np.round(predicted, 4),
            'Residual': np.round(residual, 4),
            'Residual Ratio': np.round(residual_ratio, 4)
        })

    return pd.DataFrame(results)

#----------------------------------------------------------------
# Main script
def fit_model(df_normalized, bin_edges, window_size=30, alpha=1):
    """
    Parameters:
        df_normalized (pd.DataFrame): Input data with normalized observations.
        bin_edges (list of tuples): Bin edges for integration.
        window_size (int): Rolling window size for moving average.
        alpha (float): Proportion of data to use (0 < alpha <= 1).
    """
    # Subset the data
    subset_size = int(alpha * len(df_normalized))
    df_subset = df_normalized.iloc[:subset_size]
    
    # Compute rolling means
    rolling_means = df_subset.rolling(window=window_size).mean().dropna()
    n_windows = len(rolling_means)
    n_bins = len(bin_edges)

    print(f"Number of windows in subset: {n_windows}")
    print(f"Number of bins: {n_bins}")
    
    # Initialize parameter trends and residuals storage
    trends = {param: [] for param in ['mu1', 'sigma1', 'h', 'mu2', 'sigma2']}
    residuals_list = []
    all_optimizations_successful = True

    # Sequential optimization for each rolling window
    for start in tqdm(range(n_windows), desc="Fitting model", unit="window"):
        window_means = rolling_means.iloc[start].values  # Observations for each bin at time t
    
        # Initial guess for parameters
        initial_params = [-1.0, 1.0, 100.0, 1.0, 1.0]
        bounds = [(None, None), (1e-6, None), (None, 135), (None, None), (1e-6, None)]
    
        # Minimize the objective function
        result = minimize(objective, initial_params, args=(bin_edges, window_means),
                          bounds=bounds, method='L-BFGS-B')
    
        if result.success:
            mu1_fit, sigma1_fit, h_fit, mu2_fit, sigma2_fit = result.x
            for param, value in zip(trends.keys(), result.x):
                trends[param].append(value)
            
            # Compute residuals for this window
            residuals_df = compute_residuals(result.x, bin_edges, window_means)
            residuals_df['Window'] = start  # Add window index for context
            residuals_list.append(residuals_df)
        else:
            all_optimizations_successful = False
            print(f"Optimization failed for window {start}: {result.message}")

    # Combine residuals into a single DataFrame
    if residuals_list:
        residuals_df = pd.concat(residuals_list, ignore_index=True)
    else:
        residuals_df = pd.DataFrame()  # Empty DataFrame if no successful optimizations

    # Print final status
    if all_optimizations_successful:
        print("-------All optimizations successful!------")
    else:
        print("Some optimizations failed. Review the logs for details.")

    return residuals_df, trends


#----------------------------------------------------------------

def optifit_model(df_normalized, bin_edges, window_size=30, alpha=1):
    # Subset the data (take the first alpha percentage of the data)
    subset_size = int(alpha * len(df_normalized))
    df_subset = df_normalized.iloc[:subset_size]
    
    # Compute rolling means
    rolling_means = df_subset.rolling(window=window_size, min_periods=1).mean().dropna()
    n_windows = len(rolling_means)
    n_bins = len(bin_edges)

    print(f"Number of windows in subset: {n_windows}")
    print(f"Number of bins: {n_bins}")
    
    # Initialize parameter trends and residuals storage
    trends = {param: np.zeros(n_windows) for param in ['mu1', 'sigma1', 'h', 'mu2', 'sigma2']}
    residuals_list = []
    all_optimizations_successful = True

    # Vectorized approach: Optimization over all windows at once
    for start in tqdm(range(n_windows), desc="Fitting model", unit="window"):
        window_means = rolling_means.iloc[start].values  # Observations for each bin at time t

        # Initial guess for parameters
        initial_params = [4.1, 0.37, 133, 3.1, 0.50]
        bounds = [(None, None), (1e-6, None), (None, 135), (None, None), (1e-6, None)]

        # Minimize the objective function
        result = minimize(objective, initial_params, args=(bin_edges, window_means),
                          bounds=bounds, method='L-BFGS-B')

        if result.success:
            # Store optimized parameters in the trends array
            trends['mu1'][start], trends['sigma1'][start], trends['h'][start], trends['mu2'][start], trends['sigma2'][start] = result.x

            # Compute residuals for this window
            residuals_df = compute_residuals(result.x, bin_edges, window_means)
            residuals_df['Window'] = start  # Add window index for context
            residuals_list.append(residuals_df)
        else:
            all_optimizations_successful = False
            print(f"Optimization failed for window {start}: {result.message}")

    # Combine residuals into a single DataFrame
    residuals_df = pd.concat(residuals_list, ignore_index=True) if residuals_list else pd.DataFrame()

    # Print final status
    if all_optimizations_successful:
        print("-------All optimizations successful!------")
    else:
        print("Some optimizations failed. Review the logs for details.")

    return residuals_df, trends

#----------------------------------------------------------------

# Plot the results
def plot_results(df_normalized, residuals_dfs, bin_edges, alpha=1):
    subset_size = int(alpha * len(df_normalized))
    df_subset = df_normalized.iloc[:subset_size]
    
    n_bins = len(bin_edges)
    time_index = sorted(residuals_dfs['Window'].unique())  # Unique time windows
    
    colors = sns.color_palette("husl", n_bins)
    plt.figure(figsize=(15, 8))
    
    for i in range(n_bins):
        # Extract real observations for the current bin
        real_observations = df_subset.iloc[:, i].values[:len(time_index)]
        
        # Filter the residuals DataFrame for the current bin and sort by time window
        bin_data = residuals_dfs[residuals_dfs['Bin'] == i].sort_values('Window')
        moving_avg_series = bin_data['Observation'].values
        predicted_series = bin_data['Predicted'].values
        
        # Plot the real observations (grayish color)
        plt.plot(time_index, real_observations, color='lightgray', alpha=0.5, linewidth=1.5)
        
        # Plot moving averages (solid lines with distinct colors)
        plt.plot(time_index, moving_avg_series, color=colors[i], linewidth=2, label=f'M.A Bin {i+1}')
        
        # Plot predicted values (black dotted line)
        plt.plot(time_index, predicted_series, linestyle=':', color='black', linewidth=2)
        
        # Label each moving average line
        plt.text(time_index[-1] + 1, moving_avg_series[-1], f'Bin {i+1}', fontsize=12, color=colors[i], va='center')
    
    # Add axis labels and title
    plt.xlabel('Time $t$ (s)', fontsize=16)
    plt.ylabel('Concentration $C^*$', fontsize=16)
    
    # Add legend with representative entries
    plt.plot([], [], color='lightgray', linewidth=1.5, alpha=0.5, label='Observations')
    plt.plot([], [], color='black', linestyle=':', linewidth=2, label='Predicted Values')
    plt.legend(loc='upper left', fontsize=14, title="Legend", title_fontsize=13, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('convex_combined_model.pdf', format='pdf', dpi=400)
    plt.show()

#----------------------------------------------------------------
def plot_param_trends(trend_dict):
    # List of parameter names, labels, and colors
    param_info = {
        'mu1': {'label': '$\\mu_1$', 'color': 'blue'},
        'sigma1': {'label': '$\\sigma_1$', 'color': 'green'},
        'h': {'label': '$h$', 'color': 'red'},
        'mu2': {'label': '$\\mu_2$', 'color': 'purple'},
        'sigma2': {'label': '$\\sigma_2$', 'color': 'cyan'}
    }

    for param, info in param_info.items():
        trend = trend_dict.get(param, np.array([]))  # Access the trend for the parameter
        
        if trend.size > 0:  # Check if the array has elements
            plt.figure(figsize=(9, 5))
            plt.scatter(np.arange(len(trend)), trend, label=info['label'], 
                        facecolors='none', edgecolors=info['color'], alpha=0.8)

            # Fit a linear regression model
            x_param = np.arange(len(trend)).reshape(-1, 1)
            model_param = LinearRegression().fit(x_param, trend)
            fit_line = model_param.predict(x_param)

            # Extract slope and intercept
            slope = model_param.coef_[0]
            intercept = model_param.intercept_

            # Plot regression line
            # plt.plot(np.arange(len(trend)), fit_line, 
            #          label=f'{info["label"]} = {slope:.2g}t + {intercept:.2f}', 
            #          color='orange')

            # Customize plot
            plt.xlabel('Moving window $s=t+\\tau-1$ (s)')
            plt.ylabel(info['label'])
            #plt.title(f'Trend of {info["label"]} over Time', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best', frameon=True, shadow=True, title="Legend", title_fontsize=16)
            plt.tight_layout()

            # Save and display the plot
            plt.savefig(f"{param}-trend.pdf", format="pdf", dpi=400)
            plt.show()
        else:
            print(f"No data available for {param}.")

#-------------------------------------------------------
#                   OLD SCRIPTS
#-------------------------------------------------------

# Main Script
def fit_model2(df_normalized, bin_edges, window_size=30, alpha=1):
    # Take the first 100alpha % of the data as a subset
    subset_size = int(alpha * len(df_normalized))
    df_subset = df_normalized.iloc[:subset_size]
    
    # Compute the moving averages (rolling means) for each column in the subset
    rolling_means = df_subset.rolling(window=window_size).mean().dropna()
    
    n_columns = len(df_subset.columns)
    n_windows = len(rolling_means)
    n_bins = len(bin_edges)
    print(f"Number of windows in subset: {n_windows}")
    print(f"Number of columns in subset: {n_columns}")
    
    # Sequential optimization for each rolling window in the subset
    mu1_trend = []
    sigma1_trend = []
    h_trend = []
    mu2_trend = []
    sigma2_trend = []
    residuals_dfs = []  # Store the residuals DataFrames
    
    all_optimizations_successful = True  # Track optimization success
    
    # Loop through each rolling window with a progress bar
    for start in tqdm(range(n_windows), desc="Fitting model", unit="window"):
        window_means = rolling_means.iloc[start].values
        if len(window_means) != n_bins:
            print(f"Error: window_means length ({len(window_means)}) does not match the number of bins ({n_bins}). Skipping this window.")
            continue
        
        initial_params = [3.6, 0.75, 134, 4.5, 0.53]  # Initial guess for [mu1, sigma1, h, mu2, sigma2]
        result = minimize(objective, initial_params, args=(bin_edges, window_means),
                          bounds=[(None, None), (1e-6, None), (None, 135),
                                  (None, None), (1e-6, None)],
                          method='L-BFGS-B')
        
        if result.success:
            mu1_fit, sigma1_fit, h_fit, mu2_fit, sigma2_fit = result.x
            mu1_trend.append(mu1_fit)
            sigma1_trend.append(sigma1_fit)
            mu2_trend.append(mu2_fit)
            sigma2_trend.append(sigma2_fit)
            h_trend.append(h_fit)
            
            # Compute residuals for this window and store the DataFrame
            residuals_df = compute_residuals([mu1_fit, sigma1_fit, h_fit, mu2_fit, sigma2_fit], bin_edges, window_means)
            residuals_dfs.append(residuals_df)
        else:
            all_optimizations_successful = False
            print(f"Optimization failed for window {start}. Reason: {result.message}")
    
    # Print the success message once if all optimizations succeeded
    if all_optimizations_successful:
        print("-------All optimizations successful!------")
    
    return residuals_dfs, mu1_trend, sigma1_trend, mu2_trend, sigma2_trend, h_trend


# Plot the fitted trends over time in the subset
def plot_results_perbin(df_normalized, residuals_dfs, bin_edges, alpha=1):
    # Take the first 100alpha % of the data as a subset
    subset_size = int(alpha * len(df_normalized))
    df_subset = df_normalized.iloc[:subset_size]
    # Number of bins (plots)
    n_bins = len(bin_edges)
    
    # Time Series Plots for Each Bin (with Real Observations, Moving Averages, and Predicted Values)
    for i in range(n_bins):
        # Extract time series data for bin i
        real_observations = df_subset.iloc[:, i].values  # observations for bin i
        moving_avg_series = [df.iloc[i]['Observation'] for df in residuals_dfs]  # Moving averages
        predicted_series = [df.iloc[i]['Predicted'] for df in residuals_dfs]  # Predicted values
    
        # Create a time index corresponding to each window
        time_index = np.arange(len(moving_avg_series))
    
        # Plot the real observations, moving averages, and predicted values
        plt.figure(figsize=(12, 5))
        plt.plot(real_observations[:len(time_index)], label='Observations', color='red', alpha=0.3)
        plt.plot(time_index, moving_avg_series, label='Moving Average', color='blue')
        plt.plot(time_index, predicted_series, label='Predicted Values', color='green', linestyle='dashed')
    
        # Set plot labels and title
        plt.xlabel('Time $t~(s)$')
        plt.ylabel('Concentration $C^{*}$')
        #plt.ylim((0,0.30))   # limit the y axis values
        plt.title(f'Bin {i+1}: Real Observations vs. Moving Averages vs. Predicted Values')
        plt.legend()
        plt.savefig(f'plot_bin_{i+1}.png', format='png')  # Save the plots to a file
        plt.show()
    
    # Concatenate all residuals DataFrames into a single DataFrame for further analysis
    if residuals_dfs:
        all_residuals_df = pd.concat(residuals_dfs, ignore_index=True)
        print(all_residuals_df.head())
    else:
        print("No residuals DataFrames were generated.")


def plot_results2(df_normalized, residuals_dfs, bin_edges, alpha=1):
    # Take the first 100alpha % of the data as a subset
    subset_size = int(alpha * len(df_normalized))
    df_subset = df_normalized.iloc[:subset_size]
    # Number of bins (plots)
    n_bins = len(bin_edges)
    
    # Define a color palette for moving averages (distinct but visible)
    colors = sns.color_palette("husl", n_bins)  # Use a distinct color palette for each bin
    
    # Create a single figure
    plt.figure(figsize=(15, 8))
    
    # Plot data for each bin with three types of lines (Observations, Moving Averages, Predicted Values)
    for i in range(n_bins):
        # Extract time series data for bin i
        real_observations = df_subset.iloc[:, i].values  # Real noisy observations for bin i
        moving_avg_series = [df.iloc[i]['Observation'] for df in residuals_dfs]  # Moving averages
        predicted_series = [df.iloc[i]['Predicted'] for df in residuals_dfs]  # Predicted values
    
        # Create a time index corresponding to each window
        time_index = np.arange(len(moving_avg_series))
    
        # Plot real observations (grayish color)
        plt.plot(real_observations[:len(time_index)], color='lightgray', alpha=0.5, linewidth=1.5)
    
        # Plot moving averages (solid lines with distinct colors for each bin)
        plt.plot(time_index, moving_avg_series, color=colors[i], linewidth=2)#, label=f'Bin {i+1}')
    
        # Plot predicted values (black dotted line, same for all bins)
        plt.plot(time_index, predicted_series, linestyle=':', color='black', linewidth=2)
    
        # Label each moving average line with its corresponding bin number next to the end of the line
        plt.text(time_index[-1] + 1, moving_avg_series[-1], f'Bin {i+1}', fontsize=14, color=colors[i], va='center')
    
    # Add labels for the axes and the title
    plt.xlabel('Time $t~(s)$', fontsize=16, labelpad=10)
    plt.ylabel('Concentration $C^{*}$', fontsize=16, labelpad=10)
    
    # Create a simplified legend with only three entries
    plt.plot([], [], color='gray', linewidth=1.5, label='Observations')  # Dummy plot for legend
    plt.plot([], [], color='blue', linewidth=2, label='Moving Average')  # Color placeholder, real moving average colors are distinct
    plt.plot([], [], color='black', linestyle=':', linewidth=2, label='Predicted Values')  # Dummy plot for predictions
    
    # Add the legend with only 3 entries
    plt.legend(loc='upper left', fontsize=14, frameon=True, shadow=True, title="Legend", title_fontsize=13, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Set tight layout for good spacing and display the plot
    plt.tight_layout()
    
    # Save the combined plot to a file
    plt.savefig('convex_combined_model.pdf', format='pdf', dpi=400)
    
    # Optionally show the plot
    plt.show()


def param_trend2(mu1_trend, sigma1_trend, mu2_trend, sigma2_trend, h_trend):
    # List of parameter names and trends
    params = ['mu1', 'sigma1', 'h', 'mu2', 'sigma2']
    trends = [mu1_trend, sigma1_trend, h_trend, mu2_trend, sigma2_trend]
    colors = ['blue', 'green', 'red', 'purple', 'cyan', 'magenta']
    labels = ['$\\mu_1$', '$\\sigma_1$', '$h$', '$\\mu_2$', '$\\sigma_2$']
    
    for param, trend, color, label in zip(params, trends, colors, labels):
        plt.figure(figsize=(9, 5))
        plt.scatter(np.arange(len(trend)), trend, label=label, facecolors='none', edgecolors=color)
    
        # Fit a linear regression model
        x_param = np.arange(len(trend)).reshape(-1, 1)
        model_param = LinearRegression().fit(x_param, trend)
        fit_line = model_param.predict(x_param)
    
        # Extract slope and intercept
        slope = model_param.coef_[0]
        intercept = model_param.intercept_
    
        # Plot regression line
        plt.plot(np.arange(len(trend)), fit_line, 
                 label=f'Regression Line: {label} = {slope:.2g}t + {intercept:.2f}', 
                 color='orange')
        plt.xlabel('Moving window $\\tau$ (s)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend(loc='best', fontsize=14, frameon=True, shadow=True, title="Legend", title_fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{param}-trend.pdf", format="pdf", dpi=400)
        plt.show()
