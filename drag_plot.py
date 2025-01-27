import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store hit distances and years
distance = []
year = []

# Read and process the CSV file
with open("astros_drag.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        if "NA" in row.values():
            continue  # Skip rows with any NA values
        try:
            # Convert hit_distance to float and round to 3 decimal places
            hit_dist = round(float(row["hit_distance"]), 3)
            distance.append(hit_dist)
            year.append(row["year"])
        except ValueError:
            # Handle cases where conversion to float fails
            continue

# Organize hit distances by year using defaultdict
data_by_year = defaultdict(list)
for y, d in zip(year, distance):
    data_by_year[y].append(d)

# Sort the years for consistent ordering in the plot
sorted_years = sorted(data_by_year.keys())

# Prepare the data for the boxplot
data_to_plot = [data_by_year[y] for y in sorted_years]

# Calculate mean values for each year
mean_values = [np.mean(data_by_year[y]) for y in sorted_years]

# Create the boxplot with mean
plt.figure(figsize=(12, 8))  # Optional: Adjust the figure size as needed
box = plt.boxplot(
    data_to_plot,
    labels=sorted_years,
    patch_artist=True,
    showmeans=True,  # Enable mean display
    meanprops={
        'marker': 'D',     # Diamond marker for mean
        'markerfacecolor': 'yellow',
        'markeredgecolor': 'black',
        'markersize': 8
    },
    boxprops=dict(facecolor='skyblue', color='blue'),
    medianprops=dict(color='red'),
    whiskerprops=dict(color='blue'),
    capprops=dict(color='blue')
)

# Add titles and labels
plt.title('Boxplot of Hit Distance by Year with Mean')
plt.xlabel('Year')
plt.ylabel('Hit Distance')

# Rotate x-axis labels if there are many years to prevent overlap
plt.xticks(rotation=45)

# Optional: Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# Display Mean Values Numerically
print("Mean Hit Distance by Year:")
print("----------------------------")
print("{:<10} {:>15}".format("Year", "Mean Hit Distance"))
print("----------------------------")
for y, m in zip(sorted_years, mean_values):
    print("{:<10} {:>15.3f}".format(y, m))


import pandas as pd
from scipy import stats

# Example data: hit distances and corresponding years


# Create a DataFrame from the lists
df = pd.DataFrame({
    'hit_distance': distance,
    'year': year
})

# Group the data by year and create a list of hit distances for each year
groups = [df[df['year'] == year]['hit_distance'] for year in df['year'].unique()]

# Perform the ANOVA test
anova_result = stats.f_oneway(*groups)

# Output the result
print("ANOVA test result:", anova_result)
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Interpretation of the p-value
if anova_result.pvalue < 0.05:
    print("There is a significant difference in hit distances across the years.")
    tukey_result = pairwise_tukeyhsd(df['hit_distance'], df['year'], alpha=0.05)
    print(tukey_result)
else:
    print("There is no significant difference in hit distances across the years.")
