# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
# ]
# ///

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Detect outliers using IQR (updated to handle numeric columns only)
def detect_outliers_iqr(dataframe, column):
    if not np.issubdtype(dataframe[column].dtype, np.number):  # Skip non-numeric columns
        return pd.DataFrame(columns=dataframe.columns)  # Return empty DataFrame
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]

# Remove outliers using IQR (updated to handle numeric columns only)
def remove_outliers_iqr(dataframe, column):
    if not np.issubdtype(dataframe[column].dtype, np.number):  # Skip non-numeric columns
        return dataframe
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def preprocessing(df):
    df = df.dropna()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = pd.concat([detect_outliers_iqr(df, col) for col in numeric_columns], ignore_index=True)
    print("Outliers:\n", outliers)

    # Remove outliers in all numeric columns
    cleaned_df = df.copy()
    for col in numeric_columns:
        cleaned_df = remove_outliers_iqr(cleaned_df, col)

    # Visualization
    plt.figure(figsize=(10, 6))

    # Box plot for visualization
    plt.boxplot([df[col] for col in numeric_columns], labels=numeric_columns)
    plt.title('Box Plot of Features with Outliers')
    plt.savefig('outliers_boxplot.png')  # Save as PNG
    plt.show()

    # Scatter plot for cleaned data
    plt.figure(figsize=(10, 6))
    for col in numeric_columns:
        plt.scatter(range(len(cleaned_df)), cleaned_df[col], label=col, alpha=0.7)
    plt.legend()
    plt.title('Scatter Plot after Removing Outliers')
    plt.savefig('cleaned_data_scatter.png')  # Save as PNG
    plt.show()
    return cleaned_df

def process_dataset(file_path):
    """
    Process the dataset given as a file path.
    """
    try:
        # Example: Read and display the dataset using pandas
        df = pd.read_csv(file_path)
        print("Dataset Preview:",file_path)
        #print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset file.")
    parser.add_argument("dataset", help="Path to the dataset file")

    # Parse the arguments
    args = parser.parse_args()
    df = process_dataset(args.dataset)
    df = preprocessing(df)
    # Correlation Analysis (Numeric Columns Only)
    numeric_df = df.select_dtypes(include=[np.number])  # Select numeric columns
    correlation_matrix = numeric_df.corr()
    # Save correlation matrix to a DataFrame
    correlation_df = correlation_matrix.reset_index()  # Reset index for a tabular format
    print("Correlation Matrix:\n", correlation_df)

    # Save the correlation matrix to a CSV file (optional)
    correlation_df.to_csv("correlation_matrix.csv", index=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix Heatmap")
    plt.savefig("correlation_heatmap.png")  # Save as PNG
    plt.show()
