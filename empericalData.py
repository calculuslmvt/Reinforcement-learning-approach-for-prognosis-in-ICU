import pandas as pd
import matplotlib.pyplot as plt

def calculate_mean(filename, column_name):
    """
    Input a CSV file and calculate the mean of a specified column.

    Parameters:
    - filename: the path to the CSV file.
    - column_name: the name of the column for which to calculate the mean.

    Returns:
    - The mean of the specified column.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' not found in the CSV file.")

    # Calculate the mean for the specified column
    mean_value = df[column_name].mean()

    return mean_value

def plot_histogram(data, column_name):
    """
    Plot a histogram for the given data.

    Parameters:
    - data: pandas Series containing the data to be plotted.
    - column_name: the name of the column for labeling the plot.
    """
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column_name}')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Hardcoded filename and column name
    filename = "data.csv"
    column_name = input("Enter the column name for which to calculate the mean: ")

    # Load the CSV file
    df = pd.read_csv(filename)

    # Calculate and print the mean for the specified column
    try:
        mean_value = calculate_mean(filename, column_name)
        print(f"The mean of the column '{column_name}' is: {mean_value}")

        # Plot a histogram for the specified column
        plot_histogram(df[column_name], column_name)

    except Exception as e:
        print(f"Error: {e}")
