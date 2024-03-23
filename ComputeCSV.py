import csv
import matplotlib.pyplot as plt

# Open the CSV file
with open('ICUSTAYS.csv', 'r') as file:

    # Read the CSV data using the csv reader
    reader = csv.reader(file)
    
    # Skip the header row
    next(reader)

    # Create empty lists to store the data
    x_data = []
    y_data = []

    # Loop through the rows of the CSV file
    for row in reader:
        # Convert the x and y data to float and append to the lists
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

# Create a line graph using Matplotlib
plt.plot(x_data, y_data)

# Add labels to the graph
plt.title('My Graph')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Display the graph
plt.show()
