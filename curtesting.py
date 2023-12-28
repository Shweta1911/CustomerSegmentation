import matplotlib.pyplot as plt

# Sample data
categories = ['Collaborative Filtering', 'SVD']
values = [0.59,1.0]

# Create a bar graph
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Precison Values')

# Show the plot
plt.show()
