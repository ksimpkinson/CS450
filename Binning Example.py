# Binning

# Get a random list of 100 numbers
# All between 0 and 1
data = np.random.random(100)

# Create 10 numbers evenly spaced out between 0 and 1
# Essentially, create bins
bins = np.linspace(0, 1, 10)
# This also works:
#  0 <= number < .3 = bin1
# .3 <= number < .6 = bin2
# .6 <= number <= 1 = bin3 
bins = np.array([0, .3, .6, 1])

# Categorized each data point as one of the bins
digitized = np.digitize(data, bins)

# Calculate the average for each bin
bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]