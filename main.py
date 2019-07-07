import correlation_matrix;
import csv;

data = [];

f = open('voxels_only.csv', 'r');
# create reader
reader = csv.reader(f, delimiter=';');

for row in reader:
    data.append(list(map(lambda x: float(x), row)));

# transpose the data
data = list(zip(*data));

matrix = correlation_matrix.calculate_correlation_matrix(data);
correlation_matrix.pretty_print(matrix[1:len(matrix)+1]);
