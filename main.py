import correlation_matrix;
import csv;
import scipy.cluster.hierarchy as spc

import pandas;


data = [];

f = open('voxels_only.csv', 'r');
# create reader
reader = csv.reader(f, delimiter=';');
cols = next(reader);
for row in reader:
    data.append(list(map(lambda x: float(x), row)));

# transpose the data

#matrix = correlation_matrix.calculate_correlation_matrix(data);

df = pandas.DataFrame(data = data, columns = cols);
#matrix_full = correlation_matrix.generate_full_matrix(matrix[1:len(matrix)+1]);
#correlation_matrix.pretty_print(matrix[1:len(matrix)+1]);

corr = df.corr()


pdist = spc.distance.pdist(corr)
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')

for val in zip(cols,idx):
    print("{0} is in cluster {1} ".format(val[0], val[1]))
