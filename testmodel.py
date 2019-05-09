import csv
import numpy
import random
from sksurv.linear_model import CoxnetSurvivalAnalysis


# converts cells to floats, empty cells become 0
def convertToFloat(x):
    if x == '' or x == '-':
        return 0
    else:
        return float(x)

# open data
f = open('base_diabete_lasso.csv', 'r')
# create reader
reader = csv.reader(f, delimiter=';')
# skip the first line because it has only labels
header = next(reader)

# create a list for all features
data = []
# create a list for labels (got sick or not)
label = []

# for every row in the data
for row in reader:
    # convert every feature cell
    data.append(numpy.array(list(map(convertToFloat, row[1:len(row)]))))
    # get the label and add it to the label list
    label.append(('1' == row[0], convertToFloat(row[1])))

# transform label data to required format

d = list(zip(data, label));

diabetic = []
non_diabetic = []

for val in d:
    if(val[0][0]):
        diabetic.append(val);
    else:
        non_diabetic.append(val);
    

# shuffle data
random.shuffle(diabetic);
random.shuffle(non_diabetic);

# split data and labels
train_data = diabetic[0:int(0.66 * len(data))] + non_diabetic[0:int(0.66 * len(data))];
test_data = diabetic[int(0.66 * len(data)) : len(data)] + non_diabetic[int(0.66 * len(data)) : len(data)];

data, label = zip(*d);

data = list(data);
label = list(label);



train_label = label[0:int(0.66 * len(data))]
train_label = numpy.array(train_label, dtype='bool,f4');

test_label = label[int(0.66 * len(data)) + 1 : len(data)];
test_label = numpy.array(test_label, dtype='bool,f4');



# create the cox model
clf = CoxnetSurvivalAnalysis(n_alphas=5, tol=0.1)

# train the model
clf.fit(train_data, train_label);

# calculate precision
clf.predict(test_data);
res= clf.predict(test_data);


# print out some results
print(clf.coef_)
print(res)
# save coeficientos

#numpy.savetxt("coefsWithLabels.txt", list(map(lambda x : [str(x[0]),str(x[1])], zip(clf.coef_, header[2:len(header)]))), fmt="%s")
#numpy.savetxt("res.txt", list(map(lambda x : [str(x[0]),str(x[1])], zip(clf.predict(test_data), header[2:len(header)]))), fmt="%s")


#check the value of alphas
clf.alphas_

