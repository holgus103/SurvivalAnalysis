import csv
import numpy
import random
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt;


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
    data.append(numpy.array(list(map(convertToFloat, row[2:len(row)]))))
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
train_data = diabetic[0:int(0.66 * len(diabetic))] + non_diabetic[0:int(0.66 * len(non_diabetic))];
test_data = diabetic[int(0.66 * len(diabetic)) : len(diabetic)] + non_diabetic[int(0.66 * len(non_diabetic)) : len(non_diabetic)];

_train_d, _train_l = zip(*train_data);
_test_d, _test_l = zip(*test_data);

_train_d = list(_train_d)
_test_d = list(_test_d)

# plot some lame estimator stuff
tf, time = zip(*_train_l);
x, y = kaplan_meier_estimator(tf, time);

plt.step(x, y, where="post", label="Train data");

tft, timet = zip(*_test_l);
xt, yt = kaplan_meier_estimator(tft, timet);

plt.step(xt, yt, where="post", label="Test data");

plt.legend();

plt.plot();

    _train_l = numpy.array(list(_train_l), dtype='bool,f4');

    _test_l = numpy.array(list(_test_l), dtype='bool,f4');

# create ph model
estimator = CoxPHSurvivalAnalysis();

estimator.fit(_train_d, _train_l)

# create the cox model
clf = CoxnetSurvivalAnalysis(n_alphas=5, tol=0.1)

# train model 
clf.fit(_train_d, _train_l);

result = [];
# evaluate for every alpha
for v in clf.alphas_:
    res = clf.predict(_test_d, alpha=[v])
    result.append(concordance_index_censored(tft, timet, res))



# calculate precision
clf.predict(_test_d);
res= clf.predict(_test_d);


# print out some results
print(clf.coef_)
print(res)
# save coeficientos

#numpy.savetxt("coefsWithLabels.txt", list(map(lambda x : [str(x[0]),str(x[1])], zip(clf.coef_, header[2:len(header)]))), fmt="%s")
#numpy.savetxt("res.txt", list(map(lambda x : [str(x[0]),str(x[1])], zip(clf.predict(test_data), header[2:len(header)]))), fmt="%s")



