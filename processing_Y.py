import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np


df = pd.read_csv("C:\\Users\\YAM\\Documents\\NCDB_1999_to_2017.csv",dtype=object)
y = df["C_CONF"].to_frame()
X = df.drop(["C_CONF","V_ID","C_CASE","C_CONF","C_HOUR", "C_WDAY", "C_MNTH"], axis=1)
yfnp = y.values.tolist()
dfnp = X.values.tolist()
xch = X.columns.values.tolist()
ych = y.columns.values.tolist()
ti = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
ti.fit_on_texts(dfnp)
cdata = ti.texts_to_sequences(dfnp)
cdata = preprocessing.sequence.pad_sequences(cdata)
sc = MinMaxScaler(feature_range=(0, 1))
cdata = sc.fit_transform(cdata)
Xf = pd.DataFrame(data=cdata,columns=xch,dtype =float)
to = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
to.fit_on_texts(yfnp)
ydata = to.texts_to_sequences(yfnp)
yf = pd.DataFrame(data=ydata,columns=ych,dtype =float)
#df = df.fillna(method="bfill")
'''df["C_MNTH"].replace({"UU":pd.NA, "XX":pd.NA}, inplace=True)
df["C_WDAY"].replace({"U":pd.NA, "X":pd.NA}, inplace=True)
df["C_HOUR"].replace({"UU":pd.NA, "XX":pd.NA}, inplace=True)
df["C_SEV"].replace({"U":pd.NA, "X":pd.NA}, inplace=True)
df["C_VEHS"].replace({"UU":pd.NA, "XX":pd.NA}, inplace=True)
df["C_CONF"].replace({"QQ": pd.NA, "UU":pd.NA, "XX":pd.NA}, inplace=True)
df["C_RCFG"].replace({"QQ":pd.NA, "UU":pd.NA, "XX":pd.NA}, inplace=True)
df["C_WTHR"].replace({"Q":pd.NA, "U":pd.NA, "X":pd.NA}, inplace=True)
df["C_RSUR"].replace({"Q":pd.NA, "U":pd.NA, "X":pd.NA}, inplace=True)
df["C_RALN"].replace({"Q":pd.NA, "U":pd.NA, "X":pd.NA}, inplace=True)
df["C_TRAF"].replace({"QQ":pd.NA, "UU":pd.NA, "XX":pd.NA}, inplace=True)
df["V_ID"].replace({"UU":pd.NA}, inplace=True)
df["V_TYPE"].replace({"QQ":pd.NA, "UU":pd.NA, "XX":pd.NA, "NN":pd.NA}, inplace=True)
df["V_YEAR"].replace({"NNNN":pd.NA, "UUUU":pd.NA, "XXXX":pd.NA}, inplace=True)
df["P_ID"].replace({"NN":pd.NA, "UU":pd.NA}, inplace=True)
df["P_SEX"].replace({"N":pd.NA, "U":pd.NA, "X":pd.NA, "M":"0", "F":"1"}, inplace=True)
df["P_AGE"].replace({"NN":pd.NA, "UU":pd.NA, "XX":pd.NA}, inplace=True)
df["P_PSN"].replace({"QQ":pd.NA, "UU":pd.NA, "XX":pd.NA, "NN":pd.NA}, inplace=True)
df["P_ISEV"].replace({"N":pd.NA, "U":pd.NA, "X":pd.NA}, inplace=True)
df["P_SAFE"].replace({"QQ":pd.NA, "UU":pd.NA, "XX":pd.NA, "NN":pd.NA}, inplace=True)
df["P_USER"].replace({"U":pd.NA}, inplace=True)'''

X_train, X_test, y_train, y_test =  train_test_split(Xf, yf, test_size=0.1)


accuracies = {}

#for i in range(200):
clf = RandomForestClassifier(n_estimators=150,verbose=1)
clf.fit(X_test, y_test)
y_pred = clf.predict(X_train)
accuracy = metrics.accuracy_score(y_train, y_pred)
print("Accuracy: ", accuracy*100)
'''
lists = sorted(accuracies.items())
x,y = zip(*lists)
plt.plot(x,y)
plt.show()
'''


