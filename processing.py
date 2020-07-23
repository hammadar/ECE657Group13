import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np


df = pd.read_csv("https://opendatatc.blob.core.windows.net/opendatatc/NCDB_1999_to_2017.csv")
df["P_SEX"].replace({"M":0, "F":1}, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.astype("Int64")
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

y = df["C_CONF"]
X = df.drop(["C_CONF", "C_MNTH", "C_WDAY", "C_HOUR", "V_ID"], axis=1)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3)
imp = SimpleImputer(missing_values=np.nan,strategy="mean")
X_train = imp.fit_transform(X_train)
y_train = y_train.to_numpy().reshape(-1,1)


accuracies = {}

#for i in range(200):
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
#accuracies[i] = accuracy

'''lists = sorted(accuracies.items())
x,y = zip(*lists)
plt.plot(x,y)
plt.show()'''




