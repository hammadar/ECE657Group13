import pandas as pd
import numpy as np


df = pd.read_csv("https://opendatatc.blob.core.windows.net/opendatatc/NCDB_1999_to_2017.csv")
df["P_SEX"].replace({"M":0, "F":1}, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.astype("Int64")
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


