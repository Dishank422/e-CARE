import pandas as pd
import numpy as np
data = pd.read_csv("CEQ.csv")
c_e_x = data[["Cause+Truth:Effect", "Cause:Truth+Effect"]].max(axis=1).to_numpy()
c_e = data["Cause:Effect"].to_numpy()
print("CEQ score =", np.mean(c_e_x-c_e))
