import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.datasets

data = pd.read_csv(
    "/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/spaceship-titanic/train.csv")

data_chosen = data[["HomePlanet", "CryoSleep", "Destination", "Age", "VIP", "Transported"]]

impute_data = {"HomePlanet": data_chosen.mode()["HomePlanet"][0],
               "CryoSleep": data_chosen.mode()["CryoSleep"][0],
               "Destination": data_chosen.mode()["Destination"][0],
               "Age": data_chosen["Age"].mean(),
               "VIP": data_chosen.mode()["VIP"][0]
               }

data_chosen = data_chosen.fillna(impute_data)
X_train = data_chosen[["CryoSleep", "Age", "VIP"]]
y_train = data_chosen["Transported"]
X_train = pd.concat([X_train, pd.get_dummies(data_chosen["Destination"], prefix="Des")], axis=1)
X_train['Mars'] = np.array([x == 'Mars' for x in data_chosen["HomePlanet"]])
X_train['Europa'] = np.array([x == 'Europa' for x in data_chosen["HomePlanet"]])
X_train['Earth'] = np.array([x == 'Earth' for x in data_chosen["HomePlanet"]])

##################################################################
reg = sk.linear_model.LogisticRegression().fit(X_train.astype('float64'), y_train.astype('float64'))

############################################################
data_val = pd.read_csv(
    "/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/spaceship-titanic/test.csv")

data_chosen_val = data_val[["HomePlanet", "CryoSleep", "Destination", "Age", "VIP"]]
impute_data = {"HomePlanet": data_chosen_val.mode()["HomePlanet"][0],
               "CryoSleep": data_chosen_val.mode()["CryoSleep"][0],
               "Destination": data_chosen_val.mode()["Destination"][0],
               "Age": data_chosen_val["Age"].mean(),
               "VIP": data_chosen_val.mode()["VIP"][0]
               }

data_chosen_val = data_chosen_val.fillna(impute_data)
X_val = data_chosen_val[["CryoSleep", "Age", "VIP"]]
X_val = pd.concat([X_val,
                   pd.get_dummies(data_chosen_val["HomePlanet"]),
                   pd.get_dummies(data_chosen_val["Destination"])], axis=1)

y_pred = reg.predict(X_val)
y_pred = np.array([x==1 for x in y_pred])

df_sub = pd.DataFrame({'PassengerId': data_val['PassengerId'], 'Transported': y_pred})
df_sub.to_csv("/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/spaceship-titanic/testSub.csv", index=False)

