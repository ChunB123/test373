import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model

data_dic = "/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/digit-recognizer/"

train_data = pd.read_csv(data_dic + "train.csv")
test_data = pd.read_csv(data_dic + "test.csv")

lables_train = train_data['label']

df_train = train_data[train_data.columns[1:]]

##############
model = sk.linear_model.LogisticRegression(max_tier = 100)
model.fit(df_train, lables_train)
##################


labels_pred = model.predict(test_data)

ImageId = np.array([x+1 for x in range(len(labels_pred))])
d_sub = {'ImageId': ImageId, 'Label': labels_pred}
df_sub = pd.DataFrame(d_sub)

df_sub.to_csv(data_dic + 'JiazhengSub.csv', index=False)