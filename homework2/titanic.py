import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model

data_dir = "/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/titanicContest/titanic/"

train_data = pd.read_csv(data_dir + 'train.csv')
test_data = pd.read_csv(data_dir + 'test.csv')

df_train = train_data[['Age', 'SibSp', 'Fare', 'Cabin']]
lables_train = train_data['Survived'].astype('float64')

impute_vals = {'Age': df_train['Age'].mean(), 'Fare': df_train['Fare'].mean(), 'Cabin': 'U'}

df_train = df_train.fillna(impute_vals)

cabins = [s for s in df_train['Cabin']]
decks = [c[0] for c in cabins]

df_train['isFemale'] = (train_data['Sex'] == 'female').astype('float64')
df_train['A'] = np.float64([x == 'A' for x in decks])
df_train['B'] = np.float64([x == 'B' for x in decks])
df_train['C'] = np.float64([x == 'C' for x in decks])
df_train['D'] = np.float64([x == 'D' for x in decks])
df_train['E'] = np.float64([x == 'E' for x in decks])
df_train['F'] = np.float64([x == 'F' for x in decks])
df_train['G'] = np.float64([x == 'G' for x in decks])
df_train['T'] = np.float64([x == 'T' for x in decks])
df_train['U'] = np.float64([x == 'U' for x in decks])
df_train = df_train.drop('Cabin', axis=1)
df_train['SibSp']=df_train['SibSp'].astype('float64')

## Add three features, embarked (Q,S,C)

df_train['EmbarkedQ'] = (train_data['Embarked'] == 'Q').astype('float64')
df_train['EmbarkedS'] = (train_data['Embarked'] == 'S').astype('float64')
df_train['EmbarkedC'] = (train_data['Embarked'] == 'C').astype('float64')


## Add seven features, parch (0,1,2,3,4,5,6)
df_train['parch0'] = (train_data['Parch'] == 0).astype('float64')
df_train['parch1'] = (train_data['Parch'] == 1).astype('float64')
df_train['parch2'] = (train_data['Parch'] == 2).astype('float64')
df_train['parch3'] = (train_data['Parch'] == 3).astype('float64')
df_train['parch4'] = (train_data['Parch'] == 4).astype('float64')
df_train['parch5'] = (train_data['Parch'] == 5).astype('float64')
df_train['parch6'] = (train_data['Parch'] == 6).astype('float64')



###############

model = sk.linear_model.LogisticRegression(max_iter = 500)
model.fit(df_train, lables_train)

#####################
test_data = test_data.fillna(impute_vals)

df_test = test_data[['Age', 'SibSp', 'Fare']].astype('float64')
df_test['isFemale'] = (test_data['Sex'] == 'female').astype('float64')

cabins = [s for s in test_data['Cabin']]
decks = [c[0] for c in cabins]
df_test['A'] = np.float64([x == 'A' for x in decks])
df_test['B'] = np.float64([x == 'B' for x in decks])
df_test['C'] = np.float64([x == 'C' for x in decks])
df_test['D'] = np.float64([x == 'D' for x in decks])
df_test['E'] = np.float64([x == 'E' for x in decks])
df_test['F'] = np.float64([x == 'F' for x in decks])
df_test['G'] = np.float64([x == 'G' for x in decks])
df_test['T'] = np.float64([x == 'T' for x in decks])
df_test['U'] = np.float64([x == 'U' for x in decks])

df_test['EmbarkedQ'] = (test_data['Embarked'] == 'Q').astype('float64')
df_test['EmbarkedS'] = (test_data['Embarked'] == 'S').astype('float64')
df_test['EmbarkedC'] = (test_data['Embarked'] == 'C').astype('float64')

df_test['parch0'] = (test_data['Parch'] == 0).astype('float64')
df_test['parch1'] = (test_data['Parch'] == 1).astype('float64')
df_test['parch2'] = (test_data['Parch'] == 2).astype('float64')
df_test['parch3'] = (test_data['Parch'] == 3).astype('float64')
df_test['parch4'] = (test_data['Parch'] == 4).astype('float64')
df_test['parch5'] = (test_data['Parch'] == 5).astype('float64')
df_test['parch6'] = (test_data['Parch'] == 6).astype('float64')



labels_pred = model.predict(df_test)

d_sub = {'PassengerId': test_data['PassengerId'], 'Survived': labels_pred.astype('int64')}
df_sub = pd.DataFrame(d_sub)
df_sub.to_csv(data_dir + 'JiazhengSub.csv', index=False)
