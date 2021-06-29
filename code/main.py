import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("train.csv")
strs = df['target'].value_counts()
value_map = (dict(v, i) for i,v in enumerate(strs.index))
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
df = df.replace({'target': value_map})
x_train = df.iloc[:, 1:-1]
y_train = df['target']
df = pd.read_csv("test.csv")
x_test = df.iloc[:,1:]
model = RandomForestClassifier(n_jobs=2, n_estimators=400)
model.fit(x_train, y_train)
proba = model.predict_proba(x_test)
output = pd.DataFrame({'id': df['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5':proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})
output.to_csv('my_submission_rf.csv', index=False)