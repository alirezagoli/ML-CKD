import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt


data=pd.read_csv("chronic_kidney_disease.csv")
# Some information about data
print("###########################################################################")
print("Data Shape:")
print(data.shape)
print("###########################################################################")
print("Data Description:")
print(data.describe(include="all"))
print("###########################################################################")
print("First 5 Row of Data:")
print(data.head())
print("###########################################################################")


# Missing values imputation
data.loc[data["class"]=="ckd","age"]=data['age'][data["class"]=="ckd"].fillna(data['age'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","age"]=data['age'][data["class"]=="notckd"].fillna(data['age'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","bp"]=data['bp'][data["class"]=="ckd"].fillna(data['bp'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","bp"]=data['bp'][data["class"]=="notckd"].fillna(data['bp'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","sg"]=data['sg'][data["class"]=="ckd"].fillna(data['sg'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","sg"]=data['sg'][data["class"]=="notckd"].fillna(data['sg'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","al"]=data['al'][data["class"]=="ckd"].fillna(data['al'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","al"]=data['al'][data["class"]=="notckd"].fillna(data['al'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","su"]=data['su'][data["class"]=="ckd"].fillna(data['su'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","su"]=data['su'][data["class"]=="notckd"].fillna(data['su'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","rbc"]=data['rbc'][data["class"]=="ckd"].fillna(data['rbc'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","rbc"]=data['rbc'][data["class"]=="notckd"].fillna(data['rbc'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","pc"]=data['pc'][data["class"]=="ckd"].fillna(data['pc'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","pc"]=data['pc'][data["class"]=="notckd"].fillna(data['pc'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","pcc"]=data['pcc'][data["class"]=="ckd"].fillna(data['pcc'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","pcc"]=data['pcc'][data["class"]=="notckd"].fillna(data['pcc'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","ba"]=data['ba'][data["class"]=="ckd"].fillna(data['ba'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","ba"]=data['ba'][data["class"]=="notckd"].fillna(data['ba'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","bgr"]=data['bgr'][data["class"]=="ckd"].fillna(data['bgr'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","bgr"]=data['bgr'][data["class"]=="notckd"].fillna(data['bgr'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","bu"]=data['bu'][data["class"]=="ckd"].fillna(data['bu'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","bu"]=data['bu'][data["class"]=="notckd"].fillna(data['bu'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","sc"]=data['sc'][data["class"]=="ckd"].fillna(data['sc'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","sc"]=data['sc'][data["class"]=="notckd"].fillna(data['sc'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","sod"]=data['sod'][data["class"]=="ckd"].fillna(data['sod'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","sod"]=data['sod'][data["class"]=="notckd"].fillna(data['sod'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","pot"]=data['pot'][data["class"]=="ckd"].fillna(data['pot'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","pot"]=data['pot'][data["class"]=="notckd"].fillna(data['pot'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","hemo"]=data['hemo'][data["class"]=="ckd"].fillna(data['hemo'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","hemo"]=data['hemo'][data["class"]=="notckd"].fillna(data['hemo'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","pcv"]=data['pcv'][data["class"]=="ckd"].fillna(data['pcv'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","pcv"]=data['pcv'][data["class"]=="notckd"].fillna(data['pcv'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","wbcc"]=data['wbcc'][data["class"]=="ckd"].fillna(data['wbcc'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","wbcc"]=data['wbcc'][data["class"]=="notckd"].fillna(data['wbcc'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","rbcc"]=data['rbcc'][data["class"]=="ckd"].fillna(data['rbcc'][data["class"]=="ckd"].mean())
data.loc[data["class"]=="notckd","rbcc"]=data['rbcc'][data["class"]=="notckd"].fillna(data['rbcc'][data["class"]=="notckd"].mean())

data.loc[data["class"]=="ckd","htn"]=data['htn'][data["class"]=="ckd"].fillna(data['htn'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","htn"]=data['htn'][data["class"]=="notckd"].fillna(data['htn'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","dm"]=data['dm'][data["class"]=="ckd"].fillna(data['dm'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","dm"]=data['dm'][data["class"]=="notckd"].fillna(data['dm'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","cad"]=data['cad'][data["class"]=="ckd"].fillna(data['cad'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","cad"]=data['cad'][data["class"]=="notckd"].fillna(data['cad'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","appet"]=data['appet'][data["class"]=="ckd"].fillna(data['appet'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","appet"]=data['appet'][data["class"]=="notckd"].fillna(data['appet'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","pe"]=data['pe'][data["class"]=="ckd"].fillna(data['pe'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","pe"]=data['pe'][data["class"]=="notckd"].fillna(data['pe'][data["class"]=="notckd"].mode()[0])

data.loc[data["class"]=="ckd","ane"]=data['ane'][data["class"]=="ckd"].fillna(data['ane'][data["class"]=="ckd"].mode()[0])
data.loc[data["class"]=="notckd","ane"]=data['ane'][data["class"]=="notckd"].fillna(data['ane'][data["class"]=="notckd"].mode()[0])

# Substituting non-numeric features
data.loc[data["rbc"]=="normal","rbc"]=1
data.loc[data["rbc"]=="abnormal","rbc"]=0

data.loc[data["pc"]=="normal","pc"]=1
data.loc[data["pc"]=="abnormal","pc"]=0

data.loc[data["pcc"]=="present","pcc"]=1
data.loc[data["pcc"]=="notpresent","pcc"]=0

data.loc[data["ba"]=="present","ba"]=1
data.loc[data["ba"]=="notpresent","ba"]=0

data.loc[data["htn"]=="yes","htn"]=1
data.loc[data["htn"]=="no","htn"]=0

data.loc[data["dm"]=="yes","dm"]=1
data.loc[data["dm"]=="no","dm"]=0

data.loc[data["cad"]=="yes","cad"]=1
data.loc[data["cad"]=="no","cad"]=0

data.loc[data["appet"]=="good","appet"]=1
data.loc[data["appet"]=="poor","appet"]=0

data.loc[data["pe"]=="yes","pe"]=1
data.loc[data["pe"]=="no","pe"]=0

data.loc[data["ane"]=="yes","ane"]=1
data.loc[data["ane"]=="no","ane"]=0

# Shuffling and spliting dataset to train and test
train,test=model_selection.train_test_split(data,test_size=0.3,shuffle=True, random_state=64)

# Separation of features and target
train_features= train.iloc[:,0:24].values
train_targets= train.loc[:,"class"].values

test_features= test.iloc[:,0:24].values
test_targets= test.loc[:,"class"].values

# Learning
forest= RandomForestClassifier(n_estimators=7, random_state=96)
trained_forest= forest.fit(train_features,train_targets)

# Results
print("Accuracy: "+ str(trained_forest.score(test_features,test_targets)*100) +"%")
print("###########################################################################")
prediction= trained_forest.predict(test_features)
print("Sensitivity: "+ str(metrics.recall_score(test_targets,prediction,pos_label="ckd")*100)+"%")
print("###########################################################################")
print("Specificity: "+ str(metrics.recall_score(test_targets,prediction,pos_label="notckd")*100)+"%")
print("###########################################################################")

# Visualization of features importance
plt.bar(data.keys()[0:24].values, trained_forest.feature_importances_)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()




