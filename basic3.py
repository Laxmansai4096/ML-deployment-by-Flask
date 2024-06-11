import pickle 
import pandas as pd 

df=pd.read_csv('heart.csv')

X=df.iloc[:, :-1]
y=df.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Initializing and training the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
sv=rf_model.fit(X_train, Y_train)

pickle.dump(sv,open('heart.pkl','wb'))

