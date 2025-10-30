import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
def create_model(data):
    x=data.drop(columns=["diagnosis"])
    y=data["diagnosis"]

    ##scale the data
    scaler=StandardScaler()
    x=scaler.fit_transform(x)

    ##split the data
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    ##train the model
    model=LogisticRegression()
    model.fit(x_train,y_train)
  

    ##test the model
    y_pred=model.predict(x_test)
    print("accuracy of our model:",accuracy_score(y_test,y_pred))
    print("classification of our model:",classification_report(y_test,y_pred))

    return model,scaler


def get_clean_data_():
    data=pd.read_csv(r"D:\Breast_Cancer_Diagnostic_app\breast_cancer_diagnostic\data\data.csv")
    data=data.drop(columns=["Unnamed: 32","id"])
    data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})
    return data


def main():
    data=get_clean_data_()
    # print(data.head())
    # print(data.info())
    model,scaler=create_model(data)
    with open("model/model.pkl","wb") as f:
        pickle.dump(model,f)
    with open("model/scaler.pkl","wb") as f:
        pickle.dump(scaler,f)
     
if __name__=="__main__":
    main()