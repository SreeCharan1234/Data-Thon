# Data-Thon

# F1nalyze - Formula 1 Datathon 🚀
### team Name :- The_Winners 
## The Main Code :-
  data=pd.read_csv("/kaggle/input/f1nalyze-datathon-ieeecsmuj/train.csv")
Y=data['position']
data=data.drop(['fp1_date', 'fp1_time',
       'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date',
       'quali_time', 'sprint_date', 'sprint_time','position','time_x','timetaken_in_millisec','fastestLap','rank','fastestLapTime','max_speed','position_x','time_y','driver_num','driver_code'],axis=1)
data=data.select_dtypes(exclude=['object'])
data.info()
X=data
plt.figure(figsize=[20,15])
sns.heatmap(pd.concat([X,Y],axis=1).corr(),fmt="0.2f",vmax=1,vmin=-1,cmap="RdBu",annot=True)
plt.show
X.drop(['resultId','racerId','driverId','constructorId'],axis=1,inplace=True)
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=.20,random_state=1)
print("X_train: ",x_train.shape)
print("Y_train: ",y_train.shape)
print("X_test: ",x_test.shape)
print("Y_test: ",y_test.shape)
model1=RandomForestRegressor()
model1.fit(x_train,y_train)
predict=model1.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,predict))
print(rmse)
testdata=pd.read_csv("/kaggle/input/f1nalyze-datathon-ieeecsmuj/test.csv")
testdata=testdata.drop(['fp1_date', 'fp1_time',
       'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date',
       'quali_time', 'sprint_date', 'sprint_time','time_x','timetaken_in_millisec','fastestLap','rank','fastestLapTime','max_speed','position_x','time_y','driver_num','driver_code'],axis=1)
testdata=testdata.select_dtypes(exclude=['object'])
testdata.drop(['resultId','racerId','driverId','constructorId'],axis=1,inplace=True)
testdata.drop(['resultId','racerId','driverId','constructorId'],axis=1,inplace=True)
testdata.info()
final=model1.predict(testdata)
final=pd.DataFrame({"position":final,"result_driver_standing":testdata['result_driver_standing']})
final.head(20)
final.to_csv("output1.csv",index=False)
## 1.Loding dependency
![6](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/de5bcad1-e96a-42c7-9c7c-4fed19823465)

## 2.Looing the Data
![WhatsApp Image 2024-06-29 at 12 50 36 PM](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/6a03619a-5b24-402e-93a0-191283c52aec)
## 3. Droping the  unwanted data
![WhatsApp Image 2024-06-29 at 12 50 37 PM (2)](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/285f278a-8865-4140-b5d6-673aca783856)
## 4. Checking distribution of the data
![WhatsApp Image 2024-06-29 at 12 50 37 PM (1)](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/1ad1b07c-948f-4fab-81ed-168424b778f2)
## 5. Relation between the data using Heat Map
![2](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/77cc20d4-0231-4a23-802e-e56843b1144b)
## 6.Encoing the data
![WhatsApp Image 2024-06-29 at 12 50 38 PM](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/76d27915-91f7-4667-9299-7def6753a502)

## 7.Spliting the data into test and train

![5](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/d7046f6b-a785-4bf7-9b0a-73e90c4e2fb0)

## 8.Output of the test set 
![4](https://github.com/SreeCharan1234/Data-Thon/assets/119997965/279c4910-a862-47da-ab60-d83a61be8be2)
