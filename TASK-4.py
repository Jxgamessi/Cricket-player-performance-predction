import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_dataset_path = "C:/Users/Lenovo/Desktop/Internsavy/cricket_stats_dataset.csv"
stats_dataset = pd.read_csv(train_dataset_path)


X = stats_dataset.drop(['Runs', 'Player'], axis=1)  
y = stats_dataset['Runs']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')


test_dataset_path = "C:/Users/Lenovo/Desktop/Internsavy/test.csv"
test_dataset = pd.read_csv(test_dataset_path)


test_features = test_dataset.drop(['Player', 'Runs'], axis=1)[X.columns]


predictions = rf_model.predict(test_features)


test_dataset['Predicted_Runs'] = predictions
print(test_dataset[['Player', 'Predicted_Runs']])
