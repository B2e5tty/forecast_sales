# class for functions used in building model
import logging.handlers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from flask import Flask, request, jsonify


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



class PREDICTION:
    def __init__(self, dataframe1:pd.DataFrame, dataframe2:pd.DataFrame):
            self.info_log = logging.getLogger('info')
            self.info_log.setLevel(logging.INFO)

            info_handler = logging.FileHandler('info.log')
            info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            info_handler.setFormatter(info_formatter)
            self.info_log.addHandler(info_handler)


            self.error_log = logging.getLogger('errors')
            self.error_log.setLevel(logging.ERROR)

            error_handler = logging.FileHandler('errors.log')
            error_formatter = logging.Formatter('%(asctime)s - %(name)s -  %(levelname)s - %(message)s')

            error_handler.setFormatter(error_formatter)
            self.error_log.addHandler(error_handler)

            self.df = dataframe1
            self.df2 = dataframe2

    # close log process
    def close_logs(self):
        # Close and remove the info log handler
        handlers = self.info_log.handlers[:]
        for handler in handlers:
            handler.close()
            self.info_log.removeHandler(handler)

        # Close and remove the error log handler
        handlers = self.error_log.handlers[:]
        for handler in handlers:
            handler.close()
            self.error_log.removeHandler(handler)

    # return the dataframe
    def get_dataframe(self):
        return self.df, self.df2

    # change to appropriate data type
    def change_dataType(self, dataset='train'):
        self.info_log.info('Change datatype of columns')

        data_df = self.df if dataset == 'train' else self.df2

        data_df.drop('Unnamed: 0',axis=1, inplace = True)

        data_df['Date'] = pd.to_datetime(data_df['Date'])

        columns = data_df.columns.tolist()

        for col in columns:
            if col in ('StoreType','PromoInterval','StateHoliday'):
                encoder = OneHotEncoder(drop='first', sparse=False)
                reshaped = data_df[col].values.reshape(-1, 1)
                encoded = encoder.fit_transform(reshaped)
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=data_df.index)
                data_df = pd.concat([data_df.drop(col, axis=1), encoded_df], axis=1)
            

            elif col == 'Assortment':
                label = LabelEncoder() 
                data_df[col] = label.fit_transform(data_df[col])

            else: pass

        if dataset == 'train':
            self.df = data_df

        else: self.df2 = data_df


        print(self.df.info())


        
        # print(data_df)


    def new_features(self, dataset):
        self.info_log.info('Adding new features')

        data_df = self.df if dataset == 'train' else self.df2
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        # weekdays/ weekends
        def weekdays_weekends(day_num):
            if day_num in [1,2,3,4,5]:
                return 1
            else:return 0

        data_df['weekdays'] = data_df['DayOfWeek'].apply(lambda x: weekdays_weekends(x))

        # next holiday days
        all_holidays = data_df[data_df['StateHoliday'] != '0']['Date'].unique()

        def next_holidays(row_dates, holidays):
            futureHolidays = holidays[holidays > row_dates]

            if len(futureHolidays) > 0:
                return (futureHolidays.min() - row_dates).days
            
            else:
                return 0
            
        def passed_holidays(row_dates, holidays):
            passedHolidays = holidays[holidays < row_dates]

            if len(passedHolidays) > 0:
                return (row_dates - passedHolidays.max()).days
            
            else:
                return 0
            
        data_df['nextHolidayDays'] = data_df['Date'].apply(lambda x: next_holidays(x,all_holidays))
        data_df['pastHolidayDays'] = data_df['Date'].apply(lambda x: passed_holidays(x,all_holidays))
            

        # begining, mid and end of month
        def month_range(month_day):
            if month_day >= 1 and month_day < 11:
                return 1
            elif month_day >= 11 and month_day < 21:
                return 2
            else:
                return 3

        data_df['DayOfMonth'] = data_df['Date'].dt.day
        data_df['monthRange'] = data_df['DayOfMonth'].apply(lambda x: month_range(x))
        data_df.drop('DayOfMonth',axis = 1, inplace = True)

        # begining, mid and end of year
        def year_range(month_of_year):
            if month_of_year >= 4:
                return 1
            elif month_of_year >= 8:
                return 2
            else:
                return 3

        data_df['MonthOfYear'] = data_df['Date'].dt.month
        data_df['yearRange'] = data_df['MonthOfYear'].apply(lambda x: year_range(x))
        data_df.drop('MonthOfYear',axis = 1, inplace = True)

        # competitative area or not
        average_distance = data_df['CompetitionDistance'].mean()
        data_df['competitive'] = np.where(data_df['CompetitionDistance'] > average_distance, 1, 0)

        data_df.fillna(0, axis=1, inplace = True)



    # predictive models
    def regressor_modle(self):
        self.info_log.info('Machine learning model')
        # train sets
        x_train = self.df.drop(['Sales','Date'],axis=1)
        y_train = self.df['Sales']


        # split for validation
        x_train_new,x_test_new,y_train_new,y_test_new = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        # test sets from test dataset
        x_test = self.df2.drop('Date',axis=1)

        # scaling
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_new)
        x_test_scaled = scaler.fit(x_test_new)


        # sampling
        underSampler = RandomUnderSampler(random_state = 42)
        x_train_resampled, y_train_resampled = underSampler.fit_resample(x_train_scaled, y_train_new)

        # random forest
        random_para = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 10],
            'regressor__min_samples_leaf': [1, 5],
            'regressor__max_features': ['sqrt', 'log2']
        }

        # sklearn pipeline
        random_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state = 42))
        ])

        # decision tree
        decision_para = {
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 10, 20],
        'regressor__min_samples_leaf': [1, 5, 10],
        'regressor__criterion': ['squared_error']
        }

        decision_pipline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ])

        # random model
        random_model = GridSearchCV(estimator=random_pipeline, param_grid=random_para,cv=3,n_jobs=-1, verbose=2)
        # decision tree model
        decision_model = GridSearchCV(estimator=decision_pipline, param_grid=decision_para,cv=5,n_jobs=-1, verbose=2)
        
        # fit the model
        random_model.fit(x_train_resampled,y_train_resampled)
        decision_model.fit(x_train_resampled,y_train_resampled)

        # important features
        best_random_model = random_model.best_estimator_
        best_decision_model = decision_model.best_estimator_

        feature_importance = best_random_model.named_steps['regressor'].feature_importances_

        feature_importance_df = pd.DataFrame({
            'features': x_train.columns,
            'importance_score': feature_importance
        }).sort_values(by='importance_score', ascending = False)

        # top five features
        top_features = feature_importance_df['features'].head(5).tolist()

        x_train_main = x_train_resampled[top_features]
        x_test_main = x_test_scaled[top_features]

        # remodel random forest
        best_random_model .fit(x_train_main,y_train_resampled)
        y_hat = best_random_model.predict(x_test_main)

        mse = mean_squared_error(y_test_new, y_hat)
        r2 = r2_score(y_test_new, y_hat)

        print(f"Random forest Mean Squared Error: {mse}")
        print(f"Random forest R^2 Score: {r2}")

        # prediction
        y_pred = best_random_model.predict(x_test)
        pd.DataFrame({'predictions': y_pred}).to_csv('ranodm_test_prediction.csv',index=False)

        # remodel decision tree
        best_decision_model.fit(x_train_main,y_train_resampled)
        y_hat = best_decision_model.predict(x_test_main)
        mse = mean_squared_error(y_test_new, y_hat)
        r2 = r2_score(y_test_new, y_hat)

        print(f"Decision tree Mean Squared Error: {mse}")
        print(f"Decision treee R^2 Score: {r2}")

        # prediction
        y_pred = best_decision_model.predict(x_test)
        pd.DataFrame({'predictions': y_pred}).to_csv('decision_test_prediction.csv',index=False)

        # serialize the model
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        filename1 = f'Random_model_{timestamp}.pkl'
        filename2 = f'Decision_model_{timestamp}.pkl'

        joblib.dump(best_random_model, filename1)
        joblib.dump(best_decision_model, filename2)
        print(f'The models are saved.')

    # deep learing model
    def deep_model(self):
        self.info_log.info('Deep learning model')
        self.df.set_index('Date', inplace=True)

        sales = self.df[['Sales']]

        # check for stationary
        result = adfuller(self.df['Sales'])
        print(f'Adf Statistic: {result[0]}')
        print(f'p-value {result[1]}')

        if result[1] > 0.05:
            sales_diff = sales_diff.diff().dropna()

        else: sales_diff = sales

        print(sales_diff)

        # plot autocorrelation and partial autocorrelation
        plot_acf(sales_diff)
        plot_pacf(sales_diff)
        plt.show()

        # scaling
        scaler = MinMaxScaler()
        sales_scaled = scaler.fit_transform(sales_diff)
                
        X, y = [], []

        # creating supervised window sliding
        windowSize = 30
        for i in range(windowSize, len(sales_scaled)):
            X.append(self.df[i - windowSize:i, 0])
            y.append(self.df[i,0])

        X, y = np.array(X), np.array(y)

        # reshape
        X = X.reshape((X.shape[0], X.shape[1],1))

        x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,shuffle=False)

        # build LSM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(30,1)),
            Dense(1)
        ])

        model.compile(optimizer = 'adam', loss = 'mse')

        # training
        model.fit(x_train,y_train, epochs = 10, batch_size = 32)
        y_hat = model.predict(x_test)

        y_hat_inverse = scaler.inverse_transform(y_hat)

        print(f"Mean Squared Error of : {mean_squared_error(y_test,y_hat_inverse)}")



    # API building
    def api_building(self):
             # Initialize Flask app
            app = Flask(__name__)

            # Load the serialized model
            model_path = 'models/model_xxx.pkl'  # Replace with the path to your model file
            model = joblib.load(model_path)

            # Define the prediction endpoint
            @app.route('/predict', methods=['POST'])
            def predict():
                # Get input data from request (assuming JSON format)
                data = request.get_json()
                
                # Extract feature values from the request JSON
                # Example input: {"features": [value1, value2, value3,...]}
                features = data.get('features')
                
                # Convert the input data to a NumPy array for the model
                input_data = np.array([features])
                
                # Make predictions
                prediction = model.predict(input_data)
                
                # Prepare the response
                response = {
                    'prediction': prediction[0],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return jsonify(response)

            # Define an endpoint to check the API status
            @app.route('/status', methods=['GET'])
            def status():
                return jsonify({"status": "API is running", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            # Run the API
            if __name__ == '__main__':
                app.run(host='0.0.0.0', port=5000)


        


