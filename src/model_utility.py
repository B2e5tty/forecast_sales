# class for functions used in building model
import logging.handlers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

class PREDICTION:
    def __init__(self, path):
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

            self.df = pd.read_csv(path)

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
        return self.df

    # change to appropriate data type
    def change_dataType(self, dataset='train'):
        self.info_log.info('Change datatype of columns')
        self.df.drop('Unnamed: 0',axis=1, inplace = True)

        self.df['Date'] = pd.to_datetime(self.df['Date'])

        columns = self.df.columns.tolist()

        for col in columns:
            if col in ('StoreType','PromoInterval','StateHoliday'):
                encoder = OneHotEncoder(drop='first', sparse=False)
                reshaped = self.df[col].values.reshape(-1, 1)


                encoded = encoder.fit_transform(reshaped)
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=self.df.index)
                self.df = pd.concat([self.df.drop(col, axis=1), encoded_df], axis=1)

            elif col == 'Assortment':
                label = LabelEncoder() 
                self.df[col] = label.fit_transform(self.df[col])

            else: pass


    def new_features(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        # weekdays/ weekends
        def weekdays_weekends(day_num):
            if day_num in [1,2,3,4,5]:
                return 1
            else:return 0

        self.df['weekdays'] = self.df['DayOfWeek'].apply(lambda x: weekdays_weekends(x))

        # next holiday days
        all_holidays = self.df[self.df['StateHoliday'] != '0']['Date'].unique()

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
            
        self.df['nextHolidayDays'] = self.df['Date'].apply(lambda x: next_holidays(x,all_holidays))
        self.df['pastHolidayDays'] = self.df['Date'].apply(lambda x: passed_holidays(x,all_holidays))
            

        # begining, mid and end of month
        def month_range(month_day):
            if month_day >= 1 and month_day < 11:
                return 1
            elif month_day >= 11 and month_day < 21:
                return 2
            else:
                return 3

        self.df['DayOfMonth'] = self.df['Date'].dt.day
        self.df['monthRange'] = self.df['DayOfMonth'].apply(lambda x: month_range(x))
        self.df.drop('DayOfMonth',axis = 1, inplace = True)

        # begining, mid and end of year
        def year_range(month_of_year):
            if month_of_year >= 1 and month_of_year < 11:
                return 1
            elif month_of_year >= 11 and month_of_year < 21:
                return 2
            else:
                return 3

        self.df['MonthOfYear'] = self.df['Date'].dt.month
        self.df['yearRange'] = self.df['MonthOfYear'].apply(lambda x: year_range(x))
        self.df.drop('MonthOfYear',axis = 1, inplace = True)

        # competitative area or not
        average_distance = self.df['CompetitionDistance'].mean()
        self.df['competitive'] = np.where(self.df['CompetitionDistance'] > average_distance, 1, 0)

        self.df.fillna(0, axis=1, inplace = True)

        