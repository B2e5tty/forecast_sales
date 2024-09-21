# class for functions used in EDA analysis
import logging.handlers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging



class EDA:
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

        try:
            self.info_log.info("Loading file") 
            self.df = pd.read_csv(path)

        except:
            self.error_log.error("Error occurred when loading the files")

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
        self.info_log.info('Return the dataframe')
        return self.df

    # fill null values by zero
    def fill_null(self,col):
        self.info_log.info('Replace NaN by zero')
        self.df[col].fillna(0, inplace=True)

    # remove null values
    def remove_null(self):
        self.info_log.info('Remove null values')
        self.df.dropna(inplace=True)

    # change to appropriate data type
    def change_dataType(self):
        self.info_log.info('Change datatype of columns')
        columns = self.df.columns.tolist()

        for col in columns:
            if col in ('CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'):
                self.df[col] = self.df[col].astype('int64')

            elif col in ('Promo','Promo2','Open','StateHoliday','SchoolHoliday'):
                self.df[col] = self.df[col].astype('category')

            elif col == 'Date':
                self.df[col] = pd.to_datetime(self.df[col])

            elif col == 'Store':
                self.df[col] = self.df[col].astype('str')


    # box plot to check distribut
    def overview(self):
        self.info_log.info('Information about the dataset')
        print("Shape of the dataframe:")
        print(f"{self.df.shape}\n")

        print("Information on the data:")
        print(f"{self.df.info()}\n")

        print("Describe the numerical column statistics:")
        print(f"{self.df.describe()}\n")

        print("The first five rows of the data:")
        print(f"{self.df.head(3)}\n")

            
            



