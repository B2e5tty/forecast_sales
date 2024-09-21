
# class for functions used in EDA analysis
import logging.handlers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import chi2_contingency, ttest_ind, f_oneway


class ANALYSIS:
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

    # return the dataframe
    def get_dataframe(self):
        return self.df, self.df2

    # change to appropriate data type
    def change_dataType(self, dataset='train'):
        self.info_log.info('Change datatype of columns')
        columns = self.df.columns.tolist()

        data_df = self.df if dataset == 'train' else self.df2

        for col in columns:
            if col in ('CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'):
                data_df[col] = data_df[col].astype('int64')

            elif col in ('Promo','Promo2','Open','StateHoliday','SchoolHoliday'):
                data_df[col] = data_df[col].astype('category')

            elif col == 'Date':
                data_df[col] = pd.to_datetime(data_df[col])

            elif col == 'Store':
                data_df[col] = data_df[col].astype('str')

    # set date as index
    def date_index(self):
        self.df.set_index('Date', inplace=True)
        self.df.sort_index(ascending=True, inplace=False)

        self.df2.set_index('Date', inplace=True)
        self.df2.sort_index(ascending=True, inplace=False)

    # distribution similarity
    def dist_similar(self):
        self.info_log.info('Distribution of the categorical features')
        columns = ['Promo','Promo2','Open','SchoolHoliday','StateHoliday']

        for col in columns:
            contingency_table = pd.crosstab(self.df[col], self.df2[col], dropna=False)
            _, p_val, _, _ = chi2_contingency(contingency_table)
            print(f"{col} distribution in train and test set has P-Value of {p_val}")

    # distribution similarity for numerical features
    def dist_similar(self):
        self.info_log.info('Distribution of the categorical features')
        columns = ['Promo','Promo2','Open','SchoolHoliday','StateHoliday']

        for col in columns:
            contingency_table = pd.crosstab(self.df[col], self.df2[col], dropna=False)
            _, p_val, _, _ = chi2_contingency(contingency_table)
            print(f"{col} distribution in train and test set has P-Value of {p_val}")

        # boxplot
    def box_plot(self, dataset='train'):
        self.info_log.info('Box plots of the categorical features')
        columns = ['CompetitionDistance','Sales','Customers']
        fig, axs = plt.subplots(3,1, figsize=(15,8))
        axs = axs.flatten()

        data_df = self.df if dataset == 'train' else self.df2

        for i,col in enumerate(columns):
            # sns.countplot(ax=axs[i],data = data_df, x = col)
            axs[i].hist(data_df[col])
            axs[i].set_title('Distribution of ' + col + ' ' + dataset)
            axs[i].set_xlabel(col)
            axs[i].set_ylabel('Count')

        plt.tight_layout()
        plt.show()


    # barplot
    def bar_plot(self, dataset='train'):
        self.info_log.info('Bar plots of the categorical features')
        columns = ['Promo','Promo2','Open','SchoolHoliday','StateHoliday']
        fig, axs = plt.subplots(3,2, figsize=(15,8))
        axs = axs.flatten()

        data_df = self.df if dataset == 'train' else self.df2

        for i,col in enumerate(columns):
            sns.countplot(ax=axs[i],data = data_df, x = col)
            axs[i].set_title('Distribution of ' + col + ' ' + dataset)
            axs[i].set_xlabel(col)
            axs[i].set_ylabel('Count')

        plt.tight_layout()
        plt.show()

    # sales behavior respect to holidays
    def sale_behavior(self):
        self.info_log.info('Sales behavior during, before and after holidays')
        self.df = self.df.sort_values(by='Date', ascending=True)

        self.df['duringHoliday'] = np.where(self.df['StateHoliday'] != '0', 'during', 'not')
        self.df['before'] = self.df['duringHoliday'].shift(-7)
        self.df['after'] = self.df['duringHoliday'].shift(7)

        self.df['HolidayPeriod'] = np.where((self.df['StateHoliday'] != '0') & (self.df['duringHoliday'] == 'during'), 'During',
                            np.where((self.df['StateHoliday'] == '0') & (self.df['before'] == 'during'), 'Before',
                            np.where((self.df['StateHoliday'] == '0') & (self.df['after'] == 'during'), 'After', 'Non-Holiday')))
        
        self.df.drop(['duringHoliday','before','after'], axis=1, inplace=True)

        sale_holiday_behavior = self.df.groupby('HolidayPeriod')['Sales'].mean().reset_index()
        print(sale_holiday_behavior)
        
        # Perform one-way ANOVA to check if there's a significant difference
        sales_before = self.df[self.df['HolidayPeriod'] == 'Before']['Sales']
        # sales_during = self.df[self.df['HolidayPeriod'] == 'During']['Sales']   # has small value compared to the others
        sales_after = self.df[self.df['HolidayPeriod'] == 'After']['Sales']

        f_stat, p_value = f_oneway(sales_before,sales_after)
        print(f"ANOVA P-Value: {p_value}")

        # Bar plot to show mean sales
        sale_holiday_behavior.plot(kind='bar', x='HolidayPeriod', y='Sales', title='Mean Sales by Holiday Period')
        plt.show()

    # sales behavior respect to holidays
    def sale_seasonal_behavior(self):
        self.info_log.info('Sales on specific holiday seasons')
        sale_seasonal_behavior = self.df.groupby('StateHoliday')['Sales'].mean().reset_index()
        print(sale_seasonal_behavior)

        # Bar plot to show mean sales
        sale_seasonal_behavior.plot(kind='bar', x='StateHoliday', y='Sales', title='Mean Sales by Holiday Period')
        plt.show()

    # correlation
    def sale_customer_corr(self):
        new_df = self.df[['Sales','Customers']]
        print(new_df.corr())

        # scatter plot
        plt.scatter(x=new_df['Customers'], y=new_df['Sales'])
        plt.title("Correlation plot of sales and customers")
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.show()
        