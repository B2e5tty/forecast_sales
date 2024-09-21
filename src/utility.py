
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
        self.info_log.info('Sales and Customers relation')
        new_df = self.df[['Sales','Customers']]
        print(new_df.corr())

        # scatter plot
        plt.scatter(x=new_df['Customers'], y=new_df['Sales'])
        plt.title("Correlation plot of sales and customers")
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.show()
        

    # promo effect on sales
    def promo_sale_behavior(self):
        self.info_log.info('Promo effect on sales')
        # sales
        print('Sales during promo and non-promo days')
        print(self.df.groupby('Promo')['Sales'].sum())

        # number of customers
        print('Number of Customer during promo and non-promo days')
        print(self.df.groupby('Promo')['Customers'].sum())

        # existing customers on promo and non-promo days
        print('Stores with promo existing and new customers')
        stores_with_promo = self.df[self.df['Promo'] == 1]['Store'].unique()

        customers_during_nonpromo = self.df[(self.df['Store'].isin(stores_with_promo)) & (self.df['Promo'] == 0)]['Customers'].sum()
        print(f'Number of existing customers: {customers_during_nonpromo}')

        customers_during_promo = self.df[(self.df['Store'].isin(stores_with_promo)) & (self.df['Promo'] == 1)]['Customers'].sum()
        print(f'Number of customers on promotion day: {customers_during_promo}')

        # new customers
        print(f'Number of new customers: {customers_during_promo - customers_during_nonpromo}')

    # add day name 
    def day_name(self):
        self.df['dayName'] = np.where(self.df['DayOfWeek'] == 1, 'Monday',
                                      np.where(self.df['DayOfWeek'] == 2, 'Tuesday',
                                      np.where(self.df['DayOfWeek'] == 3, 'Wednesday',
                                      np.where(self.df['DayOfWeek'] == 4, 'Thursday',
                                      np.where(self.df['DayOfWeek'] == 5, 'Friday',
                                      np.where(self.df['DayOfWeek'] == 6, 'Saturday',
                                      np.where(self.df['DayOfWeek'] == 7, 'Sunday','Not a day')))))))
    # day of promos 
    def promo_days_behavior(self):
        self.info_log.info('Days of the week most promos happen')
        new_df = self.df[self.df['Promo'] == 1]
        new_df = new_df.groupby('dayName').agg(
            number_of_promo = ('Promo', 'count'),
            sales = ('Sales', 'mean'),
            customers = ('Customers', 'mean')
        ).reset_index()

        print(new_df)

    # store type promo behavior(self)
    def store_promo_deploy(self):
        self.info_log.info('Which store type should promo be deployed')
        new_df = self.df[self.df['Promo'] == 1]
        new_df = new_df.groupby('StoreType').agg(
            number_of_promo = ('Promo', 'count'),
            sales = ('Sales', 'mean'),
            customers = ('Customers', 'mean')
        ).reset_index()

        new_df['sale_per_customer'] = new_df['sales'] / new_df['customers']

        print(new_df)

    # customers trend on opening and closing time
    def customer_time_behaviour(self):
        self.info_log.info('Customers average purchases on opening time')
        new_df = self.df[self.df['Open'] == 1]
        new_df = new_df.groupby('dayName').agg(
            sales = ('Sales', 'mean'),
            customers = ('Customers', 'mean')
        ).reset_index()

        print(new_df)

        # bar plot
        sns.barplot(data=new_df, x='dayName', y='sales')
        plt.title('Customer opening time trends')
        plt.xlabel('Day')
        plt.ylabel('Average sale')
        plt.show()

    # assortment effect on sale
    def assortment_on_sale(self):
        self.info_log.info('Assortments vs Sales')
        new_df = self.df.groupby('Assortment')['Sales'].mean().reset_index()
        new_df['Assortment'] = np.where(new_df['Assortment'] == 'a', 'Basic',
                                        np.where(new_df['Assortment'] == 'b', 'Extra',
                                        np.where(new_df['Assortment'] == 'c', 'Extended','None')))

        print(new_df)

        # bar plot
        sns.barplot(data=new_df, x='Assortment', y='Sales')
        plt.title('Assortment effect on sales')
        plt.xlabel('Assortment')
        plt.ylabel('Average sale')
        plt.show()

    # weekdays and weekends behaviour
    def weekdays_weekends_behaviour(self):
        self.info_log.info('Day of the week vs Sales')
        stores = self.df[self.df['DayOfWeek'] == 1]['Store'].unique().tolist()
 
        for i in range(2,6,1):
            nextDay = self.df[self.df['DayOfWeek'] == i]['Store'].unique().tolist()
            stores = list(set(stores).intersection(set(nextDay)))

        if len(stores) == self.df['Store'].nunique():
            print('All stores are open weekdays')

            weekdays_sales = self.df[(self.df['DayOfWeek'] >= 1) & (self.df['DayOfWeek'] < 6)]['Sales'].mean()
            weekdays_customers = self.df[(self.df['DayOfWeek'] >= 1) & (self.df['DayOfWeek'] < 6)]['Customers'].mean()

            weekends_sales = self.df[(self.df['DayOfWeek'] >= 6) & (self.df['DayOfWeek'] < 8)]['Sales'].mean()
            weekends_customers = self.df[(self.df['DayOfWeek'] >= 6) & (self.df['DayOfWeek'] < 8)]['Customers'].mean()

            print(f'Weedays average sales: {weekdays_sales}')
            print(f'Weedays average number of customers: {weekdays_customers}')
            print(f'Weekends average sales: {weekends_sales}')
            print(f'Weekends average number of customers: {weekends_customers}')

        else:
            print(f'Only {len(stores)} are open weekdays')

            weekdays_sales = self.df[(self.df['DayOfWeek'] >= 1) & (self.df['DayOfWeek'] < 6) & (self.df['Store'].isin(stores))]['Sales'].mean()
            weekdays_customers = self.df[(self.df['DayOfWeek'] >= 1) & (self.df['DayOfWeek'] < 6) & (self.df['Store'].isin(stores))]['Customers'].mean()

            weekends_sales = self.df[(self.df['DayOfWeek'] >= 6) & (self.df['DayOfWeek'] < 8) & (self.df['Store'].isin(stores))]['Sales'].mean()
            weekends_customers = self.df[(self.df['DayOfWeek'] >= 6) & (self.df['DayOfWeek'] < 8) & (self.df['Store'].isin(stores))]['Customers'].mean()

            print(f'Weedays average sales: {weekdays_sales}')
            print(f'Weedays average number of customers: {weekdays_customers}')
            print(f'Weekends average sales: {weekends_sales}')
            print(f'Weedays average number of customers: {weekends_customers}')

    
    # competitors distance effect
    def competitor_distance_sale_behavior(self):
        self.info_log.info('Competitor distance vs Sales')
        new_df = self.df[['Store','CompetitionDistance']]
        new_df.drop_duplicates(inplace=True)
        mean_distance = new_df['CompetitionDistance'].mean()

        # sales
        sales_greater = self.df[self.df['CompetitionDistance'] > mean_distance]['Sales'].mean()
        sales_less = self.df[self.df['CompetitionDistance'] <= mean_distance]['Sales'].mean()

        print(f"Stores with greater than average competitor distance has average sales: {sales_greater}")
        print(f"Stores with less than average competitor distance has average sales: {sales_less}")
        print('\n')

        # number of customers
        cust_greater = self.df[self.df['CompetitionDistance'] > mean_distance]['Customers'].mean()
        cust_less = self.df[self.df['CompetitionDistance'] <= mean_distance]['Customers'].mean()

        print(f"Stores with greater than average competitor distance has average number of customers: {cust_greater}")
        print(f"Stores with less than average competitor distance has average number of customers: {cust_less}")
        print('\n')

        # promo
        df_promo_greater = self.df[self.df['CompetitionDistance'] > mean_distance]
        promo_sale_greater = df_promo_greater[df_promo_greater['Promo'] == 1]['Sales'].sum()
        df_promo_less = self.df[self.df['CompetitionDistance'] <= mean_distance]
        promo_sale_less = df_promo_less[df_promo_less['Promo'] == 1]['Sales'].sum()
        print('\n')

        print(f"Stores with greater than average competitor distance has promotion sales: {promo_sale_greater}")
        print(f"Stores with less than average competitor distance has promotion sales: {promo_sale_less}")


        plt.scatter(self.df['CompetitionDistance'],self.df['Sales'])
        plt.title("The relation between competitor distance and sales")
        plt.xlabel('Competitor Distance')
        plt.ylabel('Sales')
        plt.show()

        