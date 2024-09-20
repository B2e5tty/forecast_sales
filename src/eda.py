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

    def get_dataframe(self):
        return self.df


