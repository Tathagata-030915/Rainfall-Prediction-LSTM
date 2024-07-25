import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Data :
    
    def __init__(self) :
        self.df = pd.DataFrame([])

    def read(self, file_path : str) :
        """Reads values/data from CSV files into dataframe"""
        if os.path.exists(file_path) :
            self.df = pd.read_csv(file_path, skiprows = 1)
            print(f"Data successfully read from {file_path}.")
        else :
            print(f"Error: The file {file_path} does not exist.")

    def check_null(self) :
        """Checks for null values in dataframe"""
        print(self.df.shape)
        print("How many NaN are there in the dataset?\n", self.df.isna().sum())

    def clean_dataset(self) :
        """Cleans the dataset by removing null values"""

        if self.df is not None :
            original_shape = self.df.shape
            self.df = self.df.dropna()
            cleaned_shape = self.df.shape
            print(f"Dataset cleaned. Original shape: {original_shape}, Cleaned shape: {cleaned_shape}.")
        else :
            print("Error: No data to clean. Please read a dataset first.")

    
    def plot_train_points(self, col = 'NORMAL (mm)', Tp = 2894) :

        self.df.columns = self.df.columns.str.strip()

        plt.figure(figsize = (15, 4))
        if col == 'NORMAL (mm)':
            plt.title("Rainfall of first {} data points of NORMAL (mm) column".format(Tp), fontsize = 16)
            plt.plot(self.df['NORMAL (mm)'][:Tp], c = 'k', lw = 1)

        if col == 'ACTUAL (mm)':
            plt.title("Rainfall of first {} data points of ACTUAL (mm) column".format(Tp), fontsize = 16)
            plt.plot(self.df['ACTUAL (mm)'][:Tp], c = 'k', lw = 1)

        plt.grid(True)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()

    def print_head(self) :
        print("Head of Dataframe :-\n")
        print(self.df.head())

    def print_desc(self) :
        print("Description of Dataframe :-\n")
        print(self.df.describe())
