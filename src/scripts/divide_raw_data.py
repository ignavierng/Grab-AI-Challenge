"""
Divide csv training data of 4m entries into two separate files
(Because GitHub doesn't allow uploading file with size >100MB)
"""
import pandas as pd


if __name__ == '__main__':
    data_df = pd.read_csv('data_loader/data/training.csv')
    upper_half_df = data_df.iloc[: len(data_df)//2]
    lower_half_df = data_df.iloc[len(data_df)//2 : ]

    print('len(data_df): {}'.format(len(data_df)))
    print('len(upper_half_df): {}'.format(len(upper_half_df)))
    print('len(lower_half_df): {}'.format(len(lower_half_df)))

    upper_half_df.to_csv('data_loader/data/upper_half_data.csv')
    lower_half_df.to_csv('data_loader/data/lower_half_data.csv')
