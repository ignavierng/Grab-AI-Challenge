import random
import numpy as np
import pandas as pd
import itertools
import collections

from helpers.data_utils import *    # TODO: Eliminate this bad case practice


class Dataset(object):
    """
    Dataset class for the purposes of:
    (1) Instantiated as an object during training phase to provide train, val and test sets
        (class attributes: X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np)
    (2) Use staticmethod `generate_features_and_labels_for_inference` to generate features and labels
        from raw csv data during inference phase
    """

    def __init__(self, data_path, geohash_path, num_steps, batch_size, sliding_features, use_geohash,
                 use_day, use_cyclical_timestamp, use_part_of_day, val_ratio, test_ratio):
        # Load raw and geohash data
        data_df = load_raw_data(data_path)
        geohash_df = pd.read_csv(geohash_path, index_col=0)

        # Assign class data members
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.sliding_features = sliding_features
        self.use_geohash = use_geohash
        self.use_day = use_day
        self.use_cyclical_timestamp = use_cyclical_timestamp
        self.use_part_of_day = use_part_of_day
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Rearrange columns and merge geohash index into data_df
        data_df = data_df[['day', 'timestamp', 'geohash6', 'demand']]

        # Get train, val and test set
        self.X_train_np, self.y_train_np, self.X_val_np, self.y_val_np, self.X_test_np, self.y_test_np \
            = self._setup(data_df, geohash_df)

    def _setup(self, data_df, geohash_df):
        """
        Setup train, val and test sets for training purpose
        """
        # X_np and y_np returned here are batched based on timestamp
        X_np, y_np = Dataset.get_features_and_labels_batched_by_timestamp(data_df, geohash_df, self.num_steps, self.sliding_features,
                                                                          self.use_geohash, self.use_day, self.use_cyclical_timestamp,
                                                                          self.use_part_of_day)

        # Useful variables
        num_samples = X_np.shape[0]
        num_train = int(num_samples * (1.0 - self.val_ratio - self.test_ratio))
        num_val = int(num_samples * self.val_ratio)
#         num_test = num_samples - num_train - num_val

        # Split into train, val and test sets
        X_train_np, y_train_np = X_np[:num_train], y_np[:num_train]
        X_val_np, y_val_np = X_np[num_train : num_train + num_val], y_np[num_train : num_train + num_val]
        X_test_np, y_test_np = X_np[num_train + num_val:], y_np[num_train + num_val:]

        # Reshape train, val and test sets to tailor for different models
        X_train_np, y_train_np = Dataset.reshape_features_labels(X_train_np, y_train_np, self.batch_size)
        X_val_np, y_val_np = Dataset.reshape_features_labels(X_val_np, y_val_np, self.batch_size)
        X_test_np, y_test_np = Dataset.reshape_features_labels(X_test_np, y_test_np, self.batch_size)

        # Shuffle train set on the first dimension (to improve generalization)
        idx_ls = list(range(X_train_np.shape[0]))
        random.shuffle(idx_ls)
        X_train_np = X_train_np[idx_ls]
        y_train_np = y_train_np[idx_ls]

        return X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np

    @staticmethod
    def reshape_features_labels(X_np, y_np, batch_size):
        """
        Reshape X_np and y_np to tailor for different models
        The final shapes are as follows:
        - LightGBM: X_np (num_train, num_features), y_np (num_train, 5)
        - MLP: X_np (num_batch, batch_size, num_features), y_np (num_batch, batch_size, 5)
        - LSTM/TCN/SpatioTCN: X_np (num_batch, batch_size, num_steps, num_features), y_np (num_batch, batch_size, 5)
        """
        if batch_size is None:
            # For MultiLightgbm
            return X_np.reshape(-1, X_np.shape[-1]), y_np.reshape(-1, y_np.shape[-1])
        else:
            return X_np, y_np

    @staticmethod
    def generate_features_and_labels_for_inference(data_df, geohash_df, num_steps, batch_size, sliding_features,
                                                   use_geohash, use_day, use_cyclical_timestamp, use_part_of_day):
        """
        Pass in raw dataframe and parameters, return features and labels in numpy arrays with desired shapes
        """
        X_np, y_np = Dataset.get_features_and_labels_batched_by_timestamp(data_df, geohash_df, num_steps, sliding_features,
                                                                          use_geohash, use_day, use_cyclical_timestamp, use_part_of_day)
        return Dataset.reshape_features_labels(X_np, y_np, batch_size)

    @staticmethod
    def get_features_and_labels_batched_by_timestamp(data_df, geohash_df, num_steps, sliding_features,
                                                     use_geohash, use_day, use_cyclical_timestamp, use_part_of_day):
        """
        Preprocess raw data to obtain features and labels batched according to timestamp
        Remark: Do not split into train, val and test sets here
        """
        day_ls = data_df['day'].drop_duplicates().tolist()
        timestamp_ls = data_df['timestamp'].drop_duplicates().tolist()
        geohash_ls = geohash_df['geohash6'].drop_duplicates().tolist()

        # Fill in entries with zero demand
        filled_data_df = pd.DataFrame(list(itertools.product(day_ls, timestamp_ls, geohash_ls)), columns = ['day', 'timestamp', 'geohash6'])
        filled_data_df = pd.merge(filled_data_df, data_df, on=['day', 'timestamp', 'geohash6'], how='left').fillna(0)
        filled_data_df = pd.merge(filled_data_df, geohash_df, on='geohash6', how='left')
        data_df = filled_data_df

        # Useful variables
        total_timestamp = len(data_df[['day', 'timestamp']].drop_duplicates())

        # Convert timestamp to total number of minutes starting from 0:0 for easy manipulation
        data_df['total_minutes'] = data_df['timestamp'].apply(lambda timestring: compute_total_minutes(timestring))

        data_df.sort_values(['geohash_index'], ascending=True, inplace=True)

        # Feature engineering
        if use_day:
            data_df['day_feature'] = data_df['day'].apply(lambda x: convert_day_to_one_hot_feature(x))

        if use_cyclical_timestamp:
            data_df['cyclical_timestamp_feature'] = data_df['total_minutes'].apply(lambda x: convert_total_minutes_to_cyclical_feature(x))

        if use_part_of_day:
            data_df['part_of_day_feature'] = data_df['total_minutes'].apply(lambda x: convert_total_minutes_to_part_of_day_one_hot_feature(x))

        # Get sliding features and labels and store them into model_data_dict
        end_idx = ((total_timestamp - num_steps) // 5) * 5 + num_steps - 1
        model_data_dict = collections.OrderedDict()
        for _, (geohash_index, geohash) in geohash_df.iterrows():
            curr_data_df = data_df.iloc[geohash_index*total_timestamp : (geohash_index + 1)*total_timestamp]
            curr_data_df = curr_data_df.sort_values(['day', 'total_minutes'], ascending=True)

            # Store features and labels
            model_data_dict[geohash] = {}

            # Static features
            if use_geohash:
                model_data_dict[geohash]['geohash_feature'] = decode_geohash(geohash, use_minmax_norm=True)

            # Temporal features
            model_data_dict[geohash]['history'] = get_sliding_features(curr_data_df['demand'].values[:end_idx],
                                                                       sliding_len=num_steps, sliding_stride=5)
            model_data_dict[geohash]['label'] = get_sliding_features(curr_data_df['demand'].values[num_steps:],
                                                                     sliding_len=5, sliding_stride=5)

            if sliding_features:    # Indicate if sliding/rolling features are used
                # For LSTM/TCN/SpatioTCN only, currently do not support for LightGBM and MLP
                if use_day:
                    model_data_dict[geohash]['day_feature'] = get_sliding_features(curr_data_df['day_feature'].values[:end_idx],
                                                                                   sliding_len=num_steps, sliding_stride=5)

                if use_cyclical_timestamp:
                    model_data_dict[geohash]['cyclical_timestamp_feature'] = get_sliding_features(curr_data_df['cyclical_timestamp_feature'].values[:end_idx],
                                                                                                  sliding_len=num_steps, sliding_stride=5)

                if use_part_of_day:
                    model_data_dict[geohash]['part_of_day_feature'] = get_sliding_features(curr_data_df['part_of_day_feature'].values[:end_idx],
                                                                                           sliding_len=num_steps, sliding_stride=5)

        # Get features and labels stored in model_data_dict and arrange them into numpy array
        X_ls = []
        y_ls = []
        i = 0
        for geohash, curr_dict in model_data_dict.items():
            i += 1
            curr_features = curr_dict['history']
            curr_labels = curr_dict['label']

            if sliding_features:    # Indicate if sliding/rolling features are used
                # For LSTM/TCN/SpatioTCN only, currently do not support for LightGBM and MLP
                curr_features = curr_features[:, :, np.newaxis]
                if use_geohash:
                    curr_features = np.concatenate((curr_features,
                                                    np.tile(curr_dict['geohash_feature'],
                                                            curr_features.shape[:-1] + (1, ))), axis=2)

                if use_day:
                    curr_features = np.concatenate((curr_features,
                                        np.array(curr_dict['day_feature'].tolist())), axis=2)

                if use_cyclical_timestamp:
                    curr_features = np.concatenate((curr_features,
                                        np.array(curr_dict['cyclical_timestamp_feature'].tolist())), axis=2)

                if use_part_of_day:
                    curr_features = np.concatenate((curr_features,
                                        np.array(curr_dict['part_of_day_feature'].tolist())), axis=2)

            else:
                # Sliding features are not used (for LightGBM and MLP)
                if use_geohash:
                    curr_features = np.hstack((curr_features, np.tile(curr_dict['geohash_feature'], (curr_dict['history'].shape[0], 1))))

            X_ls.append(curr_features)
            y_ls.append(curr_labels)

        X_np = np.array(X_ls).swapaxes(0, 1)
        y_np = np.array(y_ls).swapaxes(0, 1)

        return X_np, y_np

    @property
    def info(self):
        # Useful information
        return {
            'X_train_np.shape': self.X_train_np.shape,
            'y_train_np.shape': self.y_train_np.shape,
            'X_val_np.shape': self.X_val_np.shape,
            'y_val_np.shape': self.y_val_np.shape,
            'X_test_np.shape': self.X_test_np.shape,
            'y_test_np.shape': self.y_test_np.shape,
        }
