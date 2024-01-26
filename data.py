import polars as pl
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

class TimeSeqDataset:
    def __init__(self, path:str, remove_behaviors:tuple[str]=[], join_behaviors_list:tuple[tuple[str]]=[],
                 name_data:str='DogMoveData', name_info:str='DogInfo'):
        """
        Dataset object for Dog movement analysis.
        Args:
            path: Directory in which .csv files are placed.
            remove_behaviors: Tuple of behaviors that are not included in dataset.
            join_behaviors_list: Tuple of behavior tuples that should be joined together.
            name_data: Name of main data .csv file.
            name_info: Name of info data .csv file.
        """
        # define behavior columns
        self.behavior_cols = [f'Behavior_{i+1}' for i in range(3)]

        # define all feature columns
        sensor_features = [f"{s}{b}_{d}" for s in ('A', 'G') for b in ('Neck', 'Back') for d in ('x', 'y', 'z')]
        sensor_features_magnitude = [f"{s}{b}" for s in ('A', 'G') for b in ('Neck', 'Back')]
        self.all_features = sensor_features + sensor_features_magnitude

        # load DataFrames
        df_data, df_info = self._load_dataframes(path, name_data, name_info)

        # create set of all behavior types
        behavior_set = set()
        for behavior in self.behavior_cols:
            behavior_set.update(df_data.select(behavior).unique().collect().to_series().to_list())
        # don't include undefined movement in behavior set
        behavior_set.remove('<undefined>')


        # remove behaviors in 'remove behaviors' from DataFrame
        df_data = df_data.filter(~pl.any_horizontal(pl.col(self.behavior_cols).is_in(remove_behaviors)))
        # remove behaviors in 'remove behaviors' from behavior set
        behavior_set.difference_update(remove_behaviors)


        for join_behaviors in join_behaviors_list:
            # create name of joined behavior
            new_name = " or ".join(join_behaviors)
            # replace all 'join behaviors' with new name in DataFrame
            df_data = df_data.with_columns(
                *[pl.when(pl.col(b).is_in(join_behaviors)).then(new_name).otherwise(pl.col(b)).alias(b) 
                  for b in self.behavior_cols]
            )
            # remove all separate 'join behaviors' from behavior set
            behavior_set.difference_update(join_behaviors)
            # add joined 'join behaviors' to behavior set
            behavior_set.add(new_name)

        # join DataFrames and add, clarify columns
        df_data = self._transform_dataframes(df_data, df_info)

        # save DataFrame
        self.df_data = df_data
        # create list of class names
        self.class_list = sorted(list(behavior_set))
        # create dictionary linking class names with their respective integer representation
        self.class_dict = {c: i for i, c in enumerate(self.class_list)}

    def get_numerical_data(self, sensor_features:list[str], len_seq:int, seq_offset:float):
        """
        Create timeseries dataset.
        """
        # calculate length of offset from sequence length and offset ratio
        len_offset = int(seq_offset * len_seq)

        # create pandas DataFrame
        df_data = self.df_data.select(['DogID', 'TestNum', 't_sec', *sensor_features, *self.behavior_cols]).collect().to_pandas()

        # initialize variables
        x = []
        y = []
        i = 0

        len_df = len(df_data)
        while i < len_df - len_seq:
            sequence = df_data.iloc[i:i+len_seq, :]
            idx = self._check_valid_sequence(sequence)

            if idx == 0:
                # save valid sequence features and labels
                x.append(sequence[sensor_features].to_numpy().T.flatten())
                y.append(self._create_multi_hot(sequence))
                
                # offset index by sequence offset
                i += len_offset
            else:
                # invalid sequence, start from next index
                i += idx

        return np.array(x), np.array(y)

    def _load_dataframes(self, path:str, name_data:str='DogMoveData',
                         name_info:str='DogInfo') -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Load DataFrames and drop unwanted rows.
        """
        # define keywords that signal synchronization
        synchronization_keywords = ['Synchronization', 'Extra_Synchronization']

        # load DataFrame
        df_info = pl.scan_csv(f'{path}/{name_info}.csv')
        df_data = pl.scan_csv(f'{path}/{name_data}.csv')
        df_data = df_data.filter(
            # keep only rows where at least one behavior is defined
            pl.any_horizontal(pl.col(self.behavior_cols).ne('<undefined>')) & \
            # don't use samples where synchronization took place
            (~pl.any_horizontal(pl.col(self.behavior_cols).is_in(synchronization_keywords)))
        )

        return df_data, df_info

    def _transform_dataframes(self, df_data:pl.LazyFrame, df_info:pl.LazyFrame) -> pl.LazyFrame:
        """
        Joins two DataFrames and adds additional columns (magnitudes, Behavior) and cleans up
        NeuteringStatus, Gender columns.
        """
        # join dataframes
        df_data = df_data.join(df_info, 'DogID')

        # add additional columns, clean up categorical columns
        df_data = df_data.with_columns(
            (pl.col('ABack_x')**2 + pl.col('ABack_y')**2 + pl.col('ABack_z')**2).sqrt().alias('ABack'),
            (pl.col('GBack_x')**2 + pl.col('GBack_y')**2 + pl.col('GBack_z')**2).sqrt().alias('GBack'),
            (pl.col('ANeck_x')**2 + pl.col('ANeck_y')**2 + pl.col('ANeck_z')**2).sqrt().alias('ANeck'),
            (pl.col('GNeck_x')**2 + pl.col('GNeck_y')**2 + pl.col('GNeck_z')**2).sqrt().alias('GNeck'),
            pl.col('NeuteringStatus').map_elements(lambda x: 'Yes' if x == 1 else 'No', return_dtype=str).alias('Neutered'),
            pl.col('Gender').map_elements(lambda x: 'Female' if x == 1 else 'Male', return_dtype=str).alias('Gender'),
            pl.concat_list(self.behavior_cols).alias('Behavior')
        )

        return df_data
    
    def _check_valid_sequence(self, sequence:pd.DataFrame) -> int:
        """
        Returns 0 if sequence valid, otherwise returns index of first element that makes sequence invalid.
        """
        # sequence is valid, if DogID, TestNum, Behavior_1, Behavior_2, Behavior_3 are constant over time-sequence
        # and if all time steps are consecutive (spaced 0.01 s apart)
        idx_list = np.array([
            np.argmax(sequence['DogID'] != sequence['DogID'].iloc[0]),
            np.argmax(sequence['TestNum'] != sequence['TestNum'].iloc[0]),
            np.argmax(sequence['Behavior_1'] != sequence['Behavior_1'].iloc[0]),
            np.argmax(sequence['Behavior_2'] != sequence['Behavior_2'].iloc[0]),
            np.argmax(sequence['Behavior_3'] != sequence['Behavior_3'].iloc[0]),
            np.argmax([False, *(~np.isclose(np.diff(sequence['t_sec']), 0.01))])
        ])

        # return first index that caused sequence to be invalid
        if (idx_list > 0).any():
            return min(idx_list[idx_list > 0])
        
        return 0

    def _create_multi_hot(self, sequence:pd.DataFrame) -> np.ndarray:
        """
        Create multi-hot encoding for a time sequence.
        """
        # initialize one hot encoded vector to zero
        y = np.zeros(len(self.class_list))
        
        # iterate over behavior columns
        for col in self.behavior_cols:
            # get behavior in current behavior column
            behavior = sequence.iloc[0, :][col]
            
            # put 1 at position that corresponds to behavior class
            if behavior != '<undefined>':
                y[self.class_dict[behavior]] = 1

        return y
    

class GroupScaler:
    def __init__(self, features:list[str], seq_len:int):
        """
        Initialize a GroupScaler object that scales all acceleration and all gyroscopic
        data with the same values respectively. This preserves information of the relative
        quantity of each acceleration/gyroscopic feature.
        """
        # get indices for features coming from accelerometer
        self.idx_acc = [np.arange(i*seq_len, (i+1)*seq_len) for i, f in enumerate(features) if 'A' in f]
        self.idx_acc = np.array(self.idx_acc).flatten()
        # get indices for features coming from gyroscope
        self.idx_gyr = [np.arange(i*seq_len, (i+1)*seq_len) for i, f in enumerate(features) if 'G' in f]
        self.idx_gyr = np.array(self.idx_gyr).flatten()

        # initialize RobustScaler for accelerometer and gyroscope data
        # RobustScaler scales based on 1st and 3rd quartile and is therefore less
        # sensitive to outliers
        self.scaler_acc = RobustScaler()
        self.scaler_gyr = RobustScaler()

    def fit(self, x:np.ndarray):
        """
        Fit the GroupScaler object to the given data.
        """
        # fit two scalers on the respective sensor data
        if len(self.idx_acc) > 0:
            self.scaler_acc.fit(x[:, self.idx_acc].reshape(-1, 1))
        if len(self.idx_gyr) > 0:
            self.scaler_gyr.fit(x[:, self.idx_gyr].reshape(-1, 1))

    def transform(self, x:np.ndarray) -> np.ndarray:
        """
        Tranform the given data using the fitted scalers.
        """
        # transform data using both scalers
        if len(self.idx_acc) > 0:
            x_s_acc = self.scaler_acc.transform(x[:, self.idx_acc].reshape(-1, 1))
        if len(self.idx_gyr) > 0:
            x_s_gyr = self.scaler_gyr.transform(x[:, self.idx_gyr].reshape(-1, 1))

        # create new array with shape of original input
        x_s = np.empty_like(x)
        # assign features with original ordering
        if len(self.idx_acc) > 0:
            x_s[:, self.idx_acc] = x_s_acc.reshape(-1, len(self.idx_acc))
        if len(self.idx_gyr) > 0:
            x_s[:, self.idx_gyr] = x_s_gyr.reshape(-1, len(self.idx_gyr))

        return x_s

    def fit_transform(self, x:np.ndarray) -> np.ndarray:
        """
        Fit the GroupScaler to the given data and transform it.
        """
        self.fit(x)
        return self.transform(x)

class TimeSeqPCA:
    def __init__(self, seq_len:int):
        """
        Initialize wrapper around the PCA class of the sklearn package. This class
        ensures that principal component analysis is only applied to the
        different sensor features and not to each time step of the sensor data.
        """
        self.pca = PCA()
        self.seq_len = seq_len

    def fit(self, x:np.ndarray):
        """
        Fit the TimeSeqPCA object to the given data.
        """
        x_pca = self._transform_input(x)
        self.pca.fit(x_pca)
        
    def transform(self, x:np.ndarray, num_features:int) -> np.ndarray:
        """
        Tranform the given data using the fitted pca.
        """
        x_pca = self._transform_input(x)
        output = self.pca.transform(x_pca)[:, :num_features]
        return self._inverse_transform_output(output)
    
    def fit_transform(self, x:np.ndarray, num_features:int) -> np.ndarray:
        """
        Fit the TimeSeqPCA to the given data and transform it.
        """
        self.fit(x)
        return self.transform(x, num_features)

    def _transform_input(self, x:np.ndarray) -> np.ndarray:
        num_samples = x.shape[0]
        tmp = x.reshape(num_samples, -1, self.seq_len)
        num_features = tmp.shape[1]
        return tmp.swapaxes(1, 2).reshape(-1, num_features)
    
    def _inverse_transform_output(self, x:np.ndarray) -> np.ndarray:
        num_features = x.shape[1]
        tmp = x.reshape(-1, self.seq_len, num_features).swapaxes(1, 2)
        return tmp.reshape(-1, num_features*self.seq_len)
