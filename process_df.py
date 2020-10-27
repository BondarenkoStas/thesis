import pandas as pd
import numpy as np

from functools import reduce
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer, SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.utils import shuffle

#########################################################################
# Fill nan and reskew. Usage:
#    data = {}
#    imputer_func = KNNImputer(n_neighbors=30, weights='distance')
#    process = Process(X_train, X_test, X_val, y_train, y_test, y_val, imputer='func', imputer_func=imputer_func).skew_X().skew_y().fill_nan()
#    data['X_train'], data['X_test'], data['X_val'], data['y_train'], data['y_test'], data['y_val'] = process.return_processed()
#########################################################################

class Process:       
    def __init__(self, 
                 X_train, X_test, X_val, y_train, y_test, y_val, 
                 imputer='mean',
                 imputer_func=None,
                 minmaxrange=(-1,1), 
                 standardize_X=True,
                 standardize_y=True,
                 robust_range=(10, 90),
                ):
        
        self.y_process = []
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.X_val = X_val.copy()
        self.y_train = y_train.copy().values.reshape(-1, 1)
        self.y_test = y_test.copy().values.reshape(-1, 1)
        self.y_val = y_val.copy().values.reshape(-1, 1)
        self.imputer = imputer
        self.imputer_func = imputer_func
        self.X_minmaxscaler = MinMaxScaler(feature_range=minmaxrange)
        self.y_minmaxscaler = MinMaxScaler(feature_range=minmaxrange)
        self.y_robust_scaler = RobustScaler(quantile_range=robust_range)
        self.X_robust_scaler = RobustScaler(quantile_range=robust_range)
        self.skewer_X = PowerTransformer(standardize=standardize_X)
        self.skewer_y = PowerTransformer(standardize=standardize_y)

    def fill_nan(self):
        def fill_train_test(imputer):
            imputer.fit(self.X_train)
            self.X_train[:] = imputer.transform(self.X_train)
            self.X_test[:] = imputer.transform(self.X_test)
            self.X_val[:] = imputer.transform(self.X_val)

        if (self.imputer == 'mean'):
            fill_train_test(SimpleImputer(strategy='mean'))
        elif (self.imputer == 'median'):
            fill_train_test(SimpleImputer(strategy='median'))
        elif (self.imputer == 'knn'):
            fill_train_test(KNNImputer())
        elif (self.imputer == 'iterative'):
            fill_train_test(IterativeImputer(verbose=0, max_iter=50))
        elif (self.imputer == 'iterative_mlp'):
            fill_train_test(
                IterativeImputer(
                    estimator=MLPRegressor(learning_rate='adaptive', random_state=0),
                    verbose=2,
                )
            )
        elif (self.imputer == 'func' and self.imputer_func):
            fill_train_test(self.imputer_func)
        return self
        
    def minmaxscale_X(self):
        self.X_minmaxscaler.fit(self.X_train)
        self._apply_func_to_X(self.X_minmaxscaler.transform)
        return self
    
    def minmaxscale_X_inverse(self, data):
        return self.X_minmaxscaler.inverse_transform(data)
    
    def minmaxscale_Y(self, inverse=False):
        self.y_minmaxscaler.fit(self.y_train)
        self._apply_func_to_y(self.y_minmaxscaler.transform)
        self.y_process.append(self.minmaxscale_Y_inverse)
        return self
    
    def minmaxscale_Y_inverse(self, data): # data is in 1d array
        return self.y_minmaxscaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def robustscale_X(self):
        self.X_robust_scaler.fit(self.X_train)
        self._apply_func_to_X(self.X_robust_scaler.transform)
        return self
    
    def robustscale_X_inverse(self, data):
        return self.X_robust_scaler.inverse_transform(data)
    
    def robustscale_Y(self, inverse=False):
        self.y_robust_scaler.fit(self.y_train)
        self._apply_func_to_y(self.y_robust_scaler.transform)
        self.y_process.append(self.robustscale_Y_inverse)
        return self
    
    def robustscale_Y_inverse(self, data): # data is in 1d array
        return self.y_robust_scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def skew_X(self, inverse=False):
        self.skewer_X.fit(self.X_train)
        self._apply_func_to_X(self.skewer_X.transform)
        return self
    
    def skew_X_inverse(self, data):
        return self.skewer_X.inverse_transform(data)
    
    def skew_y(self):
        self.skewer_y.fit(self.y_train)
        self._apply_func_to_y(self.skewer_y.transform, inside=False)
        self.y_process.append(self.skew_y_inverse)
        return self

    def skew_y_inverse(self, data): # data is 1d ndarray
        return self.skewer_y.inverse_transform(data.reshape(-1, 1)).flatten()

    def return_processed(self):
        return self.X_train, self.X_test, self.X_val, self.y_train.flatten(), self.y_test.flatten(), self.y_val.flatten()  
    
    def _apply_func_to_X(self, func):
        self.X_train[:] = func(self.X_train)
        self.X_test[:] = func(self.X_test)
        self.X_val[:] = func(self.X_val)
        
    def _apply_func_to_y(self, func, inside=True):
        if inside:
            self.y_train[:] = func(self.y_train)
            self.y_test[:] = func(self.y_test)
            self.y_val[:] = func(self.y_val)
        else:
            self.y_train = func(self.y_train)
            self.y_test = func(self.y_test)
            self.y_val = func(self.y_val)


##################################################################
# Fill average values
# use df = create_average_columns_all(df)
##################################################################

def get_groups(column, iteration, df): 
    def concat_df(df, column, iteration):
        def_columns = ['budget', 'META__revenue', 'META__year', 'META__date']
        if len(iteration) > 1:
            return pd.concat([df[[f'{column}_{i}'] + def_columns].rename(columns={f'{column}_{i}': column}) for i in iteration])
        return df[list(set([column] + def_columns))]
    return concat_df(df, column, iteration).groupby(column)

def set_info(movie, groups, column_to_iterate, new_column, iteration, cut_date, *argv):
    for i in iteration:
        full_column = new_column if len(iteration) == 1 else f'{new_column}_{i}'
        value = movie[column_to_iterate] if len(iteration) == 1 else movie[f'{column_to_iterate}_{i}']
        if value in groups.groups.keys():
            group = groups.get_group(value)
            if cut_date:
                group = group[group.META__date < movie['META__date']]
        else:
            group = pd.DataFrame()
        movie = reduce(lambda movie, func: func(movie, full_column, group), argv, movie)
    return movie

def get_group_avg_revenue(group): 
    return np.rint(np.mean(group['META__revenue'].values))
def get_group_avg_profit(group): 
    return np.rint(np.mean(group['META__revenue'].values - group['budget'].values))

def set_movie_column(movie, column, value): 
    movie[column] = value
    return movie

def set_avg_profit(movie, column, group): 
    return set_movie_column(movie, f'{column}_avg_profit', get_group_avg_profit(group) if not group.empty else np.nan)
def set_avg_revenue(movie, column, group): 
    return set_movie_column(movie, f'{column}_avg_revenue', get_group_avg_revenue(group) if not group.empty else np.nan)
def set_movies_before(movie, column, group): 
    return set_movie_column(movie, f'{column}_movies_before', group.shape[0])
def set_experience(movie, column, group): 
    return set_movie_column(movie, f'{column}_experience', movie['META__year'] - group['META__year'].min() if not group.empty else 0)

def set_info_cast(movie, groups): 
    return set_info(movie, groups, 'META__cast', 'cast', range(1,9), True, set_avg_profit, set_avg_revenue, set_experience, set_movies_before)
def set_info_year(movie, groups): 
    return set_info(movie, groups, 'META__year', 'year', range(1), False, set_avg_profit, set_avg_revenue)
def set_info_production_company(movie, groups): 
    return set_info(movie, groups, 'META__production_company', 'production_company', range(1,4), True, set_avg_profit, set_avg_revenue, set_movies_before)
def set_info_collections(movie, groups):
    return set_info(movie, groups, 'META__collection_name', 'collection', range(1), True, set_avg_profit, set_avg_revenue)

def set_info_crew(movie, dict_groups_crew):
    col_producer = 'META__crew__production__producer'
    movie = set_info(movie, dict_groups_crew[col_producer], col_producer, col_producer[6:], range(1,3), True, 
                     set_avg_profit, set_avg_revenue, set_movies_before)
    crew_columns = [column for column in dict_groups_crew.keys() if 'META__crew' in column and not col_producer in column]
    for column in crew_columns:
        movie = set_info(movie, dict_groups_crew[column], column, column[6:], range(1), True, set_avg_profit, set_avg_revenue, set_movies_before)
    return movie

def set_info_cast_avg(movie):
    def set_cast_avg_column(movie, column):
        df_columns = [f'cast_{i}_{column}' for i in range(1,9)]
        return set_movie_column(movie, f'cast_avg_{column}', np.nanmean(list(map(movie.get, df_columns))))
    return reduce(lambda movie, column: set_cast_avg_column(movie, column), ['avg_revenue', 'avg_profit', 'experience'], movie)

def set_all_info_for_movie(movie, verbose, groups_collection, groups_year, groups_cast, groups_company, dict_groups_crew):
    if verbose and not movie.name % verbose:
        print (movie.name)
    movie = dict(movie)
    movie = set_info_cast(movie, groups_cast)
    movie = set_info_production_company(movie, groups_company)
    movie = set_info_collections(movie, groups_collection)
    movie = set_info_crew(movie, dict_groups_crew)
    movie = set_info_year(movie, groups_year)
    movie = set_info_cast_avg(movie)
    return pd.Series(movie)

def create_average_columns(df, full_df=None, verbose=None):
    df_ref = df if full_df is None else full_df
    groups_collection = get_groups('META__collection_name', range(1), df_ref)
    groups_year = get_groups('META__year', range(1), df_ref)
    groups_cast = get_groups('META__cast', range(1,9), df_ref)
    groups_company = get_groups('META__production_company', range(1,4), df_ref)

    dict_groups_crew = {'META__crew__production__producer': get_groups('META__crew__production__producer', range(1,3), df_ref)}
    crew_columns = [column for column in df.columns if 'META__crew' in column and not 'production__producer' in column]
    for crew_column in crew_columns:
        dict_groups_crew[crew_column] = get_groups(crew_column, range(1), df_ref)

    return df.apply(set_all_info_for_movie, args=(verbose, groups_collection, groups_year, groups_cast, groups_company, dict_groups_crew), axis=1)


######################################################
# Split and process
######################################################
def split_process_df(df_raw, reskew_X=True, reskew_y=True, train=0.8, test=0.1):
    def get_train_test_revenue(df):
        X = df.drop(['META__revenue'], axis=1)
        y = df['META__revenue']
        return X, y

    df = shuffle(df_raw, random_state=0)

    num_in_train = int(df.shape[0]*0.8)
    num_in_test = int(df.shape[0]*0.1)
    df_train = df[:num_in_train].copy()
    df_test = df[num_in_train:num_in_train+num_in_test].copy()
    df_val = df[num_in_train+num_in_test:].copy()
    X_train, y_train = get_train_test_revenue(df_train)
    X_test, y_test = get_train_test_revenue(df_test)
    X_val, y_val = get_train_test_revenue(df_val)
    
    data = {}
    imputer_func = KNNImputer(n_neighbors=30, weights='distance')
    process = Process(X_train, X_test, X_val, y_train, y_test, y_val, imputer='func', imputer_func=imputer_func)
    if reskew_X:
        process = process.skew_X()
    if reskew_y:
        process = process.skew_y()
    process = process.fill_nan()
    data['X_train'], data['X_test'], data['X_val'], data['y_train'], data['y_test'], data['y_val'] = process.return_processed()
    return data, process

def split_df(df_raw, train=0.8, test=0.1):
    def get_train_test_revenue(df):
        X = df.drop(['META__revenue'], axis=1)
        y = df['META__revenue']
        return X, y

    df = shuffle(df_raw, random_state=0)

    num_in_train = int(df.shape[0]*0.8)
    num_in_test = int(df.shape[0]*0.1)
    df_train = df[:num_in_train].copy()
    df_test = df[num_in_train:num_in_train+num_in_test].copy()
    df_val = df[num_in_train+num_in_test:].copy()
    X_train, y_train = get_train_test_revenue(df_train)
    X_test, y_test = get_train_test_revenue(df_test)
    X_val, y_val = get_train_test_revenue(df_val)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
    }