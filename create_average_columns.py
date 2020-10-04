import pandas as pd
import numpy as np

from functools import reduce


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
    return set_info(movie, groups, 'META__cast', 'cast', range(1,9), True, set_avg_profit, set_avg_revenue, set_movies_before, set_experience)
def set_info_year(movie, groups): 
    return set_info(movie, groups, 'META__year', 'year', range(1), False, set_avg_profit, set_avg_revenue)
def set_info_production_company(movie, groups): 
    return set_info(movie, groups, 'META__production_company', 'production_company', range(1,4), True, set_avg_profit, set_avg_revenue)
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
    return reduce(lambda movie, column: set_cast_avg_column(movie, column), ['avg_revenue', 'avg_profit', 'experience', 'movies_before'], movie)

def set_all_info_for_movie(movie, groups_collection, groups_year, groups_cast, groups_company, dict_groups_crew):
    if not movie.name % 100:
        print (movie.name)
    movie = dict(movie)
    movie = set_info_cast(movie, groups_cast)
    movie = set_info_production_company(movie, groups_company)
    movie = set_info_collections(movie, groups_collection)
    movie = set_info_crew(movie, dict_groups_crew)
    movie = set_info_year(movie, groups_year)
    movie = set_info_cast_avg(movie)
    return pd.Series(movie)

def create_average_columns_all(df, full_df=None):
    df_ref = df if full_df is None else full_df
    groups_collection = get_groups('META__collection_name', range(1), df_ref)
    groups_year = get_groups('META__year', range(1), df_ref)
    groups_cast = get_groups('META__cast', range(1,9), df_ref)
    groups_company = get_groups('META__production_company', range(1,4), df_ref)

    dict_groups_crew = {'META__crew__production__producer': get_groups('META__crew__production__producer', range(1,3), df_ref)}
    crew_columns = [column for column in df.columns if 'META__crew' in column and not 'production__producer' in column]
    for crew_column in crew_columns:
        dict_groups_crew[crew_column] = get_groups(crew_column, range(1), df_ref)

    return df.apply(set_all_info_for_movie, args=(groups_collection, groups_year, groups_cast, groups_company, dict_groups_crew), axis=1)