import logging
import pandas as pd
from datetime import datetime, timedelta, timezone


def get_latest_record(schema, table, engine, timestamp_column='indexed_timestamp_'):
    sql = f"""
            select *
            from {schema}.{table}
            where {timestamp_column} = (select max({timestamp_column}) from {schema}.{table});
        """
    df = pd.read_sql_query(
        sql=sql,
        con=engine
    )

    return df


def get_last_record_of_day(schema, table, engine, timestamp_column='indexed_timestamp_'):
    sql = f"""
    
            with
                distinct_time as (
                    select distinct {timestamp_column} from {schema}.{table}
                )
                , time_rank as (
                    SELECT
                        {timestamp_column},
                        ROW_NUMBER() OVER(partition by {timestamp_column}::date order by {timestamp_column} desc ) as rank
                    FROM distinct_time
                )
            select *
            from {schema}.{table}
            where {timestamp_column} in (select {timestamp_column} from time_rank where rank = 1)
            and {timestamp_column} > now() - interval '30 days'
            order by {timestamp_column} desc;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=engine
    )

    return df


def get_record_range(schema, table, start_date, end_date, engine, timestamp_column='indexed_timestamp_'):
    start_date_string = datetime.strftime(start_date, '%Y-%m-%d')
    end_date_string = datetime.strftime(end_date, '%Y-%m-%d')
    sql = f"""
            select *
            from {schema}.{table}
            where 
                {timestamp_column}::date >= '{start_date_string}'
                and {timestamp_column}::date <= '{end_date_string}'
            order by {timestamp_column} desc;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=engine
    )

    return df


def get_record_nearest_minute(schema, table, date, engine, timestamp_column='indexed_timestamp_', rounding=5):
    date_string = datetime.strftime(date, '%Y-%m-%d')
    sql = f"""
            with
                extract_time as (
                    select
                        {timestamp_column}
                        , extract(hour from {timestamp_column}) as hour
                        , extract(minute from {timestamp_column}) as minute
                        , extract(hour from to_timestamp(round(( extract ('epoch' from (now() )) ) / (60*{rounding}) ) * (60*{rounding}))) as current_hour
                        , extract(minute from to_timestamp(round(( extract ('epoch' from (now() )) ) / (60*{rounding}) ) * (60*{rounding}))) as current_minute
                    from {schema}.{table}
                    order by {timestamp_column} desc
                )
                , filtered_timestamp as (
                    select
                        {timestamp_column}
                    from extract_time
                    where
                        {timestamp_column}::date = '{date_string}'
                        and hour = current_hour
                        and minute = current_minute
                    order by {timestamp_column} desc
                )
            select *
            from {schema}.{table}
            where {timestamp_column} in (select {timestamp_column} from filtered_timestamp);
        """
    df = pd.read_sql_query(
        sql=sql,
        con=engine
    )

    return df


def get_record_range_nearest_minute(schema, table, start_date, end_date, engine, timestamp_column='indexed_timestamp_', rounding=5):
    start_date_string = datetime.strftime(start_date, '%Y-%m-%d')
    end_date_string = datetime.strftime(end_date, '%Y-%m-%d')
    sql = f"""
            with
                extract_time as (
                    select
                        {timestamp_column}
                        , extract(hour from {timestamp_column}) as hour
                        , extract(minute from {timestamp_column}) as minute
                        , extract(hour from to_timestamp(round(( extract ('epoch' from (now() )) ) / (60*{rounding}) ) * (60*{rounding}))) as current_hour
                        , extract(minute from to_timestamp(round(( extract ('epoch' from (now() )) ) / (60*{rounding}) ) * (60*{rounding}))) as current_minute
                    from {schema}.{table}
                    order by {timestamp_column} desc
                )
                , filtered_timestamp as (
                    select
                        {timestamp_column}
                    from extract_time
                    where
                        {timestamp_column}::date >= '{start_date_string}'
                        and {timestamp_column}::date <= '{end_date_string}'
                        and hour = current_hour
                        and minute = current_minute
                    order by {timestamp_column} desc
                )
            select *
            from {schema}.{table}
            where {timestamp_column} in (select {timestamp_column} from filtered_timestamp);
        """
    df = pd.read_sql_query(
        sql=sql,
        con=engine
    )

    # print(sql)
    return df
