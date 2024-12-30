import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from util.util import weighted_mean, weighted_median
import time
import json

county_dict=pd.read_csv('/data/us_fips_codes_new.csv')
county_dict['st_county_code']=[str(e) if len(str(e))==5 else '0'+str(e) for e in county_dict['st_county_code']]

unit_col=['st_county_code', 'REGION','CATEGORY_TAGS','TAG_COUNT']
var_sp_col=['RAW_TOTAL_SPEND', 'RAW_NUM_TRANSACTIONS',
    'RAW_NUM_CUSTOMERS', 'MEDIAN_SPEND_PER_TRANSACTION',
    'MEDIAN_SPEND_PER_CUSTOMER','mean_income','income_diversity']
var_vs_col=['RAW_VISIT_COUNTS', 'RAW_VISITOR_COUNTS','DISTANCE_FROM_HOME', 'MEDIAN_DWELL']


for year in range(2019,2025):
    print(year)
    for month in range(1,13):
        print(month)
        if len(str(month))<2:
            m='0'+str(month)
        else:
            m=str(month)

        directory = '/data/raw_data/' + str(year) +'/'
        file_path = directory + 'df_' + str(year) + '_' + m + '.csv'
        df_full=pd.read_csv(file_path)

        # Split tag data
        df_full['CATEGORY_TAGS'] = df_full['CATEGORY_TAGS'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

        df_full['TAG_COUNT'] = df_full['CATEGORY_TAGS'].apply(len)

        df_exploded = df_full.explode('CATEGORY_TAGS')

        df_exploded.reset_index(drop=True, inplace=True)

        df_exploded['cbg_code'] = df_exploded['POI_CBG'].apply(
            lambda e: str(int(e)).zfill(12) if pd.notnull(e) else 'NaN'
        )
        df_exploded['st_county_code']=df_exploded['cbg_code'].apply(lambda x:x[:5])

        df_exploded2=df_exploded.copy()

        df_exploded2['BUCKETED_CUSTOMER_INCOMES'] = df_exploded2['BUCKETED_CUSTOMER_INCOMES'].apply(lambda x: json.loads(x.replace("'", '"')))

        income_brackets = {
            '<25k': 12500,
            '25-45k': 35000,
            '45-60k': 52500,
            '60-75k': 67500,
            '75-100k': 87500,
            '100-150k': 125000,
            '>150k': 175000
        }
        def calculate_mean_income(income_dict):
            total_income = 0
            total_count = 0
            for bracket, count in income_dict.items():
                if bracket in income_brackets:
                    midpoint = income_brackets[bracket]
                    total_income += midpoint * count
                    total_count += count
            return total_income / total_count if total_count > 0 else 0

        def calculate_income_exposure_diversity(income_dict):
            total_count = sum(income_dict.values())
            if total_count == 0:
                return 0
            uniform_proportion = 1 / 7  # There are 7 income brackets
            sum_absolute_diff = 0
            for count in income_dict.values():
                proportion = count / total_count
                sum_absolute_diff += abs(proportion - uniform_proportion)
            return (7 / 12) * sum_absolute_diff

        df_exploded2['mean_income'] = df_exploded2['BUCKETED_CUSTOMER_INCOMES'].apply(calculate_mean_income)
        df_exploded2['income_diversity'] = df_exploded2['BUCKETED_CUSTOMER_INCOMES'].apply(calculate_income_exposure_diversity)


        df_exploded3=df_exploded2[unit_col+var_sp_col+var_vs_col]

        df_exploded4=df_exploded3.copy()
        df_exploded4['RAW_TOTAL_SPEND']=df_exploded3['RAW_TOTAL_SPEND']/df_exploded3['TAG_COUNT']
        df_exploded4['RAW_NUM_TRANSACTIONS']=df_exploded3['RAW_NUM_TRANSACTIONS']/df_exploded3['TAG_COUNT']
        df_exploded4['RAW_NUM_CUSTOMERS']=df_exploded3['RAW_NUM_CUSTOMERS']/df_exploded3['TAG_COUNT']
        df_exploded4['RAW_VISIT_COUNTS']=df_exploded3['RAW_VISIT_COUNTS']/df_exploded3['TAG_COUNT']
        df_exploded4['RAW_VISITOR_COUNTS']=df_exploded3['RAW_VISITOR_COUNTS']/df_exploded3['TAG_COUNT']
        df_exploded4['count_st_county_prop']=1/df_exploded3['TAG_COUNT']

        df_exploded4['RAW_TOTAL_SPEND_org']=df_exploded3['RAW_TOTAL_SPEND']
        df_exploded4['RAW_NUM_TRANSACTIONS_org']=df_exploded3['RAW_NUM_TRANSACTIONS']
        df_exploded4['RAW_NUM_CUSTOMERS_org']=df_exploded3['RAW_NUM_CUSTOMERS']
        df_exploded4['RAW_VISIT_COUNTS_org']=df_exploded3['RAW_VISIT_COUNTS']
        df_exploded4['RAW_VISITOR_COUNTS_org']=df_exploded3['RAW_VISITOR_COUNTS']

        print('Beginning aggregation ...')
        start_time=time.time()
        df_ct_tag=df_exploded4.groupby(['st_county_code','CATEGORY_TAGS']).apply(
            lambda x: pd.Series({
                'count_st_county_prop':x['count_st_county_prop'].sum(skipna=True),
                'count_st_county': len(x),
                'RAW_TOTAL_SPEND':x['RAW_TOTAL_SPEND'].sum(skipna=True),
                'RAW_NUM_TRANSACTIONS':x['RAW_NUM_TRANSACTIONS'].sum(skipna=True),
                'RAW_NUM_CUSTOMERS':x['RAW_NUM_CUSTOMERS'].sum(skipna=True),
                'mean_income': weighted_mean(x, 'mean_income', 'RAW_NUM_CUSTOMERS'),
                'income_diversity': weighted_mean(x, 'income_diversity', 'RAW_NUM_CUSTOMERS'),
                'RAW_VISIT_COUNTS':(x['RAW_VISIT_COUNTS']).sum(skipna=True),
                'RAW_VISITOR_COUNTS':(x['RAW_VISITOR_COUNTS']).sum(skipna=True),
                'median_DISTANCE_FROM_HOME': weighted_median(x, 'DISTANCE_FROM_HOME', 'RAW_VISITOR_COUNTS'),
                'median_MEDIAN_DWELL': weighted_median(x, 'MEDIAN_DWELL', 'RAW_VISITOR_COUNTS'),
            })
        ).reset_index()

        end_time=time.time()
        print('Agg time: ', int(end_time-start_time), ' seconds')

        df_ct_tag['year']=year
        df_ct_tag['month']=month

        df_ct_tag2=df_ct_tag.merge(county_dict,on='st_county_code',how='left')
        df_ct_tag2 = df_ct_tag2[['county'] + [col for col in df_ct_tag2.columns if col != 'county']]
        df_ct_tag2

        directory2 = '/data/output/' + str(year) + '/'
        file_path2 = directory2 + 'df_' + str(year) + '_' + m + '.csv'

        # Check if the directory exists, and if not, create it
        if not os.path.exists(directory2):
            os.makedirs(directory2)

        # Save the DataFrame to the specified CSV file
        df_ct_tag2.to_csv(file_path2, index=False)

