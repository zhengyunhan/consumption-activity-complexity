from ecomplexity import ecomplexity
from ecomplexity import proximity
import os

# import economic_complexity as ecplx
import numpy as np
import pandas as pd
# from rca import rca
# from complexity import complexity,getM
from util.util import weighted_mean, weighted_median
import matplotlib.pyplot as plt
import scipy.stats as stats


vol_var='RAW_NUM_CUSTOMERS'

county_dict=pd.read_csv('/data/us_fips_codes_new.csv')
county_dict['st_county_code']=[str(e) if len(str(e))==5 else '0'+str(e) for e in county_dict['st_county_code']]

for year in range(2019,2025):
    df_full=pd.DataFrame()
    for month in range(1,13):
        if len(str(month))<2:
            m='0'+str(month)
        else:
            m=str(month)

        # read amenity-related data
        dirc = '/data/' + str(year) + '/'
        file = dirc + 'df_' + str(year) + '_' + m + '.csv'
        df=pd.read_csv(file)
        df['st_county_code']=[str(e) if len(str(e))==5 else '0'+str(e) for e in df['st_county_code']]

        df_full=pd.concat((df_full,df),axis=0).reset_index(drop=True)

    df_full_gp=df_full.groupby(['st_county_code','CATEGORY_TAGS']).apply(
        lambda x: pd.Series({
            'RAW_NUM_CUSTOMERS':x['RAW_NUM_CUSTOMERS'].sum(skipna=True),
            'RAW_TOTAL_SPEND':x['RAW_TOTAL_SPEND'].sum(skipna=True),
            'count_st_county_prop':x['count_st_county_prop'].sum(skipna=True)/12,
            'count_st_county':x['count_st_county'].sum(skipna=True)/12,
            'mean_income': weighted_mean(x, 'mean_income', 'RAW_NUM_CUSTOMERS'),
            'income_diversity': weighted_mean(x, 'income_diversity', 'RAW_NUM_CUSTOMERS'),
            'median_MEDIAN_DWELL': weighted_median(x, 'median_MEDIAN_DWELL', 'RAW_VISITOR_COUNTS')
        })
    ).reset_index()

####
    df_full_gp['year']=year

    df_pois = df_full_gp.groupby('CATEGORY_TAGS')[[vol_var,'count_st_county']].sum().reset_index()
    df_pois = df_pois[(df_pois['count_st_county'] > 100)]
    df_counties = df_full_gp.groupby('st_county_code')[[vol_var,'count_st_county_prop']].sum().reset_index()
    df_counties = df_counties[(df_counties['count_st_county_prop'] > 10)]

    df_full_gp_filter  = df_full_gp[
    (df_full_gp['st_county_code'].isin(df_counties['st_county_code'])) & 
    (df_full_gp['CATEGORY_TAGS'].isin(df_pois['CATEGORY_TAGS']))
    ]

    df_full_sel=df_full_gp_filter[['year','st_county_code','CATEGORY_TAGS',vol_var]]
    var_cols = {'time':'year', 'loc':'st_county_code', 'prod':'CATEGORY_TAGS', 'val':vol_var}

    cdf_full = ecomplexity(df_full_sel, var_cols)

    #standardize PCI
    PCI=cdf_full[['CATEGORY_TAGS','pci']].groupby('CATEGORY_TAGS').agg('mean').reset_index()
    PCI['pci']=(PCI['pci']-PCI['pci'].mean(skipna=True))/PCI['pci'].std(skipna=True)
    del cdf_full['pci']
    cdf_full=cdf_full.merge(PCI,on='CATEGORY_TAGS')

    save_folder='complexity'
    directory2 = './output/'+save_folder+'/csv/full/'
    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    file_path2 = directory2 + 'disaggregate_' + str(year) + '.csv'
    cdf_full.to_csv(file_path2,index=False)

