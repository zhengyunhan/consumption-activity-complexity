import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from tqdm import tqdm
import util.util
importlib.reload(util.util)
from util.util import weighted_mean, weighted_median, calculate_entropy_hhi
import statsmodels.api as sm
import os
from sklearn.linear_model import LinearRegression
import seaborn as sns

input_folder='activity_complexity'
dis_full=pd.DataFrame()
for year in range(2019,2024):
    dirc=input_folder+'/csv/disaggregate_'+str(year)+'.csv'
    dis_yr=pd.read_csv(dirc)
    dis_full=pd.concat((dis_full,dis_yr),axis=0)

dis_full_sorted = dis_full.sort_values(by=['st_county_code', 'CATEGORY_TAGS', 'year'])
dis_full_sorted['mcp_lag'] = dis_full_sorted.groupby(['st_county_code', 'CATEGORY_TAGS'])['mcp'].shift(1) # 'mcp' indicates M_ij
dis_full_sorted['mcp_lead'] = dis_full_sorted.groupby(['st_county_code', 'CATEGORY_TAGS'])['mcp'].shift(-1)

dis_full_sorted['mcp_change']=dis_full_sorted['mcp_lead']-dis_full_sorted['mcp']

#Separate the data into two groups: those without the relative advantage (dis_full_noadv) and those with the advantage (dis_full_adv).
dis_full_noadv=dis_full_sorted.loc[dis_full_sorted['mcp']==0,].reset_index(drop=True)
dis_full_noadv=dis_full_noadv.loc[~dis_full_noadv['mcp_change'].isna()]
dis_full_adv=dis_full_sorted.loc[dis_full_sorted['mcp']==1,].reset_index(drop=True)
dis_full_adv=dis_full_adv.loc[~dis_full_adv['mcp_change'].isna()]

# Remove the outliers

Q1 = dis_full_noadv['density'].quantile(0.25)
Q3 = dis_full_noadv['density'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dis_full_noadv2 = dis_full_noadv[(dis_full_noadv['density'] >= lower_bound) & 
                               (dis_full_noadv['density'] <= upper_bound)].reset_index(drop=True)

Q1 = dis_full_adv['density'].quantile(0.25)
Q3 = dis_full_adv['density'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dis_full_adv2 = dis_full_adv[(dis_full_adv['density'] >= lower_bound) & 
                               (dis_full_adv['density'] <= upper_bound)].reset_index(drop=True)


# Bootstrapping
entry_prob = dis_full_noadv2[['mcp_change', 'density']].copy()
entry_prob['density_bin'] = pd.cut(entry_prob['density'], bins=10)  # Example: 10 bins

# Step 2: Perform bootstrap sampling within each bin
n_iterations = 20
bootstrap_samples = []
slopes = []

# Function to perform bootstrap sampling and regression for each bin
sample_data_full=pd.DataFrame()
for bin_label, bin_data in tqdm(entry_prob.groupby('density_bin')):
    for _ in range(n_iterations):
        # Step 2.1: Bootstrap sampling within the current bin
        sample_data = bin_data.sample(frac=1, replace=True)  # Sample with replacement
        sample_data['group']=_
        sample_data_full=pd.concat((sample_data_full,sample_data),axis=0)

sample_final=sample_data_full.groupby(['density_bin','group']).agg('mean').reset_index()

exit_prob = dis_full_adv2[['mcp_change', 'density']].copy()
exit_prob['mcp_change']=abs(exit_prob['mcp_change'])
exit_prob['density_bin'] = pd.cut(exit_prob['density'], bins=10)  # Example: 10 bins

# Step 2: Perform bootstrap sampling within each bin
n_iterations = 20
bootstrap_samples = []
slopes = []

# Function to perform bootstrap sampling and regression for each bin
sample_data_full_exit=pd.DataFrame()
for bin_label, bin_data in tqdm(exit_prob.groupby('density_bin')):
    for _ in range(n_iterations):
        # Step 2.1: Bootstrap sampling within the current bin
        sample_data = bin_data.sample(frac=1, replace=True)  # Sample with replacement
        sample_data['group']=_
        sample_data_full_exit=pd.concat((sample_data_full_exit,sample_data),axis=0)


sample_final_exit=sample_data_full_exit.groupby(['density_bin','group']).agg('mean').reset_index()


# Plotting the entry probabilities

plt.rcParams.update({
    'axes.labelsize': 18,  
    'xtick.labelsize': 18, 
    'ytick.labelsize': 18, 
    'font.size': 18
})

colors = ['#e87072','#ffd19d','#afdcef','#86a1d0']

legend_dic={0:'Low',1:'Mid-Low',2:'Mid-High',3:'High'}

# Create the figure
plt.figure(figsize=(10, 7))

X = sample_final['density']
y = sample_final['mcp_change']
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()

# Extract slope and standard deviation
slope = model.params['density']
slope_std = model.bse['density']
r_squared = model.rsquared
    
sns.regplot(x='density', y='mcp_change', data=sample_final, 
        scatter_kws={'color': 'green', 's': 20, 'label': None}, 
        line_kws={'color': 'green', 'linewidth': 2, 'label': f'{slope:.3f} ({slope_std:.3f})'},
        ci=95) 

plt.xlabel('Relatedness Density')
plt.ylabel('Entry Probability')


plt.text(0.3, 0.8, f'Slope = {slope:.3f} ({slope_std:.3f})\nR² = {r_squared:.3f}', 
         horizontalalignment='left', 
         verticalalignment='center', 
         transform=plt.gca().transAxes)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


directory2 = '/plot_paper/entry_exit/'
file_name='full_entry.png'
plt.savefig(directory2 + file_name, dpi=300, bbox_inches='tight')


#Plotting the exit probabilities

plt.rcParams.update({
    'axes.labelsize': 18, 
    'xtick.labelsize': 18,  
    'ytick.labelsize': 18,  
    'font.size': 18
})


colors = ['#e87072','#ffd19d','#afdcef','#86a1d0']

legend_dic={0:'Low',1:'Mid-Low',2:'Mid-High',3:'High'}

plt.figure(figsize=(10, 7))

X = sample_final_exit['density']
y = sample_final_exit['mcp_change']
X = sm.add_constant(X)  
model = sm.OLS(y, X).fit()

slope = model.params['density']
slope_std = model.bse['density']
r_squared = model.rsquared
    

sns.regplot(x='density', y='mcp_change', data=sample_final_exit, 
        scatter_kws={'color': 'green', 's': 20, 'label': None}, 
        line_kws={'color': 'green', 'linewidth': 2, 'label': f'{slope:.3f} ({slope_std:.3f})'},
        ci=95) 

plt.xlabel('Relatedness Density')
plt.ylabel('Exit Probability')


plt.text(0.3, 0.8, f'Slope = {slope:.3f} ({slope_std:.3f})\nR² = {r_squared:.3f}', 
         horizontalalignment='left', 
         verticalalignment='center', 
         transform=plt.gca().transAxes)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


directory2 = '/plot_paper/entry_exit/'
file_name='full_exit.png'
plt.savefig(directory2 + file_name, dpi=300, bbox_inches='tight') 

###### Figure 4c and d
input_folder='activity_complexity'
dirc='/'+input_folder+'/'
df=pd.read_csv(dirc+'df.csv')
num_quantiles = 4  

df['gdp_pc19_quantile'] = pd.qcut(df['gdp_pc19'], num_quantiles, labels=False)
dis_full_noadv3=dis_full_noadv2.merge(df[['st_county_code','gdp_pc19_quantile']],on='st_county_code')
dis_full_adv3=dis_full_adv2.merge(df[['st_county_code','gdp_pc19_quantile']],on='st_county_code')

n_iterations = 20

sample_data_full_q=pd.DataFrame()
for quantile in range(4):
    dis_full_noadv3_q=dis_full_noadv3.loc[dis_full_noadv3['gdp_pc19_quantile']==quantile,]
    entry_prob = dis_full_noadv3_q[['mcp_change', 'density']].copy()
    entry_prob['density_bin'] = pd.cut(entry_prob['density'], bins=10) 
    for bin_label, bin_data in tqdm(entry_prob.groupby('density_bin')):
        for _ in range(n_iterations):
            sample_data = bin_data.sample(frac=1, replace=True) 
            sample_data['gdp_pc19_quantile']=quantile
            sample_data['sample_group']=_
            sample_data_full_q=pd.concat((sample_data_full_q,sample_data),axis=0)

sample_final_q=sample_data_full_q.groupby(['density_bin','sample_group','gdp_pc19_quantile']).agg('mean').reset_index()


sample_data_full_exit_q=pd.DataFrame()
for quantile in range(4):
    dis_full_adv3_q=dis_full_adv3.loc[dis_full_adv3['gdp_pc19_quantile']==quantile,]
    exit_prob = dis_full_adv3_q[['mcp_change', 'density']].copy()
    exit_prob['mcp_change']=abs(exit_prob['mcp_change'])
    exit_prob['density_bin'] = pd.cut(exit_prob['density'], bins=10) 
    for bin_label, bin_data in tqdm(exit_prob.groupby('density_bin')):
        for _ in range(n_iterations):
            sample_data = bin_data.sample(frac=1, replace=True)
            sample_data['gdp_pc19_quantile']=quantile
            sample_data['sample_group']=_
            sample_data_full_exit_q=pd.concat((sample_data_full_exit_q,sample_data),axis=0)
sample_final_exit_q=sample_data_full_exit_q.groupby(['density_bin','sample_group','gdp_pc19_quantile']).agg('mean').reset_index()

# Making plots
colors = ['#e87072','#ffd19d','#afdcef','#86a1d0']

legend_dic={0:'Low',1:'Mid-Low',2:'Mid-High',3:'High'}

plt.figure(figsize=(10, 7))

for i, quantile in enumerate(range(4)):
    quantile_data = sample_final_q[sample_final_q['gdp_pc19_quantile'] == quantile]

    X = quantile_data['density']
    y = quantile_data['mcp_change']
    X = sm.add_constant(X)  
    model = sm.OLS(y, X).fit()

    slope = model.params['density']
    slope_std = model.bse['density']

    sns.regplot(
        x='density',
        y='mcp_change',
        data=quantile_data,
        scatter_kws={'color': colors[i], 's': 20, 'label': None},
        line_kws={'color': colors[i], 'linewidth': 2, 'label': f'{legend_dic[quantile]}: {slope:.3f} ({slope_std:.3f})'},
        ci=95
    )

plt.xlabel('Relatedness Density')
plt.ylabel('Entry Probability')

plt.legend(title='Economic Groups (Slope ± SE)', loc='lower right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

directory2 = '/plot_paper/entry_exit/'
file_name='quantile_entry.png'
plt.savefig(directory2 + file_name, dpi=300, bbox_inches='tight') 

# Exit probabilities
colors = ['#e87072','#ffd19d','#afdcef','#86a1d0']
legend_dic={0:'Low',1:'Mid-Low',2:'Mid-High',3:'High'}

plt.figure(figsize=(10, 7))

for i, quantile in enumerate(range(4)):
    quantile_data = sample_final_exit_q[sample_final_exit_q['gdp_pc19_quantile'] == quantile]
    X = quantile_data['density']
    y = quantile_data['mcp_change']
    X = sm.add_constant(X) 
    model = sm.OLS(y, X).fit()

    slope = model.params['density']
    slope_std = model.bse['density']

    sns.regplot(
        x='density',
        y='mcp_change',
        data=quantile_data,
        scatter_kws={'color': colors[i], 's': 20, 'label': None},
        line_kws={'color': colors[i], 'linewidth': 2, 'label': f'{legend_dic[quantile]}: {slope:.3f} ({slope_std:.3f})'},
        ci=95
    )

plt.xlabel('Relatedness Density')
plt.ylabel('Exit Probability')

plt.legend(title='Economic Groups (Slope ± SE)', loc='upper right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


directory2 = '/plot_paper/entry_exit/'
file_name='quantile_exit.png'
plt.savefig(directory2 + file_name, dpi=300, bbox_inches='tight') 


