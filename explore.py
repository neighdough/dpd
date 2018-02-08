#TODO
""" Visualzations
        1. animation of permits over time
            
        2. bar chart tracts and value of permits over time
        3. bar chart trats total numbrer of permits over time
        4. permits and tax revenue
        5. permits and tax delinquency
        6. permits and code enforcement violations
        7. permits and population density
    Analysis
        1. PCA of building permits and demolitions
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from caeser import utils
os.chdir('/home/nate/dropbox-caeser/Data/DPD/memphis_30')


#------------------------------------------------------------------------------
#------------------------------Exploratory Work--------------------------------
#------------------------------------------------------------------------------
col = lambda x: [col.lower() for col in x.columns] 
const = pd.read_csv('./data/new_construction_permits.csv')
contr = pd.read_csv('./data/contractor_permits.csv')
perm = pd.read_csv('./data/permits.csv')
const.columns = col(const)
contr.columns = col(contr)
perm.columns = col(perm)
col_diff = lambda x, y: set(x.columns).difference(set(y.columns))
drop_cols = ['loc_name', 'status', 'score', 'match_type', 
                'match_addr', 'side', 'addr_type', 'arc_street']

for df in [const, contr, perm]:
    df.drop(drop_cols, axis=1, inplace=True)
    df['sub_type'] = df['sub_type'].str.lower()
    df['const_type'] = df['const_type'].str.lower()

perm['year'] = perm.issued.str.split('/').str[0]
comb = const.append(contr, ignore_index=True)
comb.year = comb.issued.str[:4]
comb = comb.append(perm, ignore_index=True)
comb['dup'] = comb.duplicated([col for col in comb.columns])
uni = comb[comb.dup == False]

uni.replace({'descriptio':{'\r\n\r\n':' ', '\r\n':' '},
             'fraction':{'MEMP':'Memphis','CNTY':'Memphis','LKLD':'Lakeland',
                         'ARLI':'Arlington', 'MILL':'Millington',np.nan:'Memphis',
                         'BART':'Bartlett','CNY':'Memphis', 'CMTY':'Memphis',
                         'COLL':'Collierville','GTWN':'Germantown', 
                         '`':'Memphis'}}, inplace=True, regex=True)
#replace didn't work when run as part of first replace statement
#regex for 4 or more whitespace chars followed by any number of non-whitespace
#chars followed by any character except line endings
uni.address.replace('\s{4,}[\S]+.*','',inplace=True, regex=True)
uni['state'] = 'TN'
#------------------------------------------------------------------------------
#------------------------------Visualizations----------------------------------
#------------------------------------------------------------------------------

#--------------------------------Heat Maps-------------------------------------

#New Construction
uni['month'] = uni.issued.str[-5:7]
new_cons = uni[uni.const_type == 'new']
grp_new = new_cons.groupby(['month', 'year']).agg({'permit':'count'}).reset_index()
pivot_new = grp_new.pivot('month', 'year', 'permit')
cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, 
        light=.95, as_cmap=True)
sns.heatmap(pivot_new, cmap='Greens')#'Blues')#cmap)
plt.title('New Construction, 2002 to 2016')

#Demolitions
demos = uni[uni.const_type == 'demo']
grp_demo = demos.groupby(['month', 'year']).agg({'permit':'count'}).reset_index()
pivot_demo = grp_demo.pivot('month', 'year', 'permit')
sns.heatmap(pivot_demo, cmap='Purples')#Reds')
plt.title('Demolitions, 2002 to 2016')

#Net Construction
net_const = grp_new.merge(grp_demo, on=['month','year'], 
        suffixes=['const','demo'])
scale = lambda x: (x - x.mean()) / (x.max() - x.min())
net_const['scale_const'] = scale(net_const.permitconst)
net_const['scale_demo'] = scale(net_const.permitdemo)
net_const['net_const'] = net_const.scale_const - net_const.scale_demo
pivot_net = net_const.pivot('month', 'year', 'net_const')
sns.heatmap(pivot_net, cmap='PRGn')#'RdBu')
plt.title('Construction Less Demolition (Normalized), 2002 to 2016')

#------------------------------------------------------------------------------
#------------------------------Random Forest Test------------------------------
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
host = 'localhost'
params = utils.connection_properties(host, db='memphis_30')
engine = utils.connect(**params)
sel_const = ("select g.num_perm, wwl.* from wwl_2016 wwl, "
        "(select geoid10, count(id) num_perm from permits_2002_2016 p "
        "join wwl_2016 w on st_within(p.wkb_geometry, w.wkb_geometry) " 
        "where const_type = 'new' "
        "group by geoid10) g "
        "where g.geoid10 = wwl.geoid10")
const = pd.read_sql(sel_const, engine)
sel_demo = ("select g.num_perm, wwl.* from wwl_2016 wwl, "
        "(select geoid10, count(id) num_perm from permits_2002_2016 p "
        "join wwl_2016 w on st_within(p.wkb_geometry, w.wkb_geometry) " 
        "where const_type = 'demo' "
        "group by geoid10) g "
        "where g.geoid10 = wwl.geoid10")
demo = pd.read_sql(sel_demo, engine)



drop = ['tid', 'index', 'ogc_fid', 'statefp10', 'countyfp10', 'tractce10', 
        'name10', 'geoid10', 'namelsad10', 'mtfcc10', 'funcstat10', 
        'intptlat10', 'intptlon10', 'shape_length', 'shape_area', 'acreland', 
        'wwl_name10', 'wwl_acreland', 'wwl_id', 'wkb_geometry']
col_const = [col for col in const.columns if col not in drop]

geoids_const = const['geoid10']
#const.drop(drop, axis=1, inplace=True)
geoids_demo = demo['geoid10']
#demo.drop(drop, axis=1, inplace=True)
const.fillna(0, inplace=True)
demo.fillna(0, inplace=True)
rf_const = RandomForestRegressor(n_estimators=500)
X,Y = const[col_const], const.num_perm
rfr_const = rf_const.fit(X,Y)
plt.figure(figsize=(30,30))
sns.regplot(const.num_perm, rf_const.predict(X))
featimp_const = zip(X.columns, rfr_const.feature_importances_)
featimp_const = sorted(featimp_const, key=lambda val: val[1], reverse=True)
col_const = [col[0] for col in featimp_const[:50]]

rf_demo = RandomForestRegressor(n_estimators=500)
X,Y = demo[col_demo], demo[demo.columns[-1]]
rf_demo = rf_demo.fit(X,Y)
featimp_demo = zip(X.columns, rf_demo.feature_importances_)
featimp_demo = sorted(featimp_demo, key=lambda val: val[1], reverse=True)
col_demo = [col[0] for col in featimp_demo[:50]]

infill = const[const.geoid10.isin(geoid.geoid)]
infill.drop(drop, axis=1, inplace=True)
infill.fillna(0, inplace=True)
rf_infill = RandomForestRegressor(n_estimators=1000)
X,Y = infill[[col for col in infill.columns[1:]]], infill[infill.columns[1]]
rf_infill = rf_infill.fit(X,Y)
featimp_infill = zip(X.columns, rf_infill.feature_importances_)
featimp_infill = sorted(featimp_infill, key=lambda val: val[1], reverse=True)
col_infill = [col[0] for col in featimp_infill[:50]]

