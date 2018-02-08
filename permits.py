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
import itertools
os.chdir('/home/nate/dropbox-caeser/Data/DPD/memphis_30')

params_30 = utils.connection_properties('caeser-geo.memphis.edu',
                                            db='memphis_30')
engine_30 = utils.connect(**params_30)
params_mph = utils.connection_properties('caeser-midt.memphis.edu',
                                            db='blight_data')
engine_mph = utils.connect(**params_mph)

sql_perm_pd = ("select distinct on(permit) permit, issued, sub_type, "
                    "const_type, valuation, fraction, zip_code, sq_ft, pd.name "
                    "from permits_2002_2016 p "
                "join (select wkb_geometry, name from planning_districts) pd "
                    "on st_within(p.wkb_geometry, pd.wkb_geometry)")
perm_pd = pd.read_sql(sql_perm_pd, engine_30)
sql_perm_tract = ("select permit, issued, sub_type, const_type, valuation, "
                    "fraction, zip_code, sq_ft, t.geoid "
                    "from permits_2002_2016 p "
                "join (select wkb_geometry, geoid from tiger_tract_2016) t "
                    "on st_within(p.wkb_geometry, t.wkb_geometry)")
perm_tract = pd.read_sql(sql_perm_tract, engine_30)
sql_par_pd = ("select parcelid, count(i.index) num_ce, count(p.id) num_demo, "
			"sum(valuation) tot_val, pd.name "
                    "from sca_parcels par "
                    "left join (select id, wkb_geometry "
                               "from permits where const_type = 'demo') p "
                            "on st_within(p.wkb_geometry, par.wkb_geometry) " 
                    "left join (select valuation, wkb_geometry "
                               "from permits where const_type = 'new') v "
                            "on st_within(v.wkb_geometry, par.wkb_geometry) "
                    "left join com_incident i "
                            "on parcel_id = parcelid "
                    "left join geography.dpd_pd_mem30 pd "
                            "on st_within(st_centroid(par.wkb_geometry), "
                                "pd.wkb_geometry)"
                    "group by parcelid, pd.name")
par_pd = pd.read_sql(sql_par_pd, engine_mph)

def date_split(df, date_field):
    df[df[date_field].isnull()] = 0.
    df['month'] = df[date_field].dt.month.astype(np.int) 
    df['year'] = df[date_field].dt.year.astype(np.int)
    return df

perm_pd = date_split(perm_pd, 'issued')

scale = lambda x: (x-x.min())/float((x.max() - x.min()))*100
#------------------------------------------------------------------------------
#------------------------------Visualizations----------------------------------
#------------------------------------------------------------------------------

#--------------------------------Heat Maps-------------------------------------

def facet_heatmap(data, color, **kws):
    d = data.pivot('month', 'year', 'permit')
    sns.heatmap(d, cmap=color, **kws)


        
#set custom cmap to match purples in PRGn
pr_cmap=sns.light_palette(color="#40004b", as_cmap=True)

def add_missing(df):
    idx_cols = ["month", "year", "name"]
    idx = list(itertools.product(*[range(1,13),
                                   range(2002,2017),
                                   df.name.unique()]))
    df.set_index(idx_cols, inplace=True)
    df = df.reindex(idx)
    df.reset_index(inplace=True)
    df.permit.fillna(0, inplace=True)
    return df

#New Construction PD
new_cons = perm_pd[perm_pd.const_type == 'new']
grp_new = new_cons.groupby(['month', 'year', 'name'])\
            .agg({'permit':'count'})\
            .reset_index()

#grp_new = add_missing(grp_new)

with sns.plotting_context(font_scale=5.5):
    g = sns.FacetGrid(grp_new.sort_values('name'), col='name', 
            col_wrap=7, size=3, aspect=1, legend_out=True)

cbar_ax = g.fig.add_axes([1, .06, .01, .9])  # <-- Create a colorbar axes
g = g.map_dataframe(facet_heatmap, color='Greens', cbar_ax=cbar_ax, 
        vmin=0, vmax=150, cbar_kws={'label':'Number of Construction Permits'})
g.set_titles(col_template="{col_name}", fontweight='bold', fontsize=18)
g.set_axis_labels('year','month' )
g.savefig('./output/matrix_const_pd.png')
plt.close()
#Demolition PD
demo = perm_pd[perm_pd.const_type == 'demo']
grp_demo = demo.groupby(['month', 'year', 'name'])\
            .agg({'permit':'count'})\
            .reset_index()
with sns.plotting_context(font_scale=5.5):
    g = sns.FacetGrid(grp_demo.sort_values('name'), col='name', 
        col_wrap=7, size=3, aspect=1, legend_out=True)

cbar_ax = g.fig.add_axes([1, .06, .01, .9])  # <-- Create a colorbar axes
g = g.map_dataframe(facet_heatmap, color=pr_cmap, cbar_ax=cbar_ax, 
        vmin=0, vmax=20, cbar_kws={'label':'Number of Demolition Permits'})
g.set_titles(col_template="{col_name}", fontweight='bold', fontsize=18)
g.set_axis_labels('year','month' )
plt.tight_layout()
g.savefig('./output/matrix_demo_pd.png')
plt.close()

#Net Construction PD
net_pd = grp_new.merge(grp_demo, on=['month','year','name'], 
            suffixes=['const','demo'])
net_pd['scale_const'] = scale(net_pd.permitconst)
net_pd['scale_demo'] = scale(net_pd.permitdemo)
#permit column is actually net construction, but needs to be named permit
#to run correctly in fact_heatmap function
net_pd['permit'] = net_pd.scale_const - net_pd.scale_demo.abs()

net_pd = add_missing(net_pd)
net_pd.drop(['permitconst','permitdemo','scale_const','scale_demo'], axis=1,
            inplace=True)

#drop columns rename net_const to permit
with sns.plotting_context(font_scale=5.5):
    g = sns.FacetGrid(net_pd.sort_values('name'), col='name',
            col_wrap=7, size=3, aspect=1, legend_out=True)
cbar_ax = g.fig.add_axes([1, .06, .01, .9])
g = g.map_dataframe(facet_heatmap, color='PRGn', cbar_ax=cbar_ax,
        vmin=-50, vmax=50, cbar_kws={'label':'Construction minus Demolotion'})
g.set_titles(col_template="{col_name}", fontweight="bold", fontsize=18)
g.set_axis_labels("Year", "Month")
g.savefig('./output/matrix_net_pd.png')
plt.close()
#New Construction County
new = perm_pd[perm_pd.const_type == 'new']
county_new = new.groupby(['month','year']).agg({'permit':'count'}).reset_index()
pivot_new = county_new.pivot('month', 'year', 'permit')
sns.heatmap(pivot_new, cmap="Greens", vmax=750, 
        cbar_kws={"label":"Number of Construction Permits"})#'Blues')#cmap)
plt.title('New Construction, 2002 to 2016')
plt.savefig('./output/matrix_const.png')
plt.close()
#Demolitions County
demos = perm_pd[perm_pd.const_type == 'demo']
county_demo = demos.groupby(['month', 'year']).agg({'permit':'count'}).reset_index()
pivot_demo = county_demo.pivot('month', 'year', 'permit')
sns.heatmap(pivot_demo, cmap=pr_cmap,vmax=75,
        cbar_kws={"label":"Number of Demolition Permits"})#Reds')
plt.title('Demolitions, 2002 to 2016')
plt.savefig('./output/matrix_demo.png')
plt.close()
#Net Construction County
net_const = county_new.merge(county_demo, on=['month','year'], 
        suffixes=['const','demo'])
net_const['scale_const'] = scale(net_const.permitconst)
net_const['scale_demo'] = scale(net_const.permitdemo)
net_const['net_const'] = net_const.scale_const - net_const.scale_demo.abs()
net_const.drop(['permitconst','permitdemo','scale_const','scale_demo'], axis=1,
            inplace=True)
pivot_net = net_const.pivot('month', 'year', 'net_const')
sns.heatmap(pivot_net, cmap='PRGn',vmin=-80, vmax=80,
        cbar_kws={'label':'Construction minus Demolition, scaled -100 to 100'})
plt.title('Net Construction, 2002 to 2016')
plt.savefig('./output/matrix_net.png')
plt.close()

#------------------------------------------------------------------------------
#----------------------------Built Env Vars------------------------------------
#------------------------------------------------------------------------------

"""
Variables
    1. diversity of uses
    2. acres per capita greenspace
    3. population density
    4. miles per square mile greenway
    5. percent coverage building
    6. intersections per square mile
    7. percent commercial square footage
    8. average age commercial buildings
    9. percent developed parcels
    10. percent vacant parcels
    11. average age all structures
"""
