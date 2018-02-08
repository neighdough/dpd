import pandas as pd
import geopandas as gpd
import seaborn as sns
from caeser import utils
import matplotlib.pyplot as plt

engine = utils.connect(**utils.connection_properties('caeser-geo.memphis.edu', 
                                db='memphis_30'))

years = ['2003', '2014', '2016']

pop = pd.read_sql("""select geoid, pop00, pop10 from nhgis_pop""", engine)
for yr in years:
    calc_acre = 'calc_acre_{}'.format(yr)
    sca_parcels = 'sca_parcels_{}'.format(yr)
    livunit = 'livunit_{}'.format(yr)
    #par = pd.read_csv('par_{}.csv'.format(yr))
    par = pd.read_sql("""select parcelid, livunit {0}, calc_acre {1}, 
            bgid from {2}""".format(livunit, calc_acre, sca_parcels), engine)

    #TODO check years of lodes data to make sure they match, need to skip
    #2010 and any year added <> 2003 | 2014
    emp_yr = '2014' if int(yr) >= 2010 else '2003'
    tot_emp = 'tot_emp_{}'.format(emp_yr)
    #emp = pd.read_csv('otm_lodes_{}.csv'.format(emp_yr))

    emp = pd.read_sql("""select substring(id,1,12) geoid, 
                            sum(c000) {0} from otm_lodes_{1} 
                            group by geoid""".format(tot_emp, emp_yr),
                            engine)

    #pop = pd.read_csv('nhgis_pop.csv')
    #TODO check columns for each parcel year to make sure they are the same
    #for each year
    #area_res_grp.rename(columns={'calc_acre':'res_acre_{}'.format(yr)}, inplace=True)
    #if tot_emp not in pop.columns:
    #    pop = pop.merge(emp, on='geoid', how='outer')
    bg = par.groupby('bgid').agg({livunit:'sum',calc_acre:'sum'}).reindex()
    bg['lu_netden_{}'.format(yr)] = bg[livunit] / bg[calc_acre]
    
    bg = bg.merge(emp, right_on='geoid', how='outer',left_index=True)
    bg['emp_netden_{}'.format(yr)] = bg[tot_emp] / bg[calc_acre]
    
    pop = pop.merge(bg, left_on='geoid')
    #bg = bg.merge(pop[['geoid','pop00','pop10']])
    pop_yr = 'pop10' if int(yr) >= 2010 else 'pop00'
    pop_netden = 'pop_netden_{}'.format(yr)
    pop[pop_netden] = pop[pop_yr] / pop[calc_acre]
    #bg['pop10_netden'] = bg.pop10 / bg.calc_acre
    print yr
print 'complete'

pct_change = lambda row: ((row['pop10_netden']/row['pop00_netden'])\
        /row['pop00_netden'])*100 if row['pop00_netden'] > 0 else None
bg['pct_chg_density'] = bg.apply(pct_change)



