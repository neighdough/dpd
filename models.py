import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from caeser import utils
from sklearn.ensemble import import RandomForestRegressor

params = utils.connection_properties('caeser-geo.memphis.edu', db='memphis_30')
engine = utils.connect(**params)
sel_const = ("select g.count, wwl.* from wwl_2016 wwl, "
        "(select geoid10, count(id) from permits_2002_2016 p "
        "join wwl_2016 w on st_within(p.wkb_geometry, w.wkb_geometry) " 
        "where const_type = 'new' "
        "group by geoid10) g "
        "where g.geoid10 = wwl.geoid10")
const = pd.read_sql(sel_const, engine)
sel_demo = ("select * from wwl_2016 wwl "
        "join (select geoid10, count(id) from permits_2002_2016 p "
        "where const_type = 'demo' "
        "join wwl_2016 w on st_within(p.wkb_geometry, w.wkb_geometry) "
        "group by geoid10) g "
        "on g.geoid10 = wwl.geoid10")
demo = pd.read_sql(sel, engine)



drop = ['tid', 'index', 'ogc_fid', 'statefp10', 'countyfp10', 'tractce10', 
        'name10', 'geoid10', 'namelsad10', 'mtfcc10', 'funcstat10', 
        'intptlat10', 'intptlon10', 'shape_length', 'shape_area', 'acreland', 
        'wwl_name10', 'wwl_acreland', 'wwl_id', 'wkb_geometry']
geoids_const = const['geoid10']
const.drop(drop, axis=1, inplace=True)
geoids_demo = demo['geoid10']
demo.drop(drop, axis=1, inplace=True)
rf_const = RandomForestRegressor(n_estimators=500)
X,Y = const[col_const], const[const.columns[-1]]
rfr_const = rf_const.fit(X,Y)
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

