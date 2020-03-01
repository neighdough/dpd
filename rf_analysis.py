import sys
sys.path.append('/home/nate/dropbox/dev')
from caeser import utils
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split, cross_val_score
from collections import OrderedDict

pd.set_option('display.width', 180)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 125)
os.chdir('/home/nate/dropbox-caeser/Data/DPD/memphis_30')
params = utils.connection_properties('caeser-geo.memphis.edu', db='memphis_30')
engine = utils.connect(**params)

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

min_max_scale = lambda x: (x-x.min())/(x.max() - x.min())
std_scale = lambda x: (x-x.mean())/float(x.std())
#-----------------------------------------------------------------------------
#--------------------------- Urban Index -------------------------------------
#-----------------------------------------------------------------------------
sql =("select geoid10, numpermit, numdemo, ninter/sqmiland inter_density,"
        "wwl_totpop/wwl_sqmiland popdensity, wwl_hhinc, "
        #"wwl_pct_wh,wwl_pct_bl, "
        "wwl_pct_bl + wwl_pct_ai + wwl_pct_as + "
        "wwl_pct_nh + wwl_pct_ot + wwl_pct_2m pct_nonwh, "
        "wwl_pct_pov_tot, wwl_pct_to14 + wwl_pct_15to19 pct_u19,"
        "wwl_pct_20to24, wwl_pct_25to34, wwl_pct_35to49, wwl_pct_50to66, "
        "wwl_pct_67up, wwl_hsng_density, wwl_pct_comm, wwl_age_comm, "
        "wwl_pct_dev, wwl_pct_vac, wwl_park_dist, wwl_park_pcap, wwl_gwy_sqmi, "
        "wwl_age_bldg, wwl_mdnhprice,wwl_mdngrrent, wwl_pct_afford, "
        "wwl_pct_hu_vcnt, wwl_affhsgreen, wwl_foreclose,wwl_pct_own, "
        "wwl_pct_rent, wwl_pct_mf, wwl_age_sf, wwl_mdn_yr_lived, "
        "wwl_strtsdw_pct, wwl_bic_index,"
        "wwl_b08303002 + wwl_b08303003 + wwl_b08303004 tt_less15,"
        "wwl_b08303005 + wwl_b08303006 + wwl_b08303007 tt_15to29,"
        "wwl_b08303008 + wwl_b08303009 + wwl_b08303010 + wwl_b08303011 "
        "+ wwl_b08303012 + wwl_b08303013 tt30more,"
        "wwl_b08301002 tm_caralone, wwl_b08301010 tm_transit, "
        "wwl_b08301018 tm_bicycle, wwl_b08301019 tm_walk, wwl_mmcnxpsmi, "
        "wwl_transit_access, wwl_bic_sqmi, wwl_rider_sqmi, wwl_vmt_per_hh_ami, "
        "wwl_walkscore, wwl_autos_per_hh_ami, wwl_pct_canopy, "
        "wwl_green_bldgs_sqmi, wwl_pct_chgprop, wwl_avg_hours, "
        "wwl_emp_ovrll_ndx, wwl_pct_labor_force, wwl_emp_ndx, wwl_pct_unemp, "
        "wwl_pct_commercial, wwl_pct_arts, wwl_pct_health, wwl_pct_other, "
        "wwl_pct_pubadmin, wwl_pct_util, wwl_pct_mining, wwl_pct_ag, "
        "wwl_pct_food, wwl_pct_retail, wwl_pct_wholesale, wwl_pct_manuf, "
        "wwl_pct_construction, wwl_pct_waste_mgmt, wwl_pct_ed, wwl_pct_info, "
        "wwl_pct_transport, wwl_pct_finance, wwl_pct_realestate, "
        "wwl_pct_prof_services, wwl_pct_mgmt,wwl_pct_lowinc_job, "
        "wwl_pct_b15003016 pct_no_dip, wwl_pct_b15003017 pct_dip, "
        "wwl_pct_b15003018 pct_ged, wwl_pct_b15003019 pct_uni_1yr, "
        "wwl_pct_b15003020 pct_uni_no_deg, wwl_pct_b15003021 pct_assoc, "
        "wwl_pct_b15003022 pct_bach, wwl_pct_b15003023 pct_mast, "
        "wwl_pct_b15003024 pct_prof_deg, wwl_pct_b15003025 pct_phd, "
        "wwl_elem_dist, wwl_middle_dist, wwl_high_dist, "
        "wwl_pvt_dist, wwl_chldcntr_dist, wwl_cmgrdn_dist, wwl_frmrmkt_dist, "
        "wwl_library_dist, wwl_commcenter_dist,wwl_pct_medicaid, "
        "wwl_bpinc_pcap, wwl_hosp_dist, wwl_pol_dist, wwl_fire_dist, "
        "wwl_os_sqmi, wwl_pct_imp, wwl_wetland_sqmi, wwl_brnfld_sqmi, "
        "wwl_pcat_10, wwl_mata_route_sqmi, wwl_mata_stop_sqmi "
    "from (select count(s.id) ninter, t.wkb_geometry, geoid "
            "from tiger_tract_2016 t, streets_carto_intersections s "
            "where st_intersects(s.wkb_geometry, t.wkb_geometry) "
            "group by geoid, t.wkb_geometry) bg, "
            "(select geoid, "
            "count(distinct case when const_type = 'new' "
                "then permit end) numpermit, "
            "count(distinct case when const_type = 'demo' "
                "then permit end) numdemo "
            "from permits_2002_2016 p, tiger_tract_2016 t "
            "where st_within(p.wkb_geometry, t.wkb_geometry) "
            "group by geoid) p, "
            "wwl_2016 "
    "where geoid10 = bg.geoid "
    "and geoid10 = p.geoid;") 

df = pd.read_sql(sql, engine)
#scaling function
#Net Construction PD
df['scale_const'] = min_max_scale(df.numpermit)
df['scale_demo'] = min_max_scale(df.numdemo)
#permit column is actually net construction, but needs to be named permit
#to run correctly in fact_heatmap function
df['net'] = df.scale_const - df.scale_demo
new_cols = df.columns.tolist()
#strip wwl out of column names
for i in range(len(new_cols)):
    if new_cols[i][:3] == 'wwl':
        new_cols[i] = new_cols[i][4:]
df.columns = new_cols

df.fillna(0, inplace=True)
x_vars = [col for col in df.columns if col not in 
            ['numpermit', 'numdemo', 'geoid10', 'wkb_geometry', 
             'scale_const', 'scale_demo', 'net']]
X = df[x_vars]
y_net = df.net
X_pos = df[df.net > 0][x_vars]
y_pos = df[df.net > 0]['net']
X_neg = df[df.net < 0][x_vars]
y_neg = df[df.net < 0]['net']

RANDOM_STATE = 123
#determinte number of trees in forest
ensemble_clfs = [
    ("RFR, max_features='sqrt'|red|-",
        RandomForestRegressor(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RFR, max_features='log2'|green|-",
        RandomForestRegressor(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RFR, max_features=None|blue|-",
        RandomForestRegressor(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))]
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 15
max_estimators = 350

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y_net)
        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    label, color, linestyle = label.split('|')
    plt.plot(xs, ys, label=label, color=color,
            linestyle=linestyle)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(bbox_to_anchor=(0, 1.1, 1., .102), loc="upper center", ncol=2)
plt.show()

"""
Model
"""

rfr = RandomForestRegressor(max_features='sqrt', warm_start=True,
        oob_score=True, random_state=RANDOM_STATE)

from scipy.stats import spearmanr, pearsonr
rfr_error = OrderedDict()
for i in range(min_estimators, max_estimators + 1):
    rfr.set_params(n_estimators=i)
    rfr.fit(X, y_net)
    oob_error = 1 - rfr.oob_score_
    y_pred = rfr.oob_prediction_
    sp = spearmanr(y_net, y_pred)
    pe = pearsonr(y_net, y_pred)
    feat_imp = rfr.feature_importances_
    rfr_error[i] = {'error':oob_error, 
                'spearman': sp, 
                'pearson': pe, 
                'feat_imp': feat_imp}
    print i, '\n\terror: ', oob_error, '\n\tspearman: ', sp.correlation 
    print '\tpearson: ', pe[0]
    print

"""---------------------------------------------------------------------------
-----------Feature Selection to compare variable importance-------------------
---------------------------------------------------------------------------"""

from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
svr = SVR(kernel='linear')
rfecv = RFECV(estimator=svr, n_jobs=-1)
rfecv.fit(X,y_net)


"""--------------------------------------------------------------------------
------------------------------- Plots ---------------------------------------
--------------------------------------------------------------------------"""
#scatter plot showing predictions
pca = PCA(n_components=1)
X_new = pca.fit_transform(X)
y_pos_idx = y_net[y_net >= 0].index
y_neg_idx = y_net[y_net < 0].index
plt.figure(figsize=(8,8))
plt.scatter(X_new[y_pos_idx], y_pred[y_pos_idx], c='Green', s=100)
plt.scatter(X_new[y_neg_idx], y_pred[y_neg_idx], c='Purple', s=100)
plt.ylabel('Net Construction', fontsize=20)


#Feature Importances
feat_imp = 100. * (feat_imp/feat_imp.max())
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + .5
f, ax = plt.subplots(figsize=(20,24))
sns.barplot(feat_imp[sorted_idx], X.columns[sorted_idx],
        label='Feature Importance', palette='BuGn')
#top 20 variables
f, ax = plt.subplots(figsize=(16,8))
top = sorted_idx[-20:]
sns.barplot(feat_imp[top], X.columns[top], 
        label='Top 20 Features', color='Green')
# plt.figure(figsize=(24,24))
# plt.barh(pos, feat_imp[sorted_idx], align='center')
# plt.yticks(pos, X.columns[sorted_idx])

#Pairplots
rows, cols = 11,10 
f, axes = plt.subplots(rows, cols, sharex=False, sharey=True, 
        tight_layout=True, figsize=(24,24))
var_pos = 0
for row in range(rows):
    for col in range(cols):
        if var_pos < len(x_vars):
#            df.plot.scatter(x=x_vars[var_pos], y='numpermit',
#                    ax=axes[row, col], color='blue')
            df[df.net < 0].plot.scatter(x=x_vars[var_pos], y='net',
                    ax=axes[row, col], color='Purple')
            df[df.net < 0].plot.scatter(x=x_vars[var_pos], y='net', marker='+',
                    ax=axes[row, col], color='red')
            df[df.net >= 0].plot.scatter(x=x_vars[var_pos], y='net',
                    ax=axes[row, col], color='Green')
            var_pos += 1
#kde plot net construction
for row in range(rows):
    for col in range(cols):
        if var_pos < len(x_vars):
            sns.kdeplot(df[x_vars[var_pos]], df['net'],
                    ax=axes[row, col], color='red', shade=True, 
                    kind='hex')
            var_pos += 1

#correlation matrix for all variables
corr = df[x_vars].corr()
fig, ax = plt.subplots(figsize=(24,24))
sns.heatmap(corr,cmap='coolwarm')

#Plot for R2 score and Spearman's R
x = rfr_error.keys()
y_error = [val['error'] for val in rfr_error.itervalues()]
y_sp = [val['spearman'].correlation for val in rfr_error.itervalues()]
y_pe = [val['pearson'][0] for val in rfr_error.itervalues()]
plt.figure(figsize=(12,8))
plt.subplot(311)
plt.plot(x, y_error, label="OOB Error", color='orange', linewidth=2.25)
plt.ylabel("Error", fontsize=16)
plt.yticks(fontsize=12) 
plt.tight_layout()
plt.title("Model Accuracy", fontsize=18) 
plt.subplot(312)
plt.plot(x, y_sp, label="Spearman's R", color='orange', linewidth=2.25)
#plt.xlabel("n_estimators")
plt.ylabel("R", fontsize=16)
plt.yticks(fontsize=12) 
plt.tight_layout(pad=1.75)
plt.title("Spearman's Rho", fontsize=18)
plt.subplot(313)
plt.plot(x, y_pe, color='orange', linewidth=2.25)
plt.ylabel("p", fontsize=16)
plt.yticks(fontsize=12) 
plt.xlabel("n_estimators")
plt.tight_layout(pad=1.75)
plt.title("Pearson's R", fontsize=18)

#Y actual vs Y predicted
x = rfr.oob_prediction_
m, b = np.polyfit(x, y_net, 1)
plt.subplots()
plt.figure(figsize=(12,8))
plt.scatter(rfr.oob_prediction_, y_net, color='orange', s=25)
plt.ylabel('Net Construction', fontsize=16)
plt.yticks(fontsize=12)
plt.xlabel('OOB Prediction', fontsize=16)
plt.xticks(fontsize=12)
plt.title('Y-Predicted vs Y-Actual', fontsize=18)
plt.plot(x, m*x+ b, '-', color='green')

"""---------------------------------------------------------------------------
                        Tree Visualizations
---------------------------------------------------------------------------"""
#Scatter Plot using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_red = pd.DataFrame().from_records(pca.fit_transform(X))
plt.figure()
plt.scatter(X_red, y_net)
plt.plot(sample.x, sample.predict, color='red')

#Tree visual using Graphviz
#TODO:
#alter color of tree to reflect extremity of positive or negative value
#https://pythonprogramminglanguage.com/decision-tree-visual-example/
export_graphviz(rfr.estimators_[0], feature_names=X.columns,filled=True, 
        rounded=True,out_file='./output/sample_tree_d4.dot', special_characters=True, 
        max_depth=4, rotate=True, leaves_parallel=True)
os.system('dot -Tpng ./output/sample_tree_d4.dot -o ./output/sample_tree_d4.jpg')

#plots based on:
#https://aysent.github.io/2015/11/08/random-forest-leaf-visualization.html
from sklearn.tree import _tree
def leaf_depths(tree, node_id = 0):
     '''
     tree.children_left and tree.children_right store ids
     of left and right chidren of a given node
     '''
     left_child = tree.children_left[node_id]
     right_child = tree.children_right[node_id]

     '''
     If a given node is terminal, 
     both left and right children are set to _tree.TREE_LEAF
     '''
     if left_child == _tree.TREE_LEAF:
         '''
         Set depth of terminal nodes to 0
         '''
         depths = np.array([0])

     else:
         '''
         Get depths of left and right children and
         increment them by 1
         '''
         left_depths = leaf_depths(tree, left_child) + 1
         right_depths = leaf_depths(tree, right_child) + 1
         depths = np.append(left_depths, right_depths)
     return depths

def leaf_samples(tree, node_id = 0):
    
     left_child = tree.children_left[node_id]
     right_child = tree.children_right[node_id]
     if left_child == _tree.TREE_LEAF:
         samples = np.array([tree.n_node_samples[node_id]])
     else:
         left_samples = leaf_samples(tree, left_child)
         right_samples = leaf_samples(tree, right_child)
         samples = np.append(left_samples, right_samples)
     return samples

def draw_tree(ensemble, tree_id=0, linewidth=2):
     plt.figure(figsize=(8,8))
     plt.subplot(211)
     tree = ensemble.estimators_[tree_id].tree_
     depths = leaf_depths(tree)
     plt.hist(depths, histtype='step', color='#9933ff', 
              bins=range(min(depths), max(depths)+1),
              linewidth=linewidth)
     plt.xlabel("Depth of leaf nodes (tree %s)" % tree_id)
     plt.subplot(212)
     samples = leaf_samples(tree)
     plt.hist(samples, histtype='step', color='#3399ff', 
              bins=range(min(samples), max(samples)+2),
              linewidth=linewidth)
     plt.xlabel("Number of samples in leaf nodes (tree %s)" % tree_id)
     plt.show()

def draw_ensemble(ensemble, linewidth=2):
     plt.figure(figsize=(8,8))
     plt.subplot(211)
     depths_all = np.array([], dtype=int)

     for x in ensemble.estimators_:
         tree = x.tree_
         depths = leaf_depths(tree)
         depths_all = np.append(depths_all, depths)
         plt.hist(depths, histtype='step', color='orange', 
                  bins=range(min(depths), max(depths)+1))

     plt.hist(depths_all, histtype='step', color='#9933ff', 
              bins=range(min(depths_all), max(depths_all)+1), 
              weights=np.ones(len(depths_all))/len(ensemble.estimators_), 
              linewidth=linewidth)

     plt.xlabel("Depth of leaf nodes")
     samples_all = np.array([], dtype=int)
     plt.subplot(212)

     for x in ensemble.estimators_:
         tree = x.tree_
         samples = leaf_samples(tree)
         samples_all = np.append(samples_all, samples)
         plt.hist(samples, histtype='step', color='#aaddff', 
                  bins=range(min(samples), max(samples)+2))

     plt.hist(samples_all, histtype='step', color='#3399ff', 
              bins=range(min(samples_all), max(samples_all)+1), 
              weights=np.ones(len(samples_all))/len(ensemble.estimators_), 
              linewidth=linewidth)
     plt.xlabel("Number of samples in leaf nodes")
     plt.show()
