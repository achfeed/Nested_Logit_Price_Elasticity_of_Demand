import pandas as pd
import glob
import functions_nested_logit
import statsmodels.api as sm

# lecture des anciens dataframes
path = r'/data/achraf.hamid/temp' # Dossier contenant les anciens csv
all_files = glob.glob(path + "/perimeter_*.csv") # * pour lire tout les csv, utiliser le dernier pour le test plus rapide(perimeter_202008)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

prices = pd.concat(li, axis=0, ignore_index=True, sort=False)


# Regression pour variable instrumental (selection de IV)
data_iv = prices[['net_worth','Stock_value']]
data_iv.fillna(method='ffill', inplace=True)
target = prices['price']

#Ajouter constante
data_iv = sm.add_constant(data_iv)

# Regression OLS
reg = sm.OLS(endog=target, exog=data_iv, missing='drop', hasconst=True)
results = reg.fit()
print(results.summary())
prices['IV'] = results.predict(data_iv)



X_hors_concu = ['number_of_visits','buyboxpossede','IV','prix_jmoins1','prix_jmoins2','prix_jmoins3','prix_jmoins4','prix_jmoins5','prix_jmoins6','prix_jmoins7','prix_max7j','prix_min7j']
X_concu = ['pricewithp','promoreduc','prix_jmoins1_crawl','prix_jmoins2_crawl','prix_jmoins3_crawl','nom_site']
X_tot = X_hors_concu + X_concu
used_var = ['product', 'date','quantite','price','rayon_1','rayon_2','rayon_3','rayon_4'] + X_tot

prices = prices[used_var]


dico_modeles = {}
dico_elast = {}

dico_modeles_r1_r2, dico_elast_r1_r2 = functions_nested_logit.get_all_modeles_and_elast_from_selected_data_and_nests(prices,'rayon_1','prp_NL','rp2',1,'quantite','product', 'date', 'price','rayon_2', X_hors_concu , var_marche_is_tempo = True, time_eff = True, entity_eff = True)
dico_modeles['dico_model_r1_r2'] = dico_modeles_r1_r2
dico_elast['dico_elast_r1_r2'] = dico_elast_r1_r2


dico_elast_r1_r2 = functions_nested_logit.remove_none_from_dict(dico_elast_r1_r2)


elasticities=[]
for p_id, p_info in dico_elast_r1_r2.items():
    for key in p_info:
        elasticities.append((p_info[['date','product','elasticite', 'cross_elasticite_in_group', 'cross_elasticite_other_groups']]))


frame = pd.concat(elasticities, axis=0, ignore_index=True)
frame=frame.drop_duplicates()


#frame.to_csv(r'/data/achraf.hamid/temp/elasticites.csv', index = False, encoding='utf-8-sig')