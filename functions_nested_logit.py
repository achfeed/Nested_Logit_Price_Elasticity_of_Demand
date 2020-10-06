#imports
import pandas as pd
import numpy as np
from linearmodels import PanelOLS


# This function selects data to work on, gives title to elasticities that we will get in dictionnary then calls the functions that will compute PanelOLS
def get_all_modeles_and_elast_from_selected_data_and_nests(
    data, declining_data_on, nature_reg, nest, level_of_selection, var_of_ms,var_individuals, var_marche, var_price, var_nest,
    X , var_marche_is_tempo = True, time_eff = True, entity_eff = True):
    """Appelle les modèles de calcul des élasticités et les stocke dans les dictionnaires
    Prend en entrée les nests et les données liées aux produits
    """
    dico_model = {}
    dico_elast = {}
    for i in data[declining_data_on].value_counts().reset_index()['index']:
        print(i)
        deb_rp = 'rp' + str(level_of_selection) + str(i)
        name_mod = get_nomenclature('fit',nature_reg,deb_rp,nest)
        name_elast = get_nomenclature('elast',nature_reg,deb_rp,nest)
        df = data.loc[data[declining_data_on] == i]
        try:
            if nest == None :
                dico_model[name_mod], dico_elast[name_elast] = reg_demand_estimation_logit_with_elasticities(df, var_of_ms,var_individuals, var_marche, var_price, X , var_marche_is_tempo, time_eff, entity_eff)
            else:
                dico_model[name_mod], dico_elast[name_elast] = reg_demand_estimation_nested_logit_with_elasticities(df, var_of_ms,var_individuals, var_marche, var_price, var_nest, X , var_marche_is_tempo, time_eff, entity_eff)
        except:
            print("Matrice de covariance non inversible, le modèle n'y est pas adapté, Rayon : " + i)
    return [dico_model,dico_elast]



def get_nomenclature(nature,model,marche_1,nest):
    """Retourne les noms à attribuer aux dictionnaires des élasticités en fonction des nests"""
    if nest == None :
        name = nature + '_' + model + '_' + marche_1
    else:
        name = nature + '_' + model + '_' + marche_1 + '_' + nest
    return name


def reg_demand_estimation_logit_with_elasticities(data, var_of_ms, var_individuals, var_marche, var_price, X,
                                                  var_marche_is_tempo=True, time_eff=False, entity_eff=False):
    """Retourne le panel de ragresion avec les coefficients, et affiche les regressions à la fin, on pourrait changer ça si on
    veut pas de print de summary. X est le DataFrame input de la fonction principale
    """
    data_for_reg = add_Sjm_to_data(data, var_of_ms, var_individuals, var_marche, var_price, var_marche_is_tempo)
    data_for_reg = data_for_reg.set_index([var_individuals, var_marche])
    X = [var_price] + X
    mod = PanelOLS(data_for_reg['log_Si_S0'], data_for_reg[X], time_effects=time_eff, entity_effects=entity_eff,
                   drop_absorbed=True)
    mod = mod.fit()

    data_for_reg_wth_index = data_for_reg.reset_index()
    elast = get_price_elasticities_from_mod_and_data_for_reg(data_for_reg_wth_index, mod, var_individuals, var_marche,
                                                             var_price, var_marche_is_tempo)
    print(mod.summary)
    return ([mod, elast])


def add_Sjm_to_data(data, var_of_ms,var_individuals, var_marche, var_price, var_marche_is_tempo) :
    """Renvoie la table de données prices avec la variable Sjm adéquate avec le marché clusteriser par var_marche
    Les lignes avec des parts de marché nulles sont ici prises en compte en leur affectant le log de la part de marché minimale.
    Sim: Share of product in market
    Sjm: Share of other products in the market
    """
    log_Si_S0 = get_y_for_reg_lecture1_per_market(data, var_of_ms,var_individuals, var_marche)[0]
    log_Si_S0 = log_Si_S0.rename("log_Si_S0")
    data_for_reg = data.join(log_Si_S0, on=[var_individuals,var_marche])
    minu = data_for_reg.log_Si_S0.min()
    data_for_reg.log_Si_S0 = data_for_reg.log_Si_S0.fillna(minu - 1)
    if var_marche_is_tempo :
        data_for_reg[var_marche] = pd.to_datetime(data_for_reg[var_marche])
    return (data_for_reg)



def get_y_for_reg_lecture1_per_market(data, var_of_ms,var_individuals, var_marche):
    """Calcul Sjm, Sim et l'outside Function.
    data doit contenir les données agrégées au niveau des individus
    On choisit comme part de marché 0, le produit qui à la part de marché la plus élevée
    """
    # On extrait les parts de marchés
    Sim,Sjm = create_market_share_per_market(data,var_of_ms,var_individuals,var_marche)
    # On définit la part de marché 0 comme la part de marché maximale
    S0m = get_outside_option(data, var_of_ms,var_individuals, var_marche)
    # On calcule le ratio de parts de marché et on prend leur logarithme
    y = np.log(Sim/S0m)
    # On supprime les parts de marchés qui sont égales à moins l'infini
    list_inf = y.loc[(y == -np.inf)].index
    y = y.loc[~(y == -np.inf)]
    # On renvoie, le vecteur de ratio des parts de marchés,
    # la liste des indices où les parts de marchés sont égales à moins l'infini,
    #le sku du produit qui à la part de marché maximum et qui a été utilisé comme part de marché 0
    return [y,list_inf]


def create_market_share_per_market(data,var_of_ms,var_individuals,var_marche):
    """ Cette fonction, renvoie les parts de marché calculées:
        - sur la variable var_of_ms (par exemple les quantités)
        - au niveau des individus var_individuals
        - au sein d'un marché.
    """
    sum_per_product,sum_tot,sum_other_prod = create_series_for_market_share_per_market(data,var_of_ms,var_individuals,var_marche)
    Sjm = sum_other_prod / sum_tot
    Sim = sum_per_product / sum_tot
    return [Sim,Sjm]

# Compute sum_per_product, sum_tot, sum_other_prod ; these are sales aggregations that are used to compute market shares
def create_series_for_market_share_per_market(data,var_of_ms,var_individuals,var_marche):
    """
    Cette fonction, renvoie les séries qui permettent de calculer des parts de marchés par produits,
    où les parts de marchés sont calculées:
        - sur la variable var_of_ms (par exemple les quantités)
        - au niveau des individus var_individuals
        - au sein d'un marché.
    Par exemple :
        create_series_for_market_share_per_market(ex,'qte_promis_mp_j','sku','date_1')[0]
        renvoie la série du dénominateur des parts de marchés, où les parts de marchés sont caclulés :
            - en sommant les quantités
            - au niveau des sku
            - pour chaque marché défini par une valeur de date_1
    data peut déjà être agrégé au niveau des individus.
    S'il ne l'est pas avant, il faudra l'agréger avant d'ajouter les séries nouvelles au df.
    """
    sum_per_product = data.groupby([var_individuals,var_marche])[var_of_ms].sum()
    sum_tot = sum_per_product.groupby(var_marche).sum()
    sum_other_prod = sum_tot - sum_per_product
    return [sum_per_product,sum_tot,sum_other_prod]

# Outside Option
def get_outside_option(data,var_of_ms,var_individuals,var_marche):
    """Cette fonction doit renvoyer la valeur de l'outside option sur un marché.
    Cette valeur correspond à la taille de la population du marché nombre d'acheteurs potentiel moins le nombre de produits
    achetés sur le marché divisé par la taille du marché.
    """
    sum_tot_market = create_series_for_market_share_per_market(data,var_of_ms,var_individuals,var_marche)[1]
    taille_market = get_taille_market()
    return (taille_market - sum_tot_market) / (taille_market)


def get_taille_market():
    """ Renvoie la taille du marché de CDiscount en considérant:
        - l'âge du plus jeune acheteur, par défaut 18 ans
        - l'âge du plus viel acheteur, par déffaut 60 ans (plus vieux achète et moins vieux n'achètent pas)
        - la population correpondante en France
    """
    return 33128720

# ingroup market share
def add_log_Sjm_Sgjm_to_data(data, var_of_ms, var_individuals, var_marche, var_price, var_nest, var_marche_is_tempo):
    """
    Renvoie la table de données prices avec la variable Sgjm adéquate avec le marché clusteriser par var_nest.
    On entre dans cette fonction, la base qui est déjà passé dans add_Sjm_to_data, qui a déjà les parts de marché, on va ajouter les parts de marchés par sous_groupe

    Les lignes avec des parts de marché nulles sont ici prises en compte en leur affectant le log de la part de marché minimale.

    Un exemple : (avec ex un extrait de la base prices)
    df = prices.loc[prices.rayon_pricing_3 == 'Petit Déjeuner']
    df_Sjm = add_Sjm_to_data(df, 'qte_promis_j','sku', 'date_1', 'prix', var_marche_is_tempo = True)
    df_Sjm_Sgjm = add_log_Sjm_Sgjm_to_data(df_Sjm, 'qte_promis_j','sku', 'date_1', 'prix','rayon_pricing_4', var_marche_is_tempo = True)

    """
    log_Si_Sgj = get_Si_and_Sgi_nest(data, var_of_ms, var_individuals, var_marche, var_nest)[0]
    log_Si_Sgj = log_Si_Sgj.rename("log_Si_Sgj")
    # On fait la jointure sur quoi ? Sur les trois indexs :   var_individuals, var_marche, var_nest
    data_for_reg = data.merge(log_Si_Sgj, on=[var_individuals, var_marche, var_nest])
    minu = data_for_reg.log_Si_Sgj.min()
    data_for_reg.log_Si_Sgj = data_for_reg.log_Si_Sgj.fillna(minu - 1)
    if var_marche_is_tempo:
        data_for_reg[var_marche] = pd.to_datetime(data_for_reg[var_marche])
    return (data_for_reg)

# Group Sjm, Sim, ingroup product market share
def get_Si_and_Sgi_nest(data, var_of_ms,var_individuals, var_marche, var_nest):
    """data doit contenir les données agrégées au niveau des individus
    On choisit comme part de marché 0, le produit qui à la part de marché la plus élevée"""
    # On extrait les parts de marchés
    Sim,Sjm = create_market_share_per_market(data,var_of_ms,var_individuals,var_marche)
    S_gi = create_market_share_per_market(data,var_of_ms,var_nest,var_marche)[0]
    # On calcule le ratio de parts de marché et on prend leur logarithme
    y = np.log(Sim/S_gi)
    # On supprime les parts de marchés qui sont égales à moins l'infini
    list_inf = y.loc[(y == -np.inf)].index
    y = y.loc[~(y == -np.inf)]
    # On renvoie, le vecteur de ratio des parts de marchés,
    # la liste des indices où les parts de marchés sont égales à moins l'infini,
    #le sku du produit qui à la part de marché maximum et qui a été utilisé comme part de marché 0
    return [y,list_inf]

# Elasticity formula
def get_price_elasticities_from_mod_and_data_for_reg(data, mod, var_individuals, var_marche, var_price,
                                                     var_marche_is_tempo):
    # On récupère les paramètres estimés
    df = data.copy()

    # On prépare le calcul en ajoutant le paramètre \alpha à la base
    df['alpha'] = get_alpha(mod, var_price)

    # On réalise les calculs
    df['elasticite'] = - (1 - df['log_Si_S0']) * df.alpha * df[var_price]
    df['cross_elast'] = df['log_Si_S0'] * df.alpha * df[var_price]

    # On revoie les colonnes clés et les deux élasticités calculées
    return (df[[var_individuals, var_marche, var_price, 'log_Si_S0', 'elasticite', 'cross_elast']])



def get_alpha(mod, var_price):
    """Fonction pour récupérer \alpha, qui ici, vaut -\alpha selon la fonction d'élasticité"""
    fit1 = mod
    alpha = -fit1.params[var_price]
    return(alpha)

# Get model panel en présence de nests (Else de la fonction globale)
def reg_demand_estimation_nested_logit_with_elasticities(data, var_of_ms, var_individuals, var_marche, var_price,
                                                         var_nest, X, var_marche_is_tempo=True, time_eff=False,
                                                         entity_eff=False):
    """
    Renvoie la regression de panel avec les variables de regression de X (contient les caractéristiques et le prix). Il est possible de rajouter ici print mod.summary pour avoir le resultat des regressions
    """
    mod, elast = None, None
    data_for_reg = get_data_for_nested_reg(data, var_of_ms, var_individuals, var_marche, var_price, var_nest,
                                           var_marche_is_tempo=True)
    data_for_reg_wth_index = data_for_reg.copy()

    data_for_reg = data_for_reg.set_index([var_individuals, var_marche])
    X = [var_price] + ['log_Si_Sgj'] + X
    try:
        mod = PanelOLS(data_for_reg['log_Si_S0'], data_for_reg[X], time_effects=time_eff, entity_effects=entity_eff,
                       drop_absorbed=True)
        mod = mod.fit()
        elast = get_nested_price_elasticities_from_mod_and_data_for_reg(data_for_reg_wth_index, mod, var_of_ms,
                                                                        var_individuals, var_marche, var_price,
                                                                        var_marche_is_tempo)
    except:
        print('error')
    return ([mod, elast])


def get_data_for_nested_reg(data, var_of_ms,var_individuals, var_marche, var_price, var_nest, var_marche_is_tempo = True) :
    """Compile les fonctions de création des bases de données pour mettre des share dedans et avec le fait d'être nest"""
    data_with_share_from_m = add_Sjm_to_data(data, var_of_ms,var_individuals, var_marche, var_price, var_marche_is_tempo)
    data_complete = add_log_Sjm_Sgjm_to_data(data_with_share_from_m, var_of_ms,var_individuals, var_marche, var_price, var_nest, var_marche_is_tempo)
    return(data_complete)

# Elasticities with nests
def get_nested_price_elasticities_from_mod_and_data_for_reg(data, mod, var_of_ms,var_individuals, var_marche, var_price, var_nest, var_marche_is_tempo = True):
    """On récupère les paramètres estimés, calcul des élasticités et cross élasticités"""
    df = data.copy()
    df['alpha'] = get_alpha(mod,var_price)
    df['sigma'] = get_sigma(mod,var_nest)
    df['elasticite'] = - df.alpha*df[var_price]/(1 - df.sigma)*(1 - df.sigma*df.log_Si_Sgj - ( 1 - df.sigma)* df.log_Si_S0)
    df['cross_elasticite_other_groups'] = df['log_Si_S0']*df.alpha*df[var_price]
    df['cross_elasticite_in_group'] = df['log_Si_S0']*df.alpha*df[var_price] * (df.sigma*df.log_Si_Sgj + (1 - df.sigma)*df.log_Si_S0)
    return(df[[var_individuals,var_marche,'elasticite','cross_elasticite_in_group','cross_elasticite_other_groups']])


def get_sigma(mod, var_nest):
    sigma = mod.params['log_Si_Sgj']
    return(sigma)


def remove_none_from_dict(elast_dict):
    new_dict = {}
    for k, v in elast_dict.items():
        if isinstance(v, dict):
            v = remove_none_from_dict(v)
        if v is not None:
            new_dict[k] = v
    return new_dict or None



