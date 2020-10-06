import pandas as pd
import numpy as np
import glob
from datetime import date


# lecture des anciens dataframes
path = r'/data/achraf.hamid/temp' # Dossier contenant les anciens csv
all_files = glob.glob(path + "/perimeter_*.csv") # * pour lire tout les csv

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
previous_df = pd.concat(li, axis=0, ignore_index=True, sort=False)


# ancienne date
min_date = previous_df.date.max()
today = date.today()
today_date = today.strftime("%Y-%m-%d")

# mettre en forme la date pour le nommage du csv à la fin
interval_date = str(min_date) + '_' + str(today_date)
interval_date = interval_date.replace("-","")

#encodage de la variable date
previous_df['date'] = pd.to_datetime(previous_df['date'])

# requetes
competitors_crawlinterne = '''
WITH
distinct_product_code AS (
    SELECT DISTINCT product_code, date
    from database.table
    WHERE stock > 0
    AND product_code IS NOT NULL
    AND date between '{}' AND '{}'
)

SELECT
    distinct_product_code.date,
    distinct_product_code.product_code,
    price,
    shipping,
    website,
    dispo,
    LOWER(seller) as seller,
    buybox,
    promo,
    seller_rating,
    product_rating,
    lag(price, 1, price) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date) as prix_jmoins1_crawl,
    case
        when lag(price, 2) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date) is null then lag(price, 1, price) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date)
        else lag(price, 2) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date)
    end as prix_jmoins2_crawl,
    case
        when lag(price, 3) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date) is null then (case
        when lag(price, 2) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date) is null then lag(price, 1, price) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date)
        else lag(price, 2) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date)
    end)
        else lag(price, 3) over(partition by distinct_product_code.product_code order by distinct_product_code.product_code, distinct_product_code.date)
    end as prix_jmoins3_crawl
FROM distinct_product_code
INNER JOIN database.table2 as table2
ON table2.date = distinct_product_code.date AND table2.product_code = distinct_product_code.product_code
WHERE
    price < 9998
    AND price > 0
    AND buybox = True
    AND lower(etat) IN ('état : neuf', 'neuf', 'produit neuf')
'''.format(min_date, today_date)
crawl = con.read_sql_query(sql=competitors_crawlinterne, con='presto')


products_prices_and_sells = '''
SELECT
    net_worth,
    product_code,
    product,
    turnover,
    rayon_1,
    rayon_2,
    rayon_3,
    rayon_4,
    marque,
    price,
    price_old,
    quantite,
    price_old_mp_minshipping,  
    score_visibility,
    date,

    lag(price, 1, price) over(partition by product order by product, date) as prix_jmoins1,
    lag(price, 2, price) over(partition by product order by product, date) as prix_jmoins2,
    lag(price, 3, price) over(partition by product order by product, date) as prix_jmoins3,
    lag(price, 4, price) over(partition by product order by product, date) as prix_jmoins4,
    lag(price, 5, price) over(partition by product order by product, date) as prix_jmoins5,
    lag(price, 6, price) over(partition by product order by product, date) as prix_jmoins6,
    lag(price, 7, price) over(partition by product order by product, date) as prix_jmoins7,

    GREATEST
    (
    (lag(price, 1, price) over(partition by product order by product, date)),
    (lag(price, 2, price) over(partition by product order by product, date)),
    (lag(price, 3, price) over(partition by product order by product, date)),
    (lag(price, 4, price) over(partition by product order by product, date)),
    (lag(price, 5, price) over(partition by product order by product, date)),
    (lag(price, 6, price) over(partition by product order by product, date)),
    (lag(price, 7, price) over(partition by product order by product, date))
    ) as prix_max7j,

    LEAST
    (
    (lag(price, 1, price) over(partition by product order by product, date)),
    (lag(price, 2, price) over(partition by product order by product, date)),
    (lag(price, 3, price) over(partition by product order by product, date)),
    (lag(price, 4, price) over(partition by product order by product, date)),
    (lag(price, 5, price) over(partition by product order by product, date)),
    (lag(price, 6, price) over(partition by product order by product, date)),
    (lag(price, 7, price) over(partition by product order by product, date))
    ) as prix_min7j


FROM (
    SELECT
        net_worth,
        product_code,
        product,
        etat_produit,
        boolean_bundle,
        quantite__mp_minshipping, 
        tunover_mp_minshipping,
        rayon_1,
        rayon_2,
        rayon_3,
        rayon_4,
        marque,
        CASE WHEN stock > 0 THEN 1 ELSE 0 END is_stock,
        CASE WHEN stock < quantite THEN 1 ELSE 0 END is_stock_almost_null,       
        price, 
        price_old,
        COALESCE(quantite, 0) quantite,       
        price_old_mp_minshipping,
        score_visibility,
        date,
        ROW_NUMBER() OVER (PARTITION BY product, date ORDER BY variable ASC) as row_n
    FROM database.table
    WHERE date between '{}' AND '{}'
    AND price < 9998
    AND price > 0
    AND boolean_bundle = False
    AND etat_produit = 1
    AND stock > 0
    AND product_code IS NOT NULL
    AND rayon_1 <> 'No pricing'
    AND rayon_2 <> 'No pricing'
    ) tmp
WHERE row_n = 1
'''.format(min_date, today_date)
prices = con.read_sql_query(sql=products_prices_and_sells, con='presto')




#On identifie les vendeurs avec plus de 400 ventes
N_LINES = 400

w = crawl[crawl['seller'].isin(crawl['seller'].value_counts()[crawl['seller'].value_counts()>N_LINES].index)].seller.tolist()
crawl['todrop'] = crawl["seller"].isin(w)
crawl = crawl.query('todrop == True') # On filtre sur les vendeurs avec plus 400 ventes

crawl.drop(columns=['todrop'], inplace=True)


# Generation prix hors promo
# les promos ne peuvent pas être positives, on annule les promos positives car ce sont des erreurs
crawl.loc[crawl['promo'] > 0 , 'promo'] = 0
# Logiquement les soldes sont en max autour de 70% , c'est rare au dela, on supprime les plus grandes valeurs
crawl.loc[crawl['promo'] < -0.7 , 'promo'] = 0
# On remplace le reste par 0
crawl['promo'] = crawl['promo'].fillna(0)

# On calcul le prix hors promo
if [crawl['promo'] < 0]:
    crawl['pricewithp'] = crawl['price'] / (1 + crawl['promo'])
else:
    crawl['pricewithp'] = crawl['price']


# creation du boolproduct_code buy_box
# On va utiliser : price_old_mp_minshipping et price

# Majorant du prix : permet d'imputer tout en garder l'information que ces valeurs sont manquantes
za = 100000
# On remplace donc par une valeur fixe pour ne pas sortir de la logique dans les anciens df
prices['price_old_mp_minshipping']=prices['price_old_mp_minshipping'].fillna(za)

def disposimplifier(dispo2):
    """
    Transforme la variable disponibilité en catégories dans les but de l'exploiter, les variables catégoriques sont ensuite
    remplacées par des valeurs numériques en fonction de la moyenne dans la distribution par rapport à la variable cible,
    la demande dans ce cas là.
    """
    a = "Il ne reste plus que 1 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    b = "Il ne reste plus que 2 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    c = "Il ne reste plus que 3 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    d = "Il ne reste plus que 4 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    e = "Il ne reste plus que 5 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    f = "Il ne reste plus que 6 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    g = "Il ne reste plus que 1 exemplaire(s) en stock."
    h = "Il ne reste plus que 7 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    i = "Il ne reste plus que 8 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    j = "Il ne reste plus que 9 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    k = "Il ne reste plus que 10 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    l = "Il ne reste plus que 11 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    m = "Il ne reste plus que 12 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement). "
    n = "Il ne reste plus que 13 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    o = "Il ne reste plus que 14 exemplaire(s) en stock (d'autres exemplaires sont en cours d'acheminement)."
    list = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o]

    if dispo2 == 'En stock.':
        return 0.90
    if dispo2 == 'en stock':
        return 0.90
    if dispo2 == 'En stock':
        return 0.90
    if dispo2 == 'InStock':
        return 0.90
    if dispo2 == 'En stock vendeur partenaire Suivi : gratuit':
        return 0.90
    if dispo2 == "Temporairement en rupture de stock. Commandez maintenant et nous vous livrerons cet article lorsqu'il sera disponible. Nous vous enverrons un e-mail avec une date d'estimation de livraison dès que nous aurons plus d'informations. Cet article ne vous sera facturé qu'au moment de son expédition.":
        return 0.20

    if dispo2 == 'Habituellement expédié sous 2 à 3 jours.':
        return 0.70
    if dispo2 == 'Habituellement expédié sous 3 à 4 jours.':
        return 0.70
    if dispo2 == 'Habituellement expédié sous 1 à 2 mois.':
        return 0.30
    if dispo2 == 'Habituellement expédié sous 1 à 3 mois.':
        return 0.30

    if dispo2 == 'Expédié depuis France.':
        return 0.50
    if dispo2 == "Frais d'expédition et politique pour les retours.":
        return 0.40
    if dispo2 == 'Voir les offres de ces vendeurs.':
        return 0.40

    if dispo2 == 'dernières pièces disponibles ':
        return 0.80

    for x in list:
        if x.startswith('Il ne reste plus que'):
            return 0.80

    else:
        return 0

# On applique la transformation sur une nouvelle variable dispo2, on va l'utiliser ensuite avec la note vendeur
crawl["dispo2"] = crawl.dispo.copy()
crawl['dispo2'] = crawl.dispo2.apply(disposimplifier)

# On applique la transformation sur une nouvelle variable dispo2, on va l'utiliser ensuite avec la note vendeur
crawl["dispo2"] = crawl.dispo.copy()
crawl['dispo2'] = crawl.dispo2.apply(disposimplifier)

# On garde pas les couples product_code-date dupliqués
# on garde les petites valeurs de la variable filtrage pour garder les bon vendeurs
crawl = crawl.sort_values(['seller_rating', 'dispo2'], ascending=[False, True]).drop_duplicates(subset=['product_code', 'date'], keep='first')

# Calcul du montant de la réduction
crawl["promoreduc"] = crawl['promo'] * crawl['price']

# Jointure entre le Df des prix-ventes et celui du Crawl
merged_crawl_price = prices.merge(crawl, on=['product_code','date'], how='left', suffixes = ('', '_crawl'))
# on réalise un left merge pour garder toutes les lignes dans clproduct_codeprices

# On enrichit avec les variables de pourcent de rabais affiché et on complète la variable algo_visibility_score
merged_crawl_price['price_old2'] = merged_crawl_price[["price_old", "price"]].max(axis=1)

merged_crawl_price['buyboxpossede'] = np.where(merged_crawl_price['price_old_mp_minshipping'] > (1.02 * merged_crawl_price['price']), 1, 0)

merged_crawl_price = merged_crawl_price.dropna(subset=["prix_max7j", "prix_min7j"])


# save new data
print('save new data')
#dataframe.to_csv(r'/data/achraf.hamid/temp/perimeter_'+ interval_date +'.csv', index = False, encoding='utf-8-sig')


