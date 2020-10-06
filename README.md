# Nested_Logit_Price_Elasticity_of_Demand
Compute price elasticity of demand and cross elasticity of demand in time using nested logit and Panel OLS and instrumental variables<br>

## Articles scientifique source :<br>
<b>Estimating Price Elasticities in Differentiated Product Demand Models with Endogenous Characteristics </b>(Daniel A. Ackerberg, Gregory S. Crawford – 2007)<br>
<b>Models of Consumer Demand for Differentiated Products </b>(Timothy J. Richards and Celine Bonnet - 2016)<br>

## Description des fichiers :<br>
1. Data_prep.py : Ce code lit les .csv déjà contenu dans le dossier, en récupère la dernière date et lance les requêtes de ‘crawl’ et ‘kpi_pricing’, fais la préparation de données et stock le nouveau csv dans le dossier avec dans le nom la date min et max.<br>
2. Functions_nested_logit.py : Ce code contient les fonctions nécessaires pour le modèle.<br>
3. Compute_elasticities : Ce code fait appel aux fonctions Nested Logit, sélectionne les variables, rajoute la variable instrumentale et lance le calcul des élasticités.<br>

## Déroulement :<br>
Pour lancer le code n°1 il faut impérativement avoir un csv du nom (perimeter_...) avec une colonne contenant la colonne ‘dt’ sous la forme ‘YYYY-MM-JJ’.<br>
Il faut avoir le connecteur ‘Presto’, si ‘Hive’ changer le code des requêtes.<br>
Il faut ensuite indiquer l’emplacement du .csv dans les fichiers n°1 et n°3.<br>
Lancer le n°3 pour avoir les élasticités.<br>
Spécifier à la dernière ligne du n°3 où stocker les élasticités en .csv.<br>
