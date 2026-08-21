
Cotation de produits structures sur donnees de marche
Notes a capital protege, reverse convertible, autocall

Diptakshi Sen - projet personnel, 2026

Les parametres de cotation ne sont pas poses : ils sont extraits du marche.

    Spot                SPY (ETF S&P 500), dernier cours
    Taux sans risque    ^TNX, rendement du 10 ans americain
    Dividende           implicite, deduit de la parite call-put sur la chaine
                        d'options a la monnaie
    Volatilite          implicite, interpolee a la monnaie sur la chaine
                        d'options, avec la structure par terme observee
    Volatilite realisee calculee sur un an d'historique, pour comparaison

Produits traites
    1. Note a capital protege   ZC + call  -> on resout le taux de participation
    2. Reverse convertible      ZC - put   -> on resout le coupon
    3. Autocall (Athena)        payoff chemin-dependant, Monte-Carlo
                                -> on resout le coupon

Controles : Monte-Carlo vanille contre Black-Scholes, parite call-put, et
retour au pair apres resolution. Les grecques du structure sont obtenues par
choc et revalorisation.

Lancer :  pip install yfinance numpy python-docx
          python structured_products_market.py
