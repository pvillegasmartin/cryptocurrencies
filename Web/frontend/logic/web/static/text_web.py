github_links = {
    'Baseline': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis',
    'Ensemble principals': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis',
    'Price difference': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Dif_price',
    'Local points': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/MAXMIN%20Locals',
    'Sentiment': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Sentiment'
}

images_links = {
    'Baseline': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/baseline-evolutions-return.png?raw=true',
        'Bitcoin evolution returns')]
        # ,(
        # 'https://raw.githubusercontent.com/pvillegasmartin/cryptocurrencies/main/Analysis/baseline-return.PNG',
        # 'Bitcoin returns')],
    , 'Ensemble principals': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Price_forecast/DL/streamlit_persistence_lstm.png?raw=true',
        'Bitcoin price forecasting (LSTM)')]
    , 'Price difference': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Dif_price/Study%202%20-%20ML%20evolution%20values/Output_ET_-5.PNG?raw=true',
        "At least 5% BTC's price decrease in the next 24h"),
        (
            'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/baseline-return.PNG?raw=true',
            "Baseline model returns"),
        (
            'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Dif_price/Study%202%20-%20ML%20evolution%20values/Output_ET_-10.PNG?raw=true',
            "At least 10% BTC's price decrease in the next 24h"),
        (
            'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/baseline-return.PNG?raw=true',
            "Baseline model returns")
    ]
    , 'Local points': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/MAXMIN%20Locals/streamlit_maxmins.png?raw=true',
        'Bitcoin relative max/min analysis')]
    , 'Sentiment': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Sentiment/btc_sentiment.png?raw=true',
        'BTC sentiment wordcloud')]
}


text_strategies = {
    'Baseline': 'The baseline model is to buy at the beginning and keep the coins, since the annual returns of our base case are exactly the same as the currency suffers.',
    'Ensemble principals': 'Ensemble in a final model the outputs of generated models for each principal feature:',
    'Price difference': "<b>Analysis type:</b> Supervised Binary classification<br>"
                        "<b>Output:</b> Bitcoin's percentage price change in a lag of time is bigger or smaller than a value.",
    'Local points': "Output: Bitcoin's percentage price change to next relative maximum or minimum.",
    'Sentiment': 'Pending analysis of 93 sentiments and topics related to crypto market.'
}

title_strategies = {
    'Baseline': 'BUY AND HOLD',
    'Ensemble principals': 'Ensemble principals',
    'Price difference': "<b>Analysis type:</b> Supervised Binary classification<br>"
                        "<b>Output:</b> Bitcoin's percentage price change in a lag of time is bigger or smaller than a value.",
    'Local points': "Maximums and minimums locals",
    'Sentiment': 'Pending analysis of 93 sentiments and topics related to crypto market.'
}