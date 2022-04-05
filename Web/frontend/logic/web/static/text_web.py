github_links = {
    'Baseline': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis',
    'Ensemble principals': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis',
    'Price difference': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Dif_price',
    'Local points': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/MAXMIN%20Locals',
    'Sentiment': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Sentiment',
    'On chain': ''
}

images_links = {
    'Baseline': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/baseline-evolutions-return.png?raw=true',
        'Bitcoin evolution returns')]
    , 'Ensemble principals': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Price_forecast/DL/streamlit_persistence_lstm.png?raw=true',
        'Bitcoin price forecasting (LSTM)')]
    , 'Price difference': [(
        './static/diff_price.png',
        "At least 5% BTC's price decrease in the next 24h"),
    ]
    , 'Local points': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/MAXMIN%20Locals/streamlit_maxmins.png?raw=true',
        'Bitcoin relative max/min analysis')]
    , 'Sentiment': [(
        'https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Sentiment/btc_sentiment.png?raw=true',
        'BTC sentiment wordcloud')],
    'On chain': [(
        './static/blockchain.png',
        'Onchain data')]
}


text_strategies = {
    'Baseline': 'Buy and hold is an investment technique, you buy crypto currency for example and then keep hold of '
                'it for long durations of time. The market prices may go up and down, but you keep hold of your stock '
                'with the hopes that they will eventually gain profit.',
    'Ensemble principals': 'The ensemble principle is about how different features affect outputs, these features are '
                           'previously predicted with other models: price forecasting, trend, volatility',
    'Price difference': "Supervised Binary classification with output equal to 1 when Bitcoin's percentage price "
                        "change in a lag of time beats a threshold.",
    'Local points': "Using maximus and minimums locals helps you to indefinity when prices of bitcoin or other assets "
                    "are at their highest and lowest. You track this on a chart and helps identify maximum and "
                    "minimum values.",
    'Sentiment': 'This is where you try to track the market prices by what other people thinks about then, '
                 'or the general feel about certain crypto or stocks are. Pending analysis of 93 sentiments and '
                 'topics related to crypto market.',
    'On chain': 'Although it cannot be considered a strategy, this is where transaction information is stored. You '
                'can use on chain data to examine the transaction activity of bitcoin or other cryptocurrencies. '
}

title_strategies = {
    'Baseline': 'BUY AND HOLD',
    'Ensemble principals': 'Ensemble principal BTC features',
    'Price difference': "Price percentage change",
    'Local points': "Local Maximums and minimums",
    'Sentiment': 'Sentimental analysis',
    'On chain': 'On chain data'
}

description_strategies = {
    'Baseline': 'The returns presented by this simple strategy are related to the high risk it has. There is no '
                'guarantee of profit and you may invest in assets that have a history of being volatile.',
    'Ensemble principals': 'The non-stationary nature of assets means they are very hard to predict. There has been '
                           'high increases of the mean value in the last years. In general the models created follow '
                           'a general “Persistence model” where they generate an output which is equal to the last '
                           'previous value.',
    'Price difference': "Some models inside this approach have presented good returns. "
                        "Searching data-points with high volatility in a relative short amount of time generates "
                        "really unbalanced datasets, kind of challenging resolve them.",
    'Local points': "The purpose of this maximums and minimums locals’ approach is searching for patterns with "
                    "Bitcoin prices so that you know the best times to buy and the best times to sell, this maximises "
                    "profit. The challenge is working out how to calculate the maximums and minimums locals’ points, "
                    "because what can be considered as the min/max is locally relevant.",
    'Sentiment': "I'm learning more about causation, trying not to follow in the classical ''Correlation doesn't mean "
                 "causation'. ",
    'On chain': "This is a type of open information which can be very useful because it can help accurately predict "
                "prices of cryptocurrency’s and identify trends. "
}