# Analysis
<img height="25" width="25" src="https://unpkg.com/simple-icons@v6/icons/python.svg"/> <img height="25" width="25" src="https://unpkg.com/simple-icons@v6/icons/pytorch.svg"/> <img height="25" width="25" src="https://unpkg.com/simple-icons@v6/icons/scikitlearn.svg"/>

In this section you can find the different analyses carried out.  

_**NOTE**: All kinds of approaches have been done in different time frames and with different sets of inputs. As the possible combinations of these 2 factors are infinite, all the conclusions are relative to the current moment, any of these study paths are 100% discarded._

The approaches studied to date are:

0. Baseline model

   As mentioned in the main objective of the project, our baseline model is to buy at the beginning and keep the coins, since the annual returns of our base case are exactly the same as the currency suffers.
   Since I am using BTC to study the different approaches the numbers to beat are:

   <img src="?raw=True"/>

2. [Price forecasting](Price_forecast)

    - **Kind of analysis**: Regression 
    - **Output**: Bitcoin price's change
    - **Conclusion**: Discarded visually. _'Persistence model'_ happens, just returning previous values.
   

      Deep learning models (LSTM):

<img src="https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Price_forecast/DL/Test_2022_4H_out1_inputsdim7_ldim2_nsteps7.png?raw=True"/>

2. [Volatility forecasting](Volatility)

   - **Kind of analysis**: Regression 
   - **Output**: Bitcoin's realized volatility defined in different ways.
   - **Conclusion**: Discarded visually.

3. [Price difference](Dif_price)

   - **Kind of analysis**: Supervised Classification 
   - **Output**: Bitcoin's percentage price change in a lag of time is bigger or smaller than a fixed value.
   - **Conclusion**: It seems like a good form of study, it should be explored in a greater variety of inputs. One set of input data is still under study pending the availability of more current's year data (test year).


      Extra Trees (Output -5%):

<img src="https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Dif_price/Study%202%20-%20ML%20evolution%20values/Output_ET_-5.PNG?raw=True"/>

      Extra Trees (Output -10%):

<img src="https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/Dif_price/Study%202%20-%20ML%20evolution%20values/Output_ET_-10.PNG?raw=True"/>


4. [Local max/min](MAXMIN%20Locals)

   - **Kind of analysis**: Regression 
   - **Output**: Bitcoin's difference of price to next relative maximum or minimum.
   - **Conclusion**: Discarded visually. Not learning at all, big MAB error.
   

      Output:

   <img src="https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/MAXMIN%20Locals/Error%20best%20model.PNG?raw=True"/>
   <img src="https://github.com/pvillegasmartin/cryptocurrencies/blob/main/Analysis/MAXMIN%20Locals/Regression%20output.png?raw=True"/>
   
   
5. [Sentiment analysis](Sentiment)