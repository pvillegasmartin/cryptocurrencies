import streamlit as st


def streamlit_template():
    st.set_page_config(layout="wide")
    st.sidebar.markdown("<h1 style=' color: #948888;'>SECTIONS</h1>",
                        unsafe_allow_html=True)
    home = st.sidebar.checkbox("HOMEPAGE")
    if home:
        st.title("PABLO VILLEGAS MARTÍN")
        '''
            [![LinkedIn](https://img.shields.io/badge/My%20LinkedIn-informational?style=flat&logo=linkedin&logoColor=white&color=18548c)](https://www.linkedin.com/in/pablo-villegas-martin/)
            [![Github](https://img.shields.io/badge/My%20Github-informational?style=flat&logo=github&logoColor=white&color=212020)](https://github.com/pvillegasmartin)
        '''
        # --------------------------------------------- THIS LAST YEARS STORY ---------------------------------------------

        info_years = {
            2011: ('¡University time!', 'Hello'),
            2012: ('¡University time!', 'Hello'),
            2013: ('¡University time!', 'Hello'),
            2014: ('¡University time!', 'Hello'),
            2015: ('¡University time!', 'Hello'),
            2016: ('¡University time!', 'Hello'),
            2017: ('This year', 'What a year!'),
            2018: ('This year', 'What a year!'),
            2019: ('This year', 'What a year!'),
            2020: ('This year', 'What a year!'),
            2021: ('This year', 'What a year!'),
            2022: ('This year', 'What a year!')
        }

        past_years = st.sidebar.slider('Which year do you want to gossip?', min_value=1993, max_value=2022, step=1,
                                       value=2022)
        if past_years < 2011:
            st.write(f"Come on...maybe this is so far...jump to 2016, when everything starts!")

        else:
            st.markdown(f"<h2>{info_years[past_years][0]}</h2>", unsafe_allow_html=True)
            st.markdown(
                f"{info_years[past_years][1]}"
                , unsafe_allow_html=True)

        # ----------------------------------------------------------------------------------------------------------------------------------

    if not home:
        analysis_type = st.sidebar.radio("STRATEGIES",
                                         ('Baseline', 'Ensemble principals', 'Price difference', 'Local points',
                                          'Sentiment'),
                                         help='All them are in continuous revision, no one is 100% discarded')
        st.sidebar.write('\n')

        # --------------------------------------------- STRATEGIES TEMPLATE ---------------------------------------------

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
                         'Bitcoin evolution returns'),
                         (
                         'https://raw.githubusercontent.com/pvillegasmartin/cryptocurrencies/main/Analysis/baseline-return.PNG',
                         'Bitcoin returns')],
            'Ensemble principals': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis',
            'Price difference': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Dif_price',
            'Local points': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/MAXMIN%20Locals',
            'Sentiment': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Sentiment'
        }

        text_strategies = {
            'Baseline': '<b>BUY AND HOLD</b><br><br>The baseline model is to buy at the beginning and keep the coins, since the annual returns of our base case are exactly the same as the currency suffers.',
            'Ensemble principals': 'Ensemble in a final model the outputs of generated models for each principal feature:'
                                   '<li>Trend</li><li>Direction</li><li>Volatility</li>',
            'Price difference': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Dif_price',
            'Local points': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/MAXMIN%20Locals',
            'Sentiment': 'https://github.com/pvillegasmartin/cryptocurrencies/tree/main/Analysis/Sentiment'
        }

        st.title(analysis_type)
        f'''
            [![Github](https://img.shields.io/badge/Github%20repository-informational?style=flat&logo=github&logoColor=white&color=212020)]({github_links[analysis_type]})
        '''
        warning = st.expander(
            f"IMPORTANT: The information presented in this web is reduced with expose porpouse of the work done. If you want the detailed information and all the code you can reach the github web through the button on top.")
        with warning:
            st.markdown(
                f"<ul style=' color: #948888;'><br>"
                f"{text_strategies[analysis_type]}<br><br>",
                unsafe_allow_html=True)
            try:
                if len(images_links[analysis_type]) == 2:
                    col1, col2 = st.columns((3, 1))
                    col1.image(images_links[analysis_type][0][0], caption=images_links[analysis_type][0][1])
                    col2.image(images_links[analysis_type][1][0], caption=images_links[analysis_type][1][1])
                else:
                    col1, col2, col3 = st.columns((1, 1, 1))
                    col2.image(images_links[analysis_type][0][0], caption=images_links[analysis_type][0][1])
            except:
                pass
    # ----------------------------------------------------------------------------------------------------------------------------------


def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    return f'<a target="_blank" href="{link}">Link to book</a>'


if __name__ == "__main__":
    streamlit_template()
