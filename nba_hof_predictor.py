#NBA Hall of Fame Predictor

import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
import pickle

#Load saved model
with open("hof_pred_gbc.saved", "rb") as gbc_file:
    gbc = pickle.load(gbc_file)

#Streamlit App Title
st.title("🏀 NBA Hall of Fame Predictor")

#User Input: Player URL
player_url = st.text_input("Paste the Basketball Reference URL of the player:")

#Only scrape and predict if the button is clicked
if st.button("Predict Hall of Fame Status"):
    if not player_url:
        st.warning("Please paste a valid Basketball Reference player URL.")
    else:
        #Set request headers
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }

        #Send request
        response = requests.get(player_url, headers=headers)

        if response.status_code != 200:
            st.error("Failed to fetch player data. Please check the URL.")
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            try:
                #Get player name
                name = soup.find('h1').find('span').text

                #Get player headshot
                img_tag = soup.select_one('.media-item img')
                headshot_url = img_tag['src'] if img_tag else None

                #Display player name and headshot
                st.markdown(f"## ⛹️‍♂️🏀 {name}")
                if headshot_url:
                    st.image(headshot_url, width=150)

                #Function to extract numeric stat from tooltip text
                def stat_finder(word):
                    element = soup.find('span', {'data-tip': word})
                    if element:
                        element = element.find_next('p').find_next('p').text.strip()
                        return float(element)
                    return 0.0

                #Career length
                career_length = int(soup.find('strong', string=lambda s: s and ('Experience:' in s or 'Career Length' in s)).next_sibling.text.strip().split()[0])
                games = int(stat_finder('Games'))
                ppg = float(stat_finder('Points'))
                rpg = float(stat_finder('Total Rebounds'))
                apg = float(stat_finder('Assists'))
                per = float(soup.find('strong', string= lambda s: s and 'PER' in s).find_next('p').find_next('p').text)
                fg = float(stat_finder('Field Goal Percentage'))
                ft = float(stat_finder('Free Throw Percentage'))
                win_shares = float(soup.find('span', string= lambda s: s and 'WS' in s).next_sibling.find_next('p').text)

                #Awards
                def extract_award(label, is_list=True):
                    if is_list:
                        find = soup.find('li', string=lambda s: s and label in s)
                    else:
                        find = soup.find('a', string=lambda s: s and label in s)
                    if find:
                        text = find.text.strip()
                        return int(text.split('x')[0]) if 'x' in text else 1
                    return 0

                all_stars = extract_award('All Star')
                all_nba = extract_award('All-NBA')
                all_defense = extract_award('All-Defensive', is_list=False)
                all_rookie = 1 if soup.find('li', string=lambda s: s and 'All-Rookie' in s) else 0
                roy = 1 if soup.find('li', {'data-tip': lambda x: x and 'ROY' in x}) else 0
                mvps = extract_award('MVP')
                chips = extract_award('NBA Champ', is_list=False)
                scoring_champ = extract_award('Scoring Champ', is_list=False)
                dpoys = extract_award('Def. POY', is_list=False)

                #Create feature array for model
                features = np.array([
                    career_length, games, ppg, rpg, apg, per, fg, ft, win_shares,
                    all_stars, all_nba, all_defense, all_rookie, mvps, chips, roy, dpoys, scoring_champ
                ])

                #Predict
                pred = gbc.predict([features])[0]
                proba = gbc.predict_proba([features])[0][1] * 100

                #Display results
                st.subheader(f"{name} is predicted to be: {'🏅 Hall of Famer' if pred == 1 else '❌ Not a Hall of Famer'}")
                st.subheader(f"🧠 Probability: {proba:.2f}%")
                
                #Display the features used for prediction
                st.markdown("### 🔢 Stats Used for Prediction")
                labels = [
                    "Career Length (Years)", "Games Played", "Points Per Game", "Rebounds Per Game", "Assists Per Game",
                    "PER", "Field Goal %", "Free Throw %", "Win Shares",
                    "All-Star Selections", "All-NBA Selections", "All-Defensive Teams", "All-Rookie Team",
                    "MVP Awards", "Championships", "Rookie of the Year", "Defensive POY", "Scoring Titles"
                ]

                for label, value in zip(labels, features):
                    st.write(f"**{label}**: {value}")

            except Exception as e:
                st.error("Something went wrong while parsing the player's data.")
                st.error(str(e))