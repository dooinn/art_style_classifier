import streamlit as st
import pandas as pd
import pickle
import requests


artists = pd.read_csv('data/artists.csv')
cosine_sim = pickle.load(open('artifacts/similarity.pkl', 'rb'))


def recommend(artist_name):
    artist_idx = artists.index[artists['name'] == artist_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[artist_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:5]] 
    recommended_artists = artists['name'].iloc[sim_indices].tolist()
    return recommended_artists


def get_artist_info(artist_name):
    artist_row = artists.loc[artists['name'] == artist_name].iloc[0]
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": artist_row['name'],
        "prop": "extracts|pageimages|info",
        "inprop": "url",
        "format": "json",
        "exintro": True,
        "explaintext": True,
        "pithumbsize": 500
    }
    response = requests.get(URL, params=params)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    if 'missing' in page:
        return None
    image_url = page.get('thumbnail', {}).get('source', 'No image available')
    info = {
        'title': page['title'],
        'extract': page['extract'],
        'image': image_url,
        'full_url': page.get('fullurl', '#')  
    }
    return info


def display_recommendations(artist_name):
    recommendations = recommend(artist_name)
    if recommendations:
        st.write('**Recommended Artists:**')
        cols = st.columns(len(recommendations))
        for idx, artist in enumerate(recommendations):
            with cols[idx]:
                artist_data = get_artist_info(artist)
                if artist_data and artist_data['image'] != 'No image available':
                    st.image(artist_data['image'], caption=artist, width=150)
                if st.button(f"More about {artist}", key=f"more_about_{idx}_{artist}"):
                    st.session_state['selected_artist'] = artist
                    st.experimental_rerun()


def show_artist_info(artist_name):
    artist_info = get_artist_info(artist_name)
    if artist_info:
        st.title(artist_name)
        cols = st.columns(2)  
        with cols[0]: 
            if artist_info['image'] != 'No image available':
                st.image(artist_info['image'], caption=artist_info['title'])
        
        with cols[1]:  
            truncated_extract = ' '.join(artist_info['extract'].split()[:100])
            if len(artist_info['extract'].split()) > 100:
                truncated_extract += '...'
            
            st.write('**Biography:**')
            st.write(truncated_extract)
            st.write(f"[Read more on Wikipedia]({artist_info['full_url']})")
    else:
        st.error("No information found for this artist.")
    
    display_recommendations(artist_name)


artist_list = ['Select an artist'] + sorted(artists['name'].unique())
selected_artist = st.sidebar.selectbox(
    "Select an artist",
    artist_list,
    index=artist_list.index(st.session_state.get('selected_artist', 'Select an artist')),
    key='artist_selector'
)

if 'selected_artist' not in st.session_state or selected_artist != 'Select an artist':
    st.session_state['selected_artist'] = selected_artist

if st.session_state['selected_artist'] != 'Select an artist':
    show_artist_info(st.session_state['selected_artist'])
else:
    st.title('WHO IS YOUR FAVORITE ARTIST?')
    st.subheader('Select an artist from Renaissance to Modern, and see similar artists.')
    st.image('assets/gogh.png')
