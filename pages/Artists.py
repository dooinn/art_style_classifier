import streamlit as st
import pandas as pd
import pickle

# Load the artist data and cosine similarity matrix
artists = pickle.load(open('notebook/artists_list.pkl', 'rb'))
cosine_sim = pickle.load(open('notebook/similarity.pkl', 'rb'))

# Function to get recommendations
def recommend(artist_name):
    artist_idx = artists.index[artists['name'] == artist_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[artist_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:4+1]]  # adjust number as needed
    recommended_artists = artists['name'].iloc[sim_indices].tolist()
    return recommended_artists

# Sidebar dropdown to select an artist, with an option that prompts user selection
artist_options = ['Select an artist'] + list(artists['name'].unique())
selected_artist = st.sidebar.selectbox('', artist_options)

if selected_artist == 'Select an artist':
    # Default view when no artist is selected
    st.title('Who is your favorite artist?')
    st.subheader('Select an artist from Renaissance to Modern, and see similar artists.')
    st.image('assets/gogh.png')
else:
    # Show the bio of the selected artist
    artist_info = artists[artists['name'] == selected_artist]
    st.title(f'Artist: {selected_artist}')
    st.write('**Biography:**', artist_info['tags'].iloc[0])  # Assuming 'tags' is a mistake and should be 'bio'

    # Show recommendations
    recommendations = recommend(selected_artist)
    st.write('**Recommended Artists:**')
    for artist in recommendations:
        st.write(artist)