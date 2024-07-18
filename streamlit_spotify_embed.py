import streamlit as st
import streamlit.components.v1 as components

def spotify_player(track_id):
    embed_link = f"https://open.spotify.com/embed/track/{track_id}"

    return components.html(
        f'<iframe src="{embed_link}" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
        height=400)

st.title("Spotify Player Örneği")

track_id = "53QF56cjZA9RTuuMZDrSA6"
    
spotify_player(track_id)