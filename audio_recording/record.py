import streamlit as st
 
st.sidebar.title("Audio Recording App")
st.title("Record Your Audio")
st.write("Press the button to start recording and then stop when you're done.")
audio = st.audio_input("Record your audio")
 
if audio:
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio.getbuffer())
        st.write("Audio recorded and saved successfully!")
