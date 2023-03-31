import streamlit as st
import warnings
import tensorflow as tf
from doc import doc

warnings.filterwarnings("ignore", message=".*label.*empty value.*")

if 'menu' not in st.session_state:
    st.session_state.menu = 'Poetry'

def callback():
    
    if st.session_state.menu == 'Brief':
        st.session_state.disabled = True
    else:
        st.session_state.disabled = False

def load_model():
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.saved_model.load('one_step', options=load_options)
    return model

def predict(text):
    
    states=None
    next_char = tf.constant([text])
    result = [next_char]
    model = load_model()
    
    for n in range(1000):
        next_char, states = model.generate_one_step(next_char,states=states)
        result.append(next_char)
    return result

def show_poetry_page():
    
    cols=st.columns([2,4,2])
    cols[1].image("icon.png",width=250)
    cols=st.columns([2,12,2])
    cols[1].title("|..... Poetry Generator .....|")
    st.write("")

    st.subheader("Enter the text to Generate :")
    text=st.text_area("",placeholder="Type or paste the text to be generated",  label_visibility="collapsed", height=100)
    st.write("")

    st.write("")
    cols = st.columns([2,1,2])
    ok = cols[1].button("Start")

    if ok:
        if len(text) !=0:
            output = predict(text)#translate(text, lang)
            output = tf.strings.join(output)[0].numpy().decode("utf-8")
            cols=st.columns([1,3,1])
            st.subheader(" Generated text :")
            st.write(output)
        else:
            st.warning("No text found. Try again !!")

col1,col2 = st.sidebar.columns([1,5])
col2.image("icon.png",width=160)
col2.write("")

col1,col2 = st.sidebar.columns([2,6])
col2.title(""" Poetry Generator """)      

col1,col2 = st.sidebar.columns([1,8])
col2.title("Menu")

ol1,col2 = st.sidebar.columns([2,10])
menu = col2.radio('', ['Poetry', 'Brief'], key='menu', horizontal=True, label_visibility="collapsed", on_change=callback())


if menu == 'Poetry':
    show_poetry_page()
elif menu == 'Breif':
    doc()
