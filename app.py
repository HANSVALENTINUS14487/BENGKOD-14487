import streamlit as st
import pandas as pd
import pickle
import os

# ===============================
# JUDUL & IDENTITAS
# ===============================
st.title("UAS Bengkel Koding Data Science 2025/2026")
st.write("Nama: Hans Valentinus B")
st.write("NIM: A11.2022.14487")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as file:
            return pickle.load(file)
    return None

model = load_model()

if model is None:
    st.error("❌ File model.pkl tidak ditemukan. Pastikan sudah di-upload ke GitHub.")
    st.sto
