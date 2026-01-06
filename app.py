`import streamlit as st
import pandas as pd
import pickle
import os

# 1. Judul dan Identitas (Penting untuk penilaian demo)
st.title("UAS Bengkel Koding Data Science 2025/2026")
st.write("Nama: Hans Valentinus B")
st.write("NIM: A11.2022.14487")

# 2. Fungsi Load Model
@st.cache_resource
def load_model():
    # Pastikan Anda mengunggah file model.pkl hasil training di notebook ke GitHub
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    return None

model = load_model()

# 3. Input Data (Contoh fitur, sesuaikan dengan dataset Anda)
st.sidebar.header("Input Data Pasien/Nasabah/Objek")
def get_user_input():
    # Sesuaikan input ini dengan workflow yang Anda buat di notebook
    feature_1 = st.sidebar.number_input("Fitur A (Contoh: Umur)", 0, 100, 25)
    feature_2 = st.sidebar.number_input("Fitur B (Contoh: Gaji)", 0, 1000000, 50000)
    
    data = {
        'column_name_1': feature_1,
        'column_name_2': feature_2
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# 4. Tampilkan Input & Prediksi
st.subheader("Data Input yang Akan Diprediksi")
st.write(input_df)

if st.button("Jalankan Prediksi"):
    if model is not None:
        prediction = model.predict(input_df)
        st.success(f"Hasil Prediksi Model Anda adalah: {prediction[0]}")
    else:
        st.error("File model.pkl tidak ditemukan! Pastikan sudah diunggah ke GitHub.")`