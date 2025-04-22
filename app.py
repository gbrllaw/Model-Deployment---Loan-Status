import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('LoanPredictor.pkl')  # Pastikan file model .pkl sudah ada di folder yang sama

# Title of the app
st.title("Aplikasi Prediksi Persetujuan Pinjaman")

# Input fields
person_age = st.number_input("Usia", min_value=18, max_value=100, step=1)
person_gender = st.selectbox("Gender", ['Laki-laki', 'Perempuan'])
person_education = st.selectbox("Tingkat Pendidikan", ['S1', 'S2', 'S3', 'Diploma', 'Lainnya'])
person_income = st.number_input("Pendapatan Tahunan (Rp)", min_value=1000000, step=100000)
person_emp_exp = st.number_input("Pengalaman Bekerja (Tahun)", min_value=0, step=1)
person_home_ownership = st.selectbox("Status Kepemilikan Rumah", ['Milik Sendiri', 'Sewa', 'Orang Tua'])
loan_amnt = st.number_input("Jumlah Pinjaman (Rp)", min_value=1000, step=1000)
loan_intent = st.selectbox("Tujuan Pinjaman", ['Pendidikan', 'Rumah', 'Kendaraan', 'Lainnya'])
loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", min_value=0.0, step=0.1)
loan_percent_income = st.number_input("Pinjaman sebagai Persentase Pendapatan (%)", min_value=1.0, step=0.1)
cb_person_cred_hist_length = st.number_input("Durasi Kredit (Tahun)", min_value=0, step=1)
credit_score = st.slider("Skor Kredit", min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.selectbox("Tunggakan Pinjaman Sebelumnya", ['Ya', 'Tidak'])

# Convert categorical to numerical
gender_val = 1 if person_gender == 'Laki-laki' else 0
education_val = {'S1': 1, 'S2': 2, 'S3': 3, 'Diploma': 4, 'Lainnya': 5}[person_education]
home_ownership_val = {'Milik Sendiri': 1, 'Sewa': 2, 'Orang Tua': 3}[person_home_ownership]
loan_intent_val = {'Pendidikan': 1, 'Rumah': 2, 'Kendaraan': 3, 'Lainnya': 4}[loan_intent]
prev_loan_defaults_val = 1 if previous_loan_defaults_on_file == 'Ya' else 0

# Button for prediction
if st.button("Prediksi"):
    # Prepare the data for prediction
    input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_gender': gender_val,
        'person_education': education_val,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': home_ownership_val,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent_val,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': prev_loan_defaults_val
    }])

    # Make prediction
    prediction = model.predict(input_data)

    # Show result
    if prediction[0] == 1:
        st.success("Pinjaman Disetujui ✅")
    else:
        st.error("Pinjaman Ditolak ❌")
