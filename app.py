import streamlit as st
import pandas as pd
import joblib


# Sarlavha
st.title("Qandli diabetni aniqlash tizimi")

# Modelni yuklash
model_path = 'decision_tree_model.pkl'
model = joblib.load(model_path)


# Foydalanuvchi ma'lumotlarini kiritish formasi
st.header("Foydalanuvchi ma'lumotlarini kiriting")

pregnancies = st.number_input("Homiladorliklar soni", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Qondagi glyukoza miqdori (mg/dL)", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Qon bosimi (mmHg)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Teri qalinligi (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin darajasi (IU/mL)", min_value=0, max_value=1000, value=85)
bmi = st.number_input("Tana massasi indeksi (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
diabetes_pedigree = st.number_input("Diabet tarixi ko'rsatkichi", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
age = st.number_input("Yoshingiz", min_value=0, max_value=120, value=30)

# Kiruvchi ma'lumotlarni birlashtirish
user_data = pd.DataFrame({
    'Homiladorliklar': [pregnancies],
    'Glyukoza': [glucose],
    'Qon bosimi': [blood_pressure],
    'Teri qalinligi': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'Diabet tarixi': [diabetes_pedigree],
    'Yosh': [age]
})

# Model yordamida bashorat qilish
if st.button("Diabetni aniqlash"):
    prediction = model.predict(user_data)[0]
    prediction_prob = model.predict_proba(user_data)[0]

    # Natijalarni ko'rsatish
    if prediction == 1:
        st.error("Natija: Sizda diabet aniqlanish ehtimoli yuqori!")
    else:
        st.success("Natija: Sizda diabet aniqlanmadi.")

    st.write("Ehtimollar:")
    st.write(f"Diabet: {prediction_prob[1]:.4f}")
    st.write(f"Sog'lom: {prediction_prob[0]:.4f}")

    # Foydalanuvchi ma'lumotlarini chiqarish
    st.subheader("Foydalanuvchi kiritgan ma'lumotlar")
    st.write(user_data)

