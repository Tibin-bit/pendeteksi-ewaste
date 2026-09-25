import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Pendeteksi E-Waste", page_icon="⚡")

st.title("⚡ Pendeteksi Sampah Elektronik (E-Waste)")
st.write("Arahkan kamera ke e-waste atau unggah foto untuk mendeteksi jenisnya.")

# Memuat model AI
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

classifier = load_model()

CATEGORY_MAP = {
    "mobile phone or smartphone": "Smartphone / HP",
    "power bank or portable charger": "Powerbank / Baterai Cadangan",
    "laptop or notebook computer": "Laptop / Notebook",
    "desktop computer or PC tower": "Komputer Desktop / PC",
    "printed circuit board or motherboard": "Papan Sirkuit / PCB",
    "electric cable or charger or adapter": "Kabel / Charger / Adaptor",
    "battery or household battery": "Baterai Bekas (AA/AAA/Koin)",
    "washing machine": "Mesin Cuci",
    "refrigerator or fridge": "Kulkas / Lemari Es",
    "air conditioner unit": "AC / Pendingin Ruangan",
    "television or computer monitor": "TV / Monitor / Layar",
    "electric fan": "Kipas Angin",
    "kitchen blender or mixer": "Blender / Mixer / Peralatan Dapur",
    "light bulb or LED lamp": "Lampu / Bohlam",
    "headphones or earphones or speaker": "Headphone / Earphone / Speaker",
    "microwave oven": "Microwave / Oven Listrik",
    "electric power drill or power tool": "Bor Listrik / Perkakas Listrik",
    "solar panel": "Panel Surya",
    "printer or scanner": "Printer / Mesin Cetak"
}

# Input Gambar (Kamera atau Upload)
img_file = st.camera_input("Ambil foto langsung")
uploaded_file = st.file_uploader("Atau unggah file gambar", type=["jpg", "jpeg", "png"])

image = None
if img_file is not None:
    image = Image.open(img_file)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption="Gambar yang dianalisis", use_container_width=True)
    with st.spinner("Menganalisis e-waste..."):
        results = classifier(
            image, 
            candidate_labels=list(CATEGORY_MAP.keys()),
            hypothesis_template="a photo of a {}"
        )
        
        st.subheader("Hasil Deteksi:")
        for res in results[:3]:
            label_id = CATEGORY_MAP[res['label']]
            score = res['score'] * 100
            st.write(f"**{label_id}**: {score:.1f}%")
            st.progress(res['score'])
