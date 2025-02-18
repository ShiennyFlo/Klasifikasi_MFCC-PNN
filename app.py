import streamlit as st
import numpy as np
import librosa as lb
import pickle
import os
import warnings

warnings.filterwarnings("ignore")  # Hide warnings

# Fungsi Pre-emphasize
def pre_emphasize(signal, pre_emphasis=0.97):
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# Fungsi untuk ekstraksi fitur dengan jumlah frame tetap
def get_features(path, target_mfcc_shape=(20, 259)):
    y, sr = lb.load(path, sr=None)
    # Pre-emphasize
    emphasized_audio = pre_emphasize(y)
    # **Menyesuaikan hop_length agar jumlah frame tetap 259**
    total_samples = len(emphasized_audio)
    hop_length = total_samples // (target_mfcc_shape[1] - 1)  # Hitung hop_length dinamis
    n_fft = 2 ** int(np.ceil(np.log2(hop_length * 2)))  # n_fft adalah kelipatan 2 terdekat
    win_length = n_fft  # Biasanya sama dengan n_fft
    # Mel Spectrogram dengan parameter dinamis
    mel_spec = lb.feature.melspectrogram(
        y=emphasized_audio, sr=sr, n_mels=40, fmax=sr//2, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    # Konversi ke dB
    mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
    # MFCC dengan ukuran tetap
    mfcc = lb.feature.mfcc(S=mel_spec_db, sr=sr, n_mfcc=target_mfcc_shape[0], dct_type=2)
    return mfcc, mel_spec_db

# Fungsi Gaussian kernel
def gaussian_kernel(x, y, sigma):
    diff = x - y
    exponent = -(np.sum(diff ** 2)) / (2 * sigma ** 2)
    return np.exp(exponent)

# Implementasi PNN class
class ProbabilisticNeuralNetwork:
    def __init__(self, sigma=100):
        self.sigma = sigma
        self.classes_ = None
        self.prototypes_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            self.prototypes_[cls] = X[y == cls]

    def predict(self, X):
        predictions = []
        for x in X:
            probs = {}
            for cls, prototypes in self.prototypes_.items():
                probs[cls] = np.sum([gaussian_kernel(x, p, self.sigma) for p in prototypes])
            predictions.append(max(probs, key=probs.get))
        return np.array(predictions)

def classify_new_audio(audio_path):
    # Muat model dan parameter
    with open("best_pnn_model.pkl", "rb") as f:
        best_prototypes = pickle.load(f)

    with open("params.pkl", "rb") as f:
        params = pickle.load(f)

    sigma = 100
    label_encoder = params["label_encoder"]
    scaler = params["scaler"]

    # Preprocessing audio baru
    mfcc, _ = get_features(audio_path)
    mfcc = np.array(mfcc)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.reshape(mfcc, (1, -1))  # (1, 5180)
    mfcc = scaler.transform(mfcc)  # Normalisasi
    mfcc = scaler.inverse_transform(mfcc)
    # Prediksi dengan PNN terbaik
    pnn = ProbabilisticNeuralNetwork(sigma=sigma)
    pnn.prototypes_ = best_prototypes
    y_pred = pnn.predict(mfcc)
    
    # Konversi ke label asli
    diagnosis = label_encoder.inverse_transform(y_pred)
    return diagnosis[0]

# Streamlit
st.set_page_config(page_title="Klasifikasi Penyakit Pernapasan")
st.title("Klasifikasi Penyakit Pernapasan berdasarkan Suara Paru-Paru")
st.write("Untuk hasil prediksi lebih baik, masukkan file audio pernapasan 1 siklus saja.")

uploaded_file = st.file_uploader("Pilih file", type=["wav"], accept_multiple_files=False)

if uploaded_file:
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Run"):
        hasil_klasifikasi = classify_new_audio(temp_audio_path)
        st.success(f"Hasil klasifikasi: {hasil_klasifikasi}")
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass
else:
    if st.button("Run"):
        st.warning("File audio belum dimasukkan")

st.write("Aplikasi ini akan mengklasifikasikan data audio yang diberikan ke dalam 8 kelas, yaitu:")
st.write("**Asthma, Bronchiectasis, Bronchiolitis, COPD, LRTI, Pneumonia, URTI, dan Healthy (Sehat).**")