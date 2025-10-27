import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io

# --- Konfiguracja ---
# POPRAWKA: Zmiana ścieżki modelu na plik .keras
# PAMIĘTAJ: Musisz przesłać plik i nazwać go 'model.keras'
MODEL_PATH = "model.keras" 

# POPRAWKA: Dodanie ścieżki do pliku z etykietami
LABEL_PATH = "labels.csv" 

# Ścieżka do wag pozostaje bez zmian
DATA_PATH = "fruits_vegetables_weights_full (1).csv"

# POPRAWKA: Model Keras oczekuje rozmiaru 224x224
IMAGE_SIZE = (224, 224) 

# --- Funkcje pomocnicze (cache'owane dla wydajności) ---

@st.cache_resource
def load_model(model_path):
    """Ładuje pełny model Keras (.keras)."""
    try:
        # POPRAWKA: Używamy tf.keras.models.load_model zamiast TFLite Interpretera
        model = tf.keras.models.load_model(model_path)
        st.success("Model Keras załadowany pomyślnie!")
        return model
    except FileNotFoundError:
        st.error(f"Błąd: Nie znaleziono pliku modelu: '{model_path}'")
        st.error("Upewnij się, że przesłałeś swój plik .keras i nazwałeś go 'model.keras'.")
        return None
    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd podczas ładowania modelu Keras: {e}")
        return None

@st.cache_data
def load_labels(label_path):
    """Ładuje etykiety z pliku labels.csv."""
    try:
        df_labels = pd.read_csv(label_path)
        # Zakładamy, że plik ma nagłówek 'label'
        labels = df_labels['label'].tolist()
        st.success("Etykiety (labels.csv) załadowane pomyślnie!")
        return labels
    except FileNotFoundError:
        st.error(f"Błąd: Nie znaleziono pliku etykiet: '{label_path}'")
        return []
    except KeyError:
        st.error(f"Błąd: Plik '{label_path}' nie zawiera kolumny o nazwie 'label'.")
        return []
    except Exception as e:
        st.error(f"Błąd podczas ładowania pliku etykiet: {e}")
        return []

@st.cache_data
def load_weights_data(data_path):
    """Ładuje dane o wagach z pliku CSV."""
    try:
        df = pd.read_csv(data_path, sep=';')
        
        # Konwertuj kolumny wag na stringi
        df['min'] = df['min'].astype(str)
        df['max'] = df['max'].astype(str)
        df['avg'] = df['avg'].astype(str)

        # Tworzy słownik do szybkiego wyszukiwania wag
        weights_data = df.set_index('name').to_dict('index')
        st.success("Dane o wagach załadowane pomyślnie!")
        return weights_data
    except FileNotFoundError:
        st.error(f"Błąd: Nie znaleziono pliku CSV z wagami: '{data_path}'")
        return {}
    except Exception as e:
        st.error(f"Błąd podczas ładowania pliku CSV z wagami: {e}")
        return {}

def preprocess_image(image):
    """Przetwarza obrazek z PIL do formatu akceptowanego przez model."""
    img = image.convert('RGB')
    # POPRAWKA: Używamy poprawnego rozmiaru obrazu
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # KRYTYCZNA POPRAWKA: 
    # USUWAMY normalizację 'img_array / 255.0'.
    # Model .keras został zapisany z 'include_preprocessing=True' (zobacz notatnik, komórka 6),
    # co oznacza, że model SAM oczekuje pikseli w zakresie [0, 255] i sam je normalizuje.
    # Podawanie mu danych [0, 1] (jak poprzednio) powodowało błąd predykcji.
    
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_batch

def run_inference(model, img_batch):
    """Uruchamia inferencję na załadowanym modelu Keras."""
    # POPRAWKA: Używamy metody .predict() modelu Keras
    prediction = model.predict(img_batch)
    return prediction

# --- Główna aplikacja Streamlit ---

st.set_page_config(page_title="Innowacyjna Waga AI", layout="centered")
st.title("🍓 Innowacyjna Waga Owoców i Warzyw 🥦")

# Ładowanie modelu, etykiet i danych o wagach
model = load_model(MODEL_PATH)
labels = load_labels(LABEL_PATH)
weights_data = load_weights_data(DATA_PATH)

# Sprawdzenie, czy wszystko się poprawnie załadowało
if model is None or not labels or not weights_data:
    st.warning("Aplikacja nie może działać poprawnie bez załadowanego modelu, etykiet i danych o wagach.")
else:
    st.write("Zrób zdjęcie lub prześlij obraz owocu/warzywa, aby je zidentyfikować i poznać jego przybliżoną wagę.")

    tab1, tab2 = st.tabs(["Prześlij zdjęcie", "Zrób zdjęcie"])

    with tab1:
        uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "jpeg", "png"], key="uploader")
    
    with tab2:
        camera_file = st.camera_input("Użyj aparatu", key="camera")

    file_to_process = None
    caption = ""
    if uploaded_file is not None:
        file_to_process = uploaded_file
        caption = "Przesłane zdjęcie"
    elif camera_file is not None:
        file_to_process = camera_file
        caption = "Zdjęcie z aparatu"
    
    if file_to_process is not None:
        try:
            image = Image.open(file_to_process)
            st.image(image, caption=caption, use_container_width=True)
            st.write("Przetwarzanie...")
            
            # Krok 1: Przetwarzanie obrazu
            img_batch = preprocess_image(image)
            
            # Krok 2: Uruchomienie inferencji
            # POPRAWKA: Przekazujemy 'model' zamiast 'interpreter'
            prediction = run_inference(model, img_batch)
            
            # Krok 3: Post-processing wyników
            predicted_class_index = np.argmax(prediction)
            
            if 0 <= predicted_class_index < len(labels):
                predicted_label = labels[predicted_class_index]
                # Uwaga: .predict() zwraca tablicę w tablicy, stąd [0]
                confidence = np.max(prediction[0]) * 100
                
                st.success(f"Wykryto obiekt z {confidence:.2f}% pewnością.")
                
                # Krok 4: Wyświetlanie wyników
                st.header(f"1x {predicted_label}")
                
                # Krok 5: Wyszukiwanie i wyświetlanie wag
                item_weights = weights_data.get(predicted_label)
                
                if item_weights:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Min. waga", f"{item_weights['min']} g")
                    col2.metric("Śr. waga", f"{item_weights['avg']} g")
                    col3.metric("Max. waga", f"{item_weights['max']} g")
                else:
                    st.error(f"Niestety, nie znaleziono danych o wadze dla: {predicted_label}")
            else:
                st.error(f"Błąd krytyczny: Model zwrócił indeks ({predicted_class_index}), który jest poza zakresem listy etykiet ({len(labels)}).")

        except Exception as e:
            st.error(f"Wystąpił błąd podczas przetwarzania obrazu: {e}")

