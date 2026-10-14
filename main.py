from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "Artifacts" / "BiGRU_Model.keras"
tokenizer_path = BASE_DIR / "Artifacts" / "tokenizer.pkl"

max_length = 50
emotion_labels=["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


def preprocess_text(text: str)->str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text

class TextInput(BaseModel):
    text: str = Field(...,min_length=1, max_length=2000, example="I am feeling great today!", description="Input text for emotion prediction")

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence : float
    all_probabilites: dict[str, float]

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool

dl_model = {} 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading the model and tokenizer...')
    dl_model["BiGRU"] = load_model(model_path)                      #BiGRU Model
    with open(tokenizer_path, 'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print('Model are loaded successfully...')   

    yield

    dl_model.clear()

app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory="static"), name="static")

@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

@app.get('/health', response_model=HealthCheckResponse, summary="Health Check", description="Check the health of the API and model loading status")
def health_check():
    model_loaded = "BiGRU" in dl_model and "Tokenizer" in dl_model
    return HealthCheckResponse(status="Healthy", model_loaded=model_loaded)

@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    BiGRU_model     = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    #1. 
    cleaned_text = preprocess_text(text_input.text)

    #2. and 3. 
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    probabilites     = BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index = int(np.argmax(probabilites)) # 4
    all_probabilites =  {
        label: float(prob) for prob, label in zip(probabilites, emotion_labels)
          
    }

    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilites[top_emotion_index]), 
        all_probabilites = all_probabilites
    )
