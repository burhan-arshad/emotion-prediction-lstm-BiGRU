# Moodline — Emotion Detection Using BiGRU

Type a sentence and a **Bidirectional GRU** model predicts one of six emotion categories: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`. The interface shows the predicted label, the model's confidence, and the full six-way probability distribution.

**Author:** Burhan Arshad — Computer Science Student | AI & Machine Learning Developer
[Portfolio](https://burhan.cypherai.tech) · [GitHub](https://github.com/burhan-arshad)

> The model predicts emotion *categories from text*. It does not understand feelings, and its confidence is the model's output probability, not a guarantee. It has not been calibrated.

## Features
- FastAPI backend serving a trained Keras BiGRU (`/health`, `/predict`)
- Vanilla HTML/CSS/JS frontend served from the same app (no build step)
- Live health indicator (checking / ready / unavailable)
- Orb visual painted from the real six probabilities, sorted probability bars
- Input validation (1–2,000 characters), duplicate-submit protection, request timeout, clear error messages
- Sample sentences for each emotion, labeled as input only
- Accessible: labelled input, live regions, keyboard support, reduced-motion support

## Model
| Item | Value |
|---|---|
| Dataset | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face), official train / test splits |
| Classes | 6 (sadness, joy, love, anger, fear, surprise) |
| Tokenizer | Keras `Tokenizer`, `oov_token="<unk>"`, fit on training text |
| Sequence length | 50 (post-padding, post-truncation) |
| Architecture | Embedding(128, `mask_zero=True`) → BiGRU(128, return_sequences) → Dropout(0.3) → BiGRU(64) → Dropout(0.3) → Dense(6, softmax) |
| Loss / optimizer | sparse categorical cross-entropy / Adam |
| Training | up to 20 epochs, batch 32, balanced class weights, EarlyStopping (patience 3, restore best weights) |

The notebook also trains SimpleRNN, LSTM and GRU baselines for comparison. Run it to see the loss/accuracy for each; **no results are claimed here** until you add your own numbers.

> Note: the notebook uses the test split as `validation_data` for early stopping, so the reported test accuracy is slightly optimistic. For a stricter evaluation, use the dataset's separate `validation` split for early stopping and keep `test` untouched.

## Inference pipeline
1. User submits text → `POST /predict` with `{"text": "..."}`
2. Backend lowercases it, removes apostrophes, strips non-alphanumeric characters
3. Tokenizer converts text to integer sequences
4. Sequence is padded/truncated to length 50
5. BiGRU outputs six probabilities
6. Highest probability becomes `predicted_emotion`
7. Frontend renders the label, confidence and all six probabilities

## Project structure
```
.
├── main.py                  # FastAPI app
├── Artifacts/
│   ├── BiGRU_Model.keras    # trained model
│   └── tokenizer.pkl        # fitted tokenizer
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── notebook.ipynb           # training notebook
├── requirements.txt
└── README.md
```

> ⚠️ The notebook currently saves the model as `BiGRU_Modle.keras` (typo), but `main.py` loads `BiGRU_Model.keras`. Fix the filename in the notebook's save cell (or rename the file) so they match.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000. Interactive API docs: `/docs`.

## API
**GET `/health`** → `{"status": "Healthy", "model_loaded": true}`

**POST `/predict`**
```json
{ "text": "I feel terrified whenever I hear strange footsteps behind me." }
```
Response shape (values are illustrative, not real output):
```json
{
  "text": "...",
  "predicted_emotion": "fear",
  "confidence": 0.0,
  "all_probabilites": { "sadness": 0.0, "joy": 0.0, "love": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0 }
}
```
The field is spelled `all_probabilites` (sic) throughout the backend and frontend. Errors: `422` invalid input, `503` model not loaded.

## Deployment notes
- Deploy the FastAPI app on any host that can run TensorFlow (Render, Railway, Fly.io, a VPS). Use HTTPS.
- Because the frontend is served by the same app, it uses relative URLs and needs no CORS. If you host the frontend separately, change the fetch URLs in `script.js` and restrict CORS in `main.py` to your frontend origin instead of `"*"`.
- Never commit secrets. The app needs none.

## Limitations
- Trained on a specific English dataset of short, mostly first-person statements; it may generalise poorly to other styles, sarcasm, or mixed emotions.
- It always picks one of six labels, even for text that expresses none of them.
- Confidence scores are uncalibrated softmax outputs.

## License
Add a license of your choice (e.g. MIT).
