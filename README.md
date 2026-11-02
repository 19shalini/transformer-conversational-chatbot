# Transformer-Based Conversational Chatbot

A beginner-friendly encoder–decoder Transformer chatbot built with **Python, TensorFlow/Keras, and NLP**. The project uses the [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) to learn how to generate responses from pairs of dialogue lines.

> **Project status:** Prototype / learning project. Replace this note with verified results after running and evaluating your notebook.

## Project overview

The notebook demonstrates a simple conversational AI workflow:

1. Download the Cornell Movie-Dialogs Corpus.
2. Extract consecutive dialogue pairs from the movie lines and conversation IDs.
3. Clean text, filter examples, and split data into training, validation, and test sets.
4. Convert text to integer token sequences with Keras `TextVectorization`.
5. Build an encoder–decoder Transformer with learned positional embeddings, multi-head attention, masked decoder self-attention, and cross-attention.
6. Train using a padding-masked loss, evaluate on held-out examples, and generate responses one token at a time.

## Tech stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab or Jupyter Notebook

## Dataset

The [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) includes more than 300,000 dialogue utterances. This project forms *consecutive-line input/response pairs*, so the number of examples actually extracted and used for training depends on preprocessing and filtering.

The notebook downloads the dataset at runtime. **The dataset is not included in this repository.** Cite the original corpus if you use it in a report or publication:

Danescu-Niculescu-Mizil, C., & Lee, L. (2011). *Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs.* Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, ACL.

## Repository files

```text
Transformer-Conversational-Chatbot/
├── Transformer_Conversational_Chatbot.ipynb  # Add this separately
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

### Option A: Google Colab (easiest)

1. Open `Transformer_Conversational_Chatbot.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → GPU** if available.
3. Run the cells from top to bottom.
4. Allow the notebook to download and preprocess the dataset.
5. Train the Transformer, then run the response-generation cells.

You **do not** have to connect Colab to GitHub. You can download the `.ipynb` file from Colab and upload it to this repository manually.

### Option B: Local Jupyter

Install Python (a recent version compatible with TensorFlow), then run:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
pip install notebook
jupyter notebook
```

Open the notebook and run each cell in order. A GPU is optional, but training on a CPU may be slow.

## Model architecture

```text
User message
   ↓
Text cleaning and integer tokenization
   ↓
Encoder: token + positional embeddings → self-attention → feed-forward
   ↓
Decoder: shifted response tokens → causal self-attention
         → encoder-decoder cross-attention → feed-forward
   ↓
Vocabulary logits → predicted next tokens → generated response
```

## Training and evaluation

The example notebook starts with a subset of up to **20,000 filtered dialogue pairs** for a manageable prototype. It uses an 80% training, 10% validation, and 10% testing split, with padding-masked loss and token accuracy.

**Add your actual results after running the notebook:**

| Metric | Result |
|---|---|
| Training examples | To be measured |
| Validation loss | To be measured |
| Test loss | To be measured |
| Test masked token accuracy | To be measured |

### Chatbot examples

Add real input/output examples here after training. Generated responses may be short, repetitive, or inaccurate because this is a small model trained from scratch on movie dialogue.

## Limitations and next steps

- Movie conversations may contain informal, inappropriate, or outdated language; outputs should not be assumed safe or factual.
- A small model trained from scratch is not expected to match a production LLM.
- Test different vocabulary sizes, larger training subsets, more layers, dropout, and improved decoding.
- Add reproducible evaluation results and optional interactive UI once the core notebook works.

## Acknowledgements

Dataset: Cornell Movie-Dialogs Corpus. Transformer architecture: Vaswani et al., *Attention Is All You Need* (2017).
