import os
import warnings

# --- Sembunyikan peringatan dan progress bar (harus sebelum import transformers) ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging
from huggingface_hub import logging as hub_logging

hf_logging.set_verbosity_error()
hub_logging.set_verbosity_error()

# ---- Slide 5: klasifikasi sentimen ----
pipe = pipeline("text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english")
print(pipe("Hello world"))

# ---- Slide 7: terjemahan lewat pipeline ----
def translation(text):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
    return translator(text, max_length=40)[0]["translation_text"]

print(translation("I am a student"))

# ---- Slide 10-13: encoder-decoder secara manual ----
model_name = "Helsinki-NLP/opus-mt-en-id"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence = "I am a student"
inputs = tokenizer.encode(sentence, return_tensors="pt")   # slide 11
print(inputs)

outputs = model.generate(inputs, max_length=40)            # slide 12
print(outputs)

decoded = tokenizer.decode(outputs[0])                     # slide 13
print(decoded)

input("\nTekan Enter untuk menutup...")