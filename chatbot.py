import os
import streamlit as st
import openai
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load OpenAI key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Local model for fallback
_flant5_model = None
_tokenizer = None

def load_local_model():
    global _flant5_model, _tokenizer
    if _flant5_model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _flant5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_chat_response(user_input, context=None):
    try:
        messages = [{"role": "system", "content": context}] if context else []
        messages.append({"role": "user", "content": user_input})

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        print("OpenAI API failed, using local model:", e)
        return generate_local_response(user_input, context)

def generate_local_response(prompt, context=None):
    load_local_model()
    final_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
    inputs = _tokenizer(final_prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _flant5_model.generate(**inputs, max_new_tokens=256)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)

def format_context(resume_text, matched_jobs=None):
    jobs_summary = "\n".join([f"- {j}" for j in matched_jobs]) if matched_jobs else ""
    return f"Resume:\n{resume_text}\n\nTop Matching Jobs:\n{jobs_summary}"
