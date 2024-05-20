
import streamlit as st
import base64
import os
import tempfile
import torch
from streamlit_pdf_viewer import pdf_viewer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, T5Tokenizer, T5ForConditionalGeneration

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

model_checkpoint = "glazzova/ml_translation_model1"
translator = pipeline("translation", model=model_checkpoint)

def generation_pipeline():
    pipe_generation = pipeline('text2text-generation',
                               model=base_model,
                               tokenizer=tokenizer,
                               max_length=512,
                               do_sample=True,
                               temperature=0.7,
                               top_p=0.95,
                               repetition_penalty=1.15)

    return pipe_generation

def summarization_pipeline(filepath):
    input_text, input_length = file_preprocessing(filepath)
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length= input_length // 8,
        min_length=25)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result


def generated_text (prompt):
    pip_text_gen = generation_pipeline()
    pip_text_gen_prom = pip_text_gen(prompt)
    return pip_text_gen_prom

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts, len(final_texts)


def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def chatbot():
    st.title('Chatbot 🤖')
    container = st.container()
    prompt = container.chat_input('Ask me something')
    if prompt:
        with (container.chat_message('assistant')):
            english_text = translator(prompt)[0]['translation_text']
            answer = generated_text(english_text)
            answer_all = answer[0]['generated_text']
            translated_text = translator(answer_all)
            container.markdown(translated_text[0]['translation_text'])


def main():
    st.title("Составитель резюме PDF-файла с помощью ИИ 📃")
    # Initialize Streamlit
    st.sidebar.title("Обработка документов")
    uploaded_files = st.sidebar.file_uploader("Загружать файлы PDF", accept_multiple_files=True, type=('pdf'))

    if uploaded_files :
        loader = None
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension == ".pdf":
               loader = temp_file_path
               col1, col2 = st.columns([50, 50])
               with col1:
                  st.info("PDF фаил")
                  # binary_data = loader.getvalue()
                  # pdf_viewer(input=binary_data, width=700)
                  pdf_view = displayPDF(loader)
               with col2:
                   summaru = summarization_pipeline(loader)
                   st.info("Составитель резюме")
                   translator_summari = translator(summaru)
                   st.success(translator_summari[0]['translation_text'])
               with st.container():
                  chatbot()

    else:
        with st.container():
            chatbot()


if __name__ == "__main__":
    main()

