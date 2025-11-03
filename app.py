import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# 🎤 Taylor's Version: título y presentación
st.title("📖 Taylor's RAG — Read & Analyze (Taylor’s Version)")
st.write("Versión de Python:", platform.python_version())

# Imagen (puedes mantener la original o poner una temática)
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar
with st.sidebar:
    st.subheader("✨ Taylor te ayuda a leer entre líneas")
    st.write("Imagina que estás revisando letras inéditas, notas de estudio o contratos de tu era favorita. "
             "Esta app convierte cualquier PDF en una charla con Taylor para entenderlo mejor.")

# Clave de API
ke = st.text_input('🔑 Ingresa tu clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Subida de PDF
pdf = st.file_uploader("📂 Sube el archivo PDF (por ejemplo, letras, notas o documentos de estudio)", type="pdf")

# Procesamiento
if pdf is not None and ke:
    try:
        # Extraer texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"📄 Texto extraído: {len(text)} caracteres (una nueva Era detectada)")
        
        # Dividir texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"🧩 Documento dividido en {len(chunks)} fragmentos listos para analizar")

        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = Chroma.from_texts(chunks, embeddings)
        
        # Interfaz de preguntas
        st.subheader("💬 Pregúntale a Taylor sobre el documento")
        user_question = st.text_area(" ", placeholder="Por ejemplo: ¿De qué trata esta sección? o ¿Cuál es el tono emocional del texto?")
        
        # Cuando el usuario pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Modelo actual
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            
            response = chain.run(input_documents=docs, question=user_question)
            
            # Mostrar respuesta
            st.markdown("### 💡 Respuesta (Taylor’s Insight):")
            st.markdown(response)
            st.caption("*(Basado en las letras, notas o PDFs que has compartido con Taylor’s AI.)*")
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("💌 Carga un archivo PDF para que Taylor te ayude a interpretarlo.")
