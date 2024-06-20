from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import textwrap
from langchain.schema import prompt
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch
from textblob import TextBlob

@staticmethod
def save_file_db(pdf_dir):
    loader = PyPDFLoader(pdf_dir)
    page = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256)
    all_splits = text_splitter.split_documents(page)

    # Use SentenceTransformer to create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([doc.page_content for doc in all_splits])

    # Use faiss for similarity search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, model, all_splits


class beautify_result:
    @staticmethod
    def process_llm_response(text, width=80):
        lines = text.split('\n')
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text

class prompt_process:
    @staticmethod
    def prompt_processor(prompt=None):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = """
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
            SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            return prompt_template

        instruction = """CONTEXT:\n\n {context}\n\nQuestion: {question}"""

        if prompt is None:
            prompt_template = get_prompt(instruction, DEFAULT_SYSTEM_PROMPT)
        else:
            prompt_template = get_prompt(instruction, prompt)
        prompt_temp = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt_temp}

        return chain_type_kwargs

class query_notes:
    @staticmethod
    def general_query(query, index, model, documents, length=1000):
        sys_prompt_company = """You're an expert in AI, ML, LLM and Deep Learning. You as a model understand each and every line of the LLM and Machine, Deep learning notes.
        Give the answer for each parameter in separate line.
        """

        prompt_class = prompt_process()
        chain_type_kwargs = prompt_class.prompt_processor(sys_prompt_company)

        # Encode the query
        query_embedding = model.encode([query])

        # Search in the faiss index
        D, I = index.search(query_embedding, 1)
        context = " ".join([documents[i].page_content for i in I[0]])

        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model_qa = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

        inputs = tokenizer.encode_plus(query, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model_qa(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Ensure answer end is within input_ids length
        if answer_end >= len(input_ids):
            answer_end = len(input_ids) - 1

        # Extract the answer tokens and convert to string
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        print(answer)
        # Post-process the answer to ensure completeness
        answer = query_notes.complete_sentence(answer, context)
        return answer



    @staticmethod
    def complete_sentence(answer, context):
    # If the answer seems incomplete, try to extend it
        if not answer.endswith(('.', '!', '?')):
            context_words = context.split()
            answer_words = answer.split()
            answer_len = len(answer_words)

            for i in range(answer_len, len(context_words)):
                if context_words[i].endswith(('.', '!', '?')):
                    answer_words.append(context_words[i])
                    break
                else:
                    answer_words.append(context_words[i])

            answer = ' '.join(answer_words)
        
        # Correct grammar and spelling
        blob = TextBlob(answer)
        corrected_answer = str(blob.correct())
        
        return corrected_answer

def main():
    st.title("PDF Question Answering System")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_path = f"./{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        vectorstore, model, all_splits = save_file_db(pdf_path)
        st.write(f"PDF loaded and processed into chunks.")

        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                result = query_notes.general_query(question, vectorstore, model, all_splits)
                st.write("Answer:")
                st.write(result)
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
