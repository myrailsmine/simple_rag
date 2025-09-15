#!/usr/bin/env python3
"""
Enhanced PDF Chatbot with Hybrid Retrieval, Math/Table Formatting,
Vision Model as Fallback, Thinking Process, and Web UI
"""

import os, re, fitz, pdfplumber, json, base64, requests, warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Added for better chunking

# Flask for UI
from flask import Flask, render_template, request, jsonify, send_from_directory

warnings.filterwarnings('ignore')


# ============================
# Hybrid Retriever with FAISS and Reranking
# ============================
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss  # Added for vector DB

class HybridRetriever:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=8000, stop_words='english')
        self.embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")  # Upgraded embedder
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')  # Added reranker
        self.chunks, self.metadata = [], []
        self.tfidf_matrix = None
        self.embedding_matrix = None
        self.faiss_index = None
    
    def build_index(self, chunks, metadata):
        self.chunks, self.metadata = chunks, metadata
        self.tfidf_matrix = self.tfidf.fit_transform(chunks)
        self.embedding_matrix = self.embedder.encode(chunks, normalize_embeddings=True)
        dim = self.embedding_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embedding_matrix.astype('float32'))
    
    def retrieve(self, query, top_k=5):
        # TF-IDF retrieval
        tfidf_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec, self.tfidf_matrix)[0]
        
        # Semantic retrieval with FAISS
        emb = self.embedder.encode([query], normalize_embeddings=True).astype('float32')
        _, sem_indices = self.faiss_index.search(emb, top_k * 2)  # Retrieve more for blending
        sem_scores = np.dot(self.embedding_matrix[sem_indices[0]], emb[0].T)
        
        # Blend scores (0.4 TF-IDF + 0.6 semantic for better balance on technical docs)
        candidate_idx = np.unique(np.concatenate((np.argsort(tfidf_scores)[::-1][:top_k*2], sem_indices[0])))
        blended_scores = 0.4 * tfidf_scores[candidate_idx] + 0.6 * sem_scores[:len(candidate_idx)]  # Align lengths
        
        # Rerank top candidates
        pairs = [(query, self.chunks[i]) for i in candidate_idx]
        rerank_scores = self.reranker.predict(pairs)
        top_idx = candidate_idx[np.argsort(rerank_scores)[::-1][:top_k]]
        
        return [(self.chunks[i], self.metadata[i], blended_scores[j]) for j, i in enumerate(top_idx)]


# ============================
# Helpers for formatting
# ============================
def format_formula_for_context(formula: Dict) -> str:
    return f"Page {formula['page']}:\n$$ {formula['text']} $$"

def format_table_for_context(table: Dict) -> str:
    df = table['data']
    return f"Page {table['page']}:\n{df.to_markdown(index=False)}"

def build_context(results):
    """Assemble structured context for LLM"""
    context = ""
    for chunk, meta, score in results:
        if meta["type"] == "formula":
            context += "\n\n### FORMULA CONTEXT\n" + format_formula_for_context(meta)
        elif meta["type"] == "table":
            context += "\n\n### TABLE CONTEXT\n" + format_table_for_context(meta)
        else:
            context += f"\n\n### TEXT CONTEXT\n(Page {meta.get('page','?')})\n{chunk}"
        if meta.get("vision_fallback"):
            context += "\n⚠️ Extracted using vision OCR (may be less precise)"
    return context


# ============================
# Enhanced PDF Extractor
# ============================
class EnhancedPDFExtractor:
    def __init__(self):
        self.extracted_content = {
            'tables': [],
            'formulas': [],
            'text': '',
            'images': []
        }
        self.llm, self.vision_model = None, None
        self.connection_status = "disconnected"

    # -------------------
    # Connection setup
    # -------------------
    def setup_llm_connection(self, config: Dict) -> bool:
        try:
            self.llm = ChatOpenAI(
                base_url=config.get('base_url', "http://localhost:8123/v1"),
                api_key=config.get('api_key', "dummy"),
                model=config.get('model', "llama3"),
                temperature=0.3,
                max_tokens=2000
            )
            test_response = self.llm.invoke([
                SystemMessage(content="You are a PDF analysis expert. Respond 'Connected' if this works."),
                HumanMessage(content="Test connection")
            ])
            if test_response and "Connected" in test_response.content:
                self.connection_status = "connected"
                return True
        except Exception as e:
            print(f"LLM connection error: {e}")
        self.connection_status = "disconnected"
        return False

    def setup_vision_model(self, config: Dict) -> bool:
        self.vision_model = {
            'base_url': config.get('vision_base_url', config.get('base_url')),
            'api_key': config.get('vision_api_key', config.get('api_key')),
            'model': config.get('vision_model', "llava"),
            'enabled': True
        }
        return True

    # -------------------
    # Extraction
    # -------------------
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        tables_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    for t in tables:
                        if not t or len(t) < 2: continue
                        df = pd.DataFrame(t[1:], columns=t[0])
                        if not df.empty:
                            tables_data.append({
                                'page': page_num+1,
                                'data': df,
                                'type': 'table'
                            })
                except Exception as e:
                    print(f"Table error page {page_num+1}: {e}")
        self.extracted_content['tables'] = tables_data
        return tables_data

    def extract_text(self, pdf_path: str) -> str:
        text = []
        doc = fitz.open(pdf_path)
        for p in doc:
            page_text = p.get_text().strip()
            if page_text: text.append(page_text)
        doc.close()
        all_text = "\n".join(text)
        self.extracted_content['text'] = all_text
        return all_text

    def extract_formulas(self, pdf_path: str) -> List[Dict]:
        doc, formulas = fitz.open(pdf_path), []
        patterns = [r'[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=\n]+']
        for i, p in enumerate(doc):
            txt = p.get_text()
            for pat in patterns:
                for m in re.finditer(pat, txt):
                    formulas.append({'text': m.group().strip(), 'page': i+1, 'type': 'formula'})
        doc.close()
        self.extracted_content['formulas'] = formulas
        return formulas

    def extract_images_and_analyze(self, pdf_path: str) -> List[Dict]:
        """Vision fallback if page text nearly empty"""
        images_data, doc = [], fitz.open(pdf_path)
        for i, p in enumerate(doc):
            txt = p.get_text().strip()
            need_vision = (len(txt.split()) < 50)  # More aggressive threshold
            for idx, img in enumerate(p.get_images(full=True)):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < 200 or pix.height < 200: continue
                    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
                    info = {'page': i+1, 'image_id': f"img_{i}_{idx}", 'size': (pix.width, pix.height), 'type': 'image'}
                    if self.vision_model and self.vision_model['enabled'] and need_vision:
                        vision = self._analyze_image_with_vision(img_b64)
                        vision['vision_fallback'] = True
                        info.update(vision)
                    images_data.append(info)
                except Exception as e:
                    print(f"Vision error page {i+1}: {e}")
        doc.close()
        self.extracted_content['images'] = images_data
        return images_data

    def _analyze_image_with_vision(self, image_b64: str) -> Dict:
        try:
            vision_prompt = """
            Analyze this PDF page image. Extract:
            1. All mathematical formulas in LaTeX format (e.g., $$ \\sum_k WS_k^2 $$).
            2. All tables in Markdown format (e.g., | Column1 | Column2 | \\n|---|---|).
            3. Structured text sections with headings.
            Ignore headers/footers unless relevant. If no formulas/tables, return empty.
            Output as JSON: {"formulas": [{"latex": "...", "context": "surrounding text"}], "tables": [{"markdown": "...", "context": "surrounding text"}], "text": "clean text"}
            """
            resp = requests.post(
                f"{self.vision_model['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {self.vision_model['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": self.vision_model['model'],
                    "messages": [{"role":"user","content":[{"type":"text","text": vision_prompt},{"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}}]}],
                    "max_tokens": 500
                }, timeout=30
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                try:
                    return json.loads(content)  # Parse JSON output
                except:
                    return {"vision_raw": content}
            return {"vision_error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"vision_error": str(e)}


# ============================
# Chatbot
# ============================
class IntelligentChatbotWithLLM:
    def __init__(self, extractor: EnhancedPDFExtractor):
        self.extractor, self.retriever = extractor, HybridRetriever()

    def build_index_from_pdf(self, pdf_path: str):
        text = self.extractor.extract_text(pdf_path)
        formulas = self.extractor.extract_formulas(pdf_path)
        tables = self.extractor.extract_tables(pdf_path)
        images = self.extractor.extract_images_and_analyze(pdf_path)

        # Improved chunking: Semantic-aware splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
        text_chunks = splitter.split_text(text)
        
        # Create chunks with metadata
        chunks, meta = [], []
        for idx, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            meta.append({"type": "text", "id": f"text_{idx}"})
        
        for f in formulas:
            chunks.append(f["text"])
            f['id'] = f"formula_{len(chunks)}"
            meta.append(f)
        
        for t in tables:
            chunks.append(t["data"].to_string())
            t['id'] = f"table_{len(chunks)}"
            meta.append(t)
        
        for img in images:
            if "formulas" in img:
                for form in img["formulas"]:
                    chunks.append(form["latex"])
                    meta.append({"type": "formula", "page": img["page"], "id": f"vision_formula_{len(chunks)}", "vision_fallback": True})
            if "tables" in img:
                for tab in img["tables"]:
                    chunks.append(tab["markdown"])
                    meta.append({"type": "table", "page": img["page"], "id": f"vision_table_{len(chunks)}", "vision_fallback": True})
            if "text" in img:
                vision_chunks = splitter.split_text(img["text"])  # Split vision text too
                for vchunk in vision_chunks:
                    chunks.append(vchunk)
                    meta.append({"type": "text", "page": img["page"], "id": f"vision_text_{len(chunks)}", "vision_fallback": True})

        self.retriever.build_index(chunks, meta)

    def ask(self, query: str) -> Dict[str, Any]:
        # Step 1: Query Rewriting
        thinking_steps = []
        thinking_steps.append("Step 1: Rewriting query for optimal retrieval...")
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite this query in 3 variations for better retrieval in a regulatory PDF context: {query}. Focus on key terms like 'GIRR', 'delta', 'formulas'."
        )
        rewrite_chain = LLMChain(llm=self.extractor.llm, prompt=rewrite_prompt)
        variants = rewrite_chain.run(query=query).split("\n")[:3]  # Get top 3 variants
        thinking_steps.append(f"Generated variants: {', '.join(variants)}")
        
        # Step 2: Retrieval
        thinking_steps.append("Step 2: Retrieving relevant chunks from PDF...")
        all_results = []
        for v in [query] + variants:
            all_results.extend(self.retriever.retrieve(v, top_k=3))  # Retrieve per variant
        
        # Dedupe by ID
        unique_results = {m['id']: (c, m, s) for c, m, s in all_results}
        results = list(unique_results.values())[:5]  # Top 5 unique
        thinking_steps.append(f"Retrieved {len(results)} unique chunks.")
        
        context = build_context(results)
        
        # Step 3: Generation
        thinking_steps.append("Step 3: Generating response using LLM...")
        # Enhanced prompt with few-shot example for structured output
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a regulatory risk expert.
When answering:
- Start with overview.
- Use numbered steps (e.g., Step 1: Identify Risk Factors).
- Render formulas in LaTeX ($$...$$).
- Use Markdown tables.
- Cite sources inline as [Page X, ID Y] after claims.
- Structure answers as numbered steps (1. Definitions, 2. Formulas, 3. Risk Weights, 4. Correlations, 5. Aggregation)
- If context includes ⚠️, warn that it came from OCR and may be less precise.
Few-shot example: For 'explain GIRR', output like: ### Overview... Step 1: ... $$ formula $$ ... | Tenor | Weight | ... [Page 7, text_5]"""
            ),
            HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        ])
        chain = LLMChain(llm=self.extractor.llm, prompt=prompt)
        
        output = chain.run(context=context, query=query)
        
        # Step 4: Self-Reflection
        thinking_steps.append("Step 4: Refining response for accuracy...")
        reflect_prompt = ChatPromptTemplate.from_template("Refine this answer for accuracy, structure, and completeness: {output}")
        reflect_chain = LLMChain(llm=self.extractor.llm, prompt=reflect_prompt)
        refined_output = reflect_chain.run(output=output)
        
        thinking_steps.append("Step 5: Response ready.")
        
        return {
            "thinking_steps": thinking_steps,
            "final_output": refined_output
        }


# ============================
# Web UI with Flask
# ============================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global chatbot instance
config = {}  # Load from env or file
extractor = EnhancedPDFExtractor()
extractor.setup_llm_connection(config)
extractor.setup_vision_model(config)
chatbot = IntelligentChatbotWithLLM(extractor)
pdf_path = None

@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')  # Assume templates/index.html exists

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_path
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF uploaded'})
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(pdf_path)
    chatbot.build_index_from_pdf(pdf_path)
    return jsonify({'message': 'PDF indexed successfully'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'})
    if pdf_path is None:
        return jsonify({'error': 'Upload PDF first'})
    response = chatbot.ask(query)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
