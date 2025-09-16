#!/usr/bin/env python3
"""
Enhanced PDF Chatbot with LLaMA Index, Llama 70B, Vision Fallback, and Web UI
"""

import os, re, fitz, pdfplumber, json, base64, warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
from langdetect import detect
from PIL import Image, ImageEnhance
from bitsandbytes import BitsAndBytesConfig
from flask import Flask, render_template, request, jsonify, send_from_directory

# LangChain and LLaMA Index
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

warnings.filterwarnings('ignore')

# ============================
# Enhanced PDF Extractor
# ============================
class EnhancedPDFExtractor:
    def __init__(self):
        self.extracted_content = {'tables': [], 'formulas': [], 'text': '', 'images': [], 'lang': 'en'}
        self.llm, self.vision_model = None, None
        self.connection_status = "disconnected"

    def setup_llm_connection(self, config: Dict) -> bool:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        try:
            self.llm = ChatOpenAI(
                base_url=config.get('base_url', "http://localhost:8123/v1"),
                api_key=config.get('api_key', "dummy"),
                model=config.get('model', "llama3"),
                temperature=0.3,
                max_tokens=2000,
                model_kwargs={"quantization_config": quantization_config}
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

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        tables_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    for t in tables or []:
                        if len(t) < 2: continue
                        df = pd.DataFrame(t[1:], columns=t[0])
                        if not df.empty:
                            tables_data.append({'page': page_num+1, 'data': df, 'type': 'table'})
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
        self.extracted_content['lang'] = detect(all_text[:1000]) if all_text else 'en'
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
        images_data, doc = [], fitz.open(pdf_path)
        for i, p in enumerate(doc):
            txt = p.get_text().strip()
            need_vision = len(txt.split()) < 50
            for idx, img in enumerate(p.get_images(full=True)):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < 200 or pix.height < 200: continue
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.tobytes("rgb"))
                    enhancer = ImageEnhance.Contrast(img)
                    img_b64 = base64.b64encode(enhancer.enhance(2.0).tobytes("png")).decode()
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
            Analyze this PDF page (could be any document). Extract:
            1. Math formulas in LaTeX (e.g., $$ \\sum_k WS_k^2 $$).
            2. Tables in Markdown.
            3. Headings, text sections, logos/images descriptions.
            Handle scanned/low-quality: Use OCR accurately. Output JSON: {"formulas": [...], "tables": [...], "text": "...", "images_desc": "Logo: Bank for International Settlements"}
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
                    return json.loads(content)
                except:
                    return {"vision_raw": content}
            return {"vision_error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"vision_error": str(e)}

# ============================
# Chatbot with LLaMA Index
# ============================
class IntelligentChatbotWithLLM:
    def __init__(self, extractor: EnhancedPDFExtractor):
        self.extractor = extractor
        self.retriever = LlamaIndexRetriever()

    def build_index_from_pdf(self, pdf_path: str):
        try:
            if isinstance(pdf_path, list):
                all_chunks, all_meta = [], []
                for path in pdf_path:
                    text = self.extractor.extract_text(path)
                    formulas = self.extractor.extract_formulas(path)
                    tables = self.extractor.extract_tables(path)
                    images = self.extractor.extract_images_and_analyze(path)

                    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
                    text_chunks = splitter.split_text(text)
                    
                    chunks, meta = [], []
                    for idx, chunk in enumerate(text_chunks):
                        chunks.append(chunk)
                        meta.append({"type": "text", "id": f"text_{idx}-{path}"})
                    
                    for f in formulas:
                        chunks.append(f["text"])
                        f['id'] = f"formula_{len(chunks)}-{path}"
                        meta.append(f)
                    
                    for t in tables:
                        chunks.append(t["data"].to_string())
                        t['id'] = f"table_{len(chunks)}-{path}"
                        meta.append(t)
                    
                    for img in images:
                        if "formulas" in img:
                            for form in img["formulas"]:
                                chunks.append(form["latex"])
                                meta.append({"type": "formula", "page": img["page"], "id": f"vision_formula_{len(chunks)}-{path}", "vision_fallback": True})
                        if "tables" in img:
                            for tab in img["tables"]:
                                chunks.append(tab["markdown"])
                                meta.append({"type": "table", "page": img["page"], "id": f"vision_table_{len(chunks)}-{path}", "vision_fallback": True})
                        if "text" in img:
                            vision_chunks = splitter.split_text(img["text"])
                            for vchunk in vision_chunks:
                                chunks.append(vchunk)
                                meta.append({"type": "text", "page": img["page"], "id": f"vision_text_{len(chunks)}-{path}", "vision_fallback": True})
                    all_chunks.extend(chunks)
                    all_meta.extend(meta)
                self.retriever.build_index(all_chunks, all_meta)
            else:
                text = self.extractor.extract_text(pdf_path)
                formulas = self.extractor.extract_formulas(pdf_path)
                tables = self.extractor.extract_tables(pdf_path)
                images = self.extractor.extract_images_and_analyze(pdf_path)

                splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
                text_chunks = splitter.split_text(text)
                
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
                        vision_chunks = splitter.split_text(img["text"])
                        for vchunk in vision_chunks:
                            chunks.append(vchunk)
                            meta.append({"type": "text", "page": img["page"], "id": f"vision_text_{len(chunks)}", "vision_fallback": True})

                self.retriever.build_index(chunks, meta)
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}. Please upload a valid file.")

    def ask(self, query: str) -> Dict[str, Any]:
        thinking_steps = []
        thinking_steps.append("🧠 Grok-ing the query: Rewriting for cosmic clarity...")
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite this query in 3 variations for better retrieval in a PDF context: {query}. Focus on key terms."
        )
        rewrite_chain = LLMChain(llm=self.extractor.llm, prompt=rewrite_prompt)
        variants = rewrite_chain.run(query=query).split("\n")[:3]
        thinking_steps.append(f"Generated variants: {', '.join(variants)}")
        
        thinking_steps.append("🌌 Retrieved chunks—analyzing for relevance...")
        all_results = []
        for v in [query] + variants:
            all_results.extend(self.retriever.retrieve(v, top_k=3))
        
        unique_results = {m['id']: (c, m, s) for c, m, s in all_results}
        results = list(unique_results.values())[:5]
        thinking_steps.append(f"Retrieved {len(results)} unique chunks.")
        
        context = build_context(results)
        
        thinking_steps.append("Step 3: Generating response using LLM...")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                f"""You are Grok-inspired PDF Expert, a cosmic guide. Be helpful, witty, and structured: Start with an engaging overview (add humor if fitting), use numbered steps, Markdown tables, LaTeX ($$...$$) for formulas. Cite inline as [Page X, ID Y]. Warn on OCR with ⚠️. Respond in {self.extractor.extracted_content.get('lang', 'en')}.
Few-shot example: For 'explain GIRR',: ### Overview (with witty hook)... Step 1: ... $$ formula $$ ... | Tenor | Weight | ... [Page 7, text_5]. End with: 'Need more cosmic insights?'"""
            ),
            HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        ])
        chain = LLMChain(llm=self.extractor.llm, prompt=prompt)
        
        output = chain.run(context=context, query=query)
        
        thinking_steps.append("Step 4: Refining response for accuracy...")
        reflect_prompt = ChatPromptTemplate.from_template("Refine this answer for accuracy, structure, and completeness: {output}")
        reflect_chain = LLMChain(llm=self.extractor.llm, prompt=reflect_prompt)
        refined_output = reflect_chain.run(output=output)
        
        thinking_steps.append("Step 5: Response ready.")
        refined_output += "\n\n🌟 Grok Tip: For more, ask about related sections—or upload another PDF!"
        
        return {"thinking_steps": thinking_steps, "final_output": refined_output}

    def summarize(self) -> str:
        context = build_context(self.retriever.retrieve("Summarize the document", top_k=10))
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are Grok-inspired. Summarize this PDF in 3-5 bullet points, witty and concise."""
            ),
            HumanMessagePromptTemplate.from_template("Context:\n{context}\nAnswer:")
        ])
        chain = LLMChain(llm=self.extractor.llm, prompt=prompt)
        return chain.run(context=context)

# ============================
# Helpers
# ============================
def format_formula_for_context(formula: Dict) -> str:
    return f"Page {formula['page']}:\n$$ {formula['text']} $$"

def format_table_for_context(table: Dict) -> str:
    df = table['data']
    return f"Page {table['page']}:\n{df.to_markdown(index=False)}"

def build_context(results):
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
# LLaMA Index Retriever
# ============================
class LlamaIndexRetriever:
    def __init__(self):
        self.index = None
        self.metadata = []

    def build_index(self, chunks: List[str], metadata: List[Dict]):
        documents = []
        for chunk, meta in zip(chunks, metadata):
            doc = {"text": chunk, **meta}
            documents.append(doc)
        self.index = VectorStoreIndex.from_documents(documents)
        self.metadata = metadata

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)
        nodes = response.source_nodes
        results = [(node.text, self.metadata[i], node.score) for i, node in enumerate(nodes[:top_k])]
        return results

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
    return send_from_directory('.', 'templates/index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_path
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF uploaded'})
    files = request.files.getlist('pdf')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'})
    pdf_path = [os.path.join(app.config['UPLOAD_FOLDER'], f.filename) for f in files if f.filename]
    for f in files:
        if f.filename:
            f.save(pdf_path[files.index(f)])
    chatbot.build_index_from_pdf(pdf_path)
    return jsonify({'message': 'PDF(s) indexed successfully'})

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

@app.route('/summarize', methods=['POST'])
def summarize():
    if pdf_path is None:
        return jsonify({'error': 'Upload PDF first'})
    summary = chatbot.summarize()
    return jsonify({'summary': summary})

if __name__ == '__main__':
    # Configure LLaMA Index with offline DistilBERT
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="distilbert-base-uncased",
        device="cpu",
        model_kwargs={"quantization_config": quantization_config}
    )
    app.run(debug=True, host='0.0.0.0', port=5000)
