#!/usr/bin/env python3
"""
Enhanced PDF Chatbot with Hybrid Retrieval, Math/Table Formatting,
and Vision Model as Fallback
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

warnings.filterwarnings('ignore')


# ============================
# Hybrid Retriever
# ============================
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=8000, stop_words='english')
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks, self.metadata = [], []
        self.tfidf_matrix, self.embedding_matrix = None, None
    
    def build_index(self, chunks, metadata):
        self.chunks, self.metadata = chunks, metadata
        self.tfidf_matrix = self.tfidf.fit_transform(chunks)
        self.embedding_matrix = self.embedder.encode(chunks, normalize_embeddings=True)
    
    def retrieve(self, query, top_k=5):
        tfidf_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec, self.tfidf_matrix)[0]
        emb = self.embedder.encode([query], normalize_embeddings=True)
        sem_scores = np.dot(self.embedding_matrix, emb[0])
        final_scores = 0.5 * tfidf_scores + 0.5 * sem_scores
        top_idx = np.argsort(final_scores)[::-1][:top_k]
        return [(self.chunks[i], self.metadata[i], final_scores[i]) for i in top_idx]


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
        """Vision fallback only if page text nearly empty"""
        images_data, doc = [], fitz.open(pdf_path)
        for i, p in enumerate(doc):
            txt = p.get_text().strip()
            need_vision = (len(txt.split()) < 10)
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
            resp = requests.post(
                f"{self.vision_model['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {self.vision_model['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": self.vision_model['model'],
                    "messages": [{"role":"user","content":[{"type":"text","text":"Extract table or math from image"},{"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}}]}],
                    "max_tokens": 500
                }, timeout=30
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
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

        # Create chunks
        chunks, meta = [], []
        for f in formulas:
            chunks.append(f["text"]); meta.append(f)
        for t in tables:
            chunks.append(t["data"].to_string()); meta.append(t)
        if text:
            for sec in text.split("\n\n"):
                chunks.append(sec.strip()); meta.append({"type":"text"})
        for img in images:
            if "vision_raw" in img:
                chunks.append(img["vision_raw"]); meta.append(img)

        self.retriever.build_index(chunks, meta)

    def ask(self, query: str) -> str:
        results = self.retriever.retrieve(query, top_k=5)
        context = build_context(results)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a regulatory risk expert.
When answering:
- Use LaTeX ($$...$$) for formulas
- Use Markdown tables for tabular data
- Structure answers as numbered steps (1. Definitions, 2. Formulas, 3. Risk Weights, 4. Correlations, 5. Aggregation)
- If context includes ⚠️, warn that it came from OCR and may be less precise."""
            ),
            HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        ])
        chain = LLMChain(llm=self.extractor.llm, prompt=prompt)
        return chain.run(context=context, query=query)
