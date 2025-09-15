#!/usr/bin/env python3
"""
Enhanced PDF Table & Mathematical Formula Extraction Chatbot
Now with LLM and Vision Model integration for superior performance
"""

import os
import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
import json
import streamlit as st
from collections import defaultdict, Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import base64
from io import BytesIO
from PIL import Image
import requests

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain

warnings.filterwarnings('ignore')

class EnhancedPDFExtractor:
    """Enhanced PDF extractor with LLM and Vision model integration"""
    
    def __init__(self):
        self.extracted_content = {
            'tables': [],
            'formulas': [],
            'text': '',
            'structured_content': [],
            'images': [],
            'metadata': {}
        }
        self.llm = None
        self.vision_model = None
        self.connection_status = "disconnected"
    
    def setup_llm_connection(self, config: Dict) -> bool:
        """Setup LangChain ChatOpenAI connection"""
        try:
            self.llm = ChatOpenAI(
                base_url=config.get('base_url', "http://localhost:8123/v1"),
                api_key=config.get('api_key', "dummy"),
                model=config.get('model', "llama3"),
                temperature=config.get('temperature', 0.3),
                max_tokens=config.get('max_tokens', 4000),
                streaming=config.get('streaming', False)
            )
            
            # Test connection
            test_response = self.llm.invoke([
                SystemMessage(content="You are a PDF analysis expert. Respond with 'Connected' if you receive this."),
                HumanMessage(content="Test connection")
            ])
            
            if test_response and test_response.content:
                self.connection_status = "connected"
                return True
            else:
                self.connection_status = "disconnected"
                return False
                
        except Exception as e:
            st.error(f"LLM Connection Error: {str(e)}")
            self.connection_status = "disconnected"
            return False
    
    def setup_vision_model(self, config: Dict) -> bool:
        """Setup vision model connection for image analysis"""
        try:
            # For vision model, we'll use the same endpoint but with vision capabilities
            self.vision_model = {
                'base_url': config.get('vision_base_url', config.get('base_url', "http://localhost:8123/v1")),
                'api_key': config.get('vision_api_key', config.get('api_key', "dummy")),
                'model': config.get('vision_model', "llava"),
                'enabled': True
            }
            return True
        except Exception as e:
            st.error(f"Vision Model Setup Error: {str(e)}")
            return False
    
    def extract_tables_with_llm(self, pdf_path: str) -> List[Dict]:
        """Extract tables using both traditional methods and LLM enhancement"""
        # First, use traditional extraction
        tables_data = self.extract_tables_pdfplumber(pdf_path)
        
        if self.llm and self.connection_status == "connected":
            # Enhance table extraction with LLM
            for table in tables_data:
                try:
                    enhanced_table = self._enhance_table_with_llm(table)
                    table.update(enhanced_table)
                except Exception as e:
                    print(f"LLM table enhancement failed: {e}")
        
        self.extracted_content['tables'] = tables_data
        return tables_data
    
    def _enhance_table_with_llm(self, table: Dict) -> Dict:
        """Use LLM to enhance table understanding"""
        table_text = table['data'].to_string()
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert at analyzing tabular data. Analyze the following table and provide insights."
            ),
            HumanMessagePromptTemplate.from_template(
                """Analyze this table and provide:
1. A descriptive title/summary
2. Key insights or patterns
3. Data types for each column
4. Any notable relationships or trends

Table data:
{table_text}

Respond in JSON format:
{{"title": "...", "insights": "...", "column_types": {{}}, "trends": "..."}}"""
            )
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(table_text=table_text)
        
        try:
            llm_analysis = json.loads(response)
            return {
                'llm_title': llm_analysis.get('title', ''),
                'llm_insights': llm_analysis.get('insights', ''),
                'llm_column_types': llm_analysis.get('column_types', {}),
                'llm_trends': llm_analysis.get('trends', '')
            }
        except:
            return {'llm_analysis': response}
    
    def extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Extract tables using pdfplumber (baseline method)"""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            try:
                                headers = table[0] if table[0] else [f"Col_{i}" for i in range(len(table[1]))]
                                df = pd.DataFrame(table[1:], columns=headers)
                                df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')
                                
                                if not df.empty:
                                    quality_score = self._calculate_table_quality(df)
                                    
                                    tables_data.append({
                                        'table_id': len(tables_data),
                                        'page': page_num + 1,
                                        'data': df,
                                        'quality_score': quality_score,
                                        'row_count': len(df),
                                        'col_count': len(df.columns),
                                        'extraction_method': 'pdfplumber'
                                    })
                            except Exception as e:
                                print(f"Error processing table on page {page_num + 1}: {e}")
                                continue
        except Exception as e:
            print(f"Error opening PDF: {e}")
        
        return tables_data
    
    def extract_images_and_analyze(self, pdf_path: str) -> List[Dict]:
        """Extract images from PDF and analyze with vision model"""
        images_data = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract images from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        # Convert to base64 for vision model
                        img_b64 = base64.b64encode(img_data).decode()
                        
                        image_info = {
                            'page': page_num + 1,
                            'image_id': f"img_{page_num}_{img_index}",
                            'size': (pix.width, pix.height),
                            'image_data': img_b64
                        }
                        
                        # Analyze with vision model if available
                        if self.vision_model and self.vision_model.get('enabled'):
                            vision_analysis = self._analyze_image_with_vision(img_b64)
                            image_info.update(vision_analysis)
                        
                        images_data.append(image_info)
                    
                    pix = None
                except Exception as e:
                    print(f"Error processing image on page {page_num + 1}: {e}")
        
        doc.close()
        self.extracted_content['images'] = images_data
        return images_data
    
    def _analyze_image_with_vision(self, image_b64: str) -> Dict:
        """Analyze image using vision model"""
        try:
            # Prepare vision model request
            vision_prompt = """Analyze this image from a PDF document. Identify:
1. Type of content (table, chart, diagram, formula, etc.)
2. Key information or data visible
3. Any text or numbers you can read
4. Relevant insights for document understanding

Respond in JSON format:
{"content_type": "...", "description": "...", "extracted_text": "...", "insights": "..."}"""
            
            # Make request to vision model endpoint
            response = requests.post(
                f"{self.vision_model['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.vision_model['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.vision_model['model'],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                            ]
                        }
                    ],
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    vision_analysis = json.loads(content)
                    return {
                        'vision_analysis': vision_analysis,
                        'vision_raw': content
                    }
                except:
                    return {'vision_raw': content}
            else:
                return {'vision_error': f"Vision API error: {response.status_code}"}
                
        except Exception as e:
            return {'vision_error': f"Vision analysis failed: {str(e)}"}
    
    def extract_mathematical_content_with_llm(self, pdf_path: str) -> List[Dict]:
        """Extract mathematical content using regex + LLM enhancement"""
        # First, use regex-based extraction
        formulas = self.extract_mathematical_content_regex(pdf_path)
        
        if self.llm and self.connection_status == "connected":
            # Enhance with LLM analysis
            for formula_page in formulas:
                try:
                    enhanced_formulas = self._enhance_formulas_with_llm(formula_page)
                    formula_page.update(enhanced_formulas)
                except Exception as e:
                    print(f"LLM formula enhancement failed: {e}")
        
        self.extracted_content['formulas'] = formulas
        return formulas
    
    def _enhance_formulas_with_llm(self, formula_page: Dict) -> Dict:
        """Use LLM to enhance formula understanding"""
        formulas_text = "\n".join([f["text"] for f in formula_page['formulas']])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a mathematics expert. Analyze mathematical formulas and equations."
            ),
            HumanMessagePromptTemplate.from_template(
                """Analyze these mathematical expressions and provide:
1. What each formula represents
2. The mathematical domain/field
3. Key variables and their likely meanings
4. Applications or context

Mathematical expressions:
{formulas_text}

Respond in JSON format with analysis for each formula."""
            )
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(formulas_text=formulas_text)
        
        return {'llm_math_analysis': response}
    
    def extract_mathematical_content_regex(self, pdf_path: str) -> List[Dict]:
        """Extract mathematical content using regex patterns (baseline)"""
        formulas = []
        doc = fitz.open(pdf_path)
        
        math_patterns = {
            'equations': [
                r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=\n]+',
                r'[a-zA-Z_]\([^)]+\)\s*=\s*[^=\n]+',
                r'\b\w+\s*=\s*\d+\.?\d*\s*[+\-*/]\s*\w+',
            ],
            'functions': [
                r'\b(sin|cos|tan|cot|sec|csc|sinh|cosh|tanh)\s*\([^)]+\)',
                r'\b(log|ln|exp|sqrt|abs)\s*\([^)]+\)',
                r'\b(max|min|sum|prod)\s*\([^)]+\)',
            ],
            'integrals': [
                r'∫[^∫]*d[a-zA-Z]',
                r'\\int[^\\]*d[a-zA-Z]',
            ],
            'derivatives': [
                r'd[a-zA-Z]/d[a-zA-Z]',
                r'∂[a-zA-Z]/∂[a-zA-Z]',
            ],
            'greek_letters': [r'[αβγδεζηθικλμνξοπρστυφχψω]'],
            'mathematical_symbols': [r'[≤≥≠≈∞∑∏∫∂√±×÷]']
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            page_formulas = []
            for category, patterns in math_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, page_text, re.IGNORECASE)
                    for match in matches:
                        formula_text = match.group().strip()
                        if len(formula_text) > 2:
                            page_formulas.append({
                                'text': formula_text,
                                'category': category,
                                'confidence': self._calculate_formula_confidence(formula_text, category)
                            })
            
            if page_formulas:
                formulas.append({
                    'page': page_num + 1,
                    'formulas': page_formulas,
                    'extraction_method': 'regex_patterns'
                })
        
        doc.close()
        return formulas
    
    def _calculate_table_quality(self, df: pd.DataFrame) -> float:
        """Calculate table quality score"""
        if df.empty:
            return 0.0
        
        score = 0.0
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        score += completeness * 0.4
        
        consistency = 0.0
        for col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                try:
                    pd.to_numeric(non_null_values)
                    consistency += 1
                except:
                    consistency += 0.5
        consistency = consistency / len(df.columns) if len(df.columns) > 0 else 0
        score += consistency * 0.3
        
        size_score = min(len(df) / 20, 1.0) * min(len(df.columns) / 10, 1.0)
        score += size_score * 0.3
        
        return min(score, 1.0)
    
    def _calculate_formula_confidence(self, formula_text: str, category: str) -> float:
        """Calculate formula confidence score"""
        score = 0.5
        if len(formula_text) > 10:
            score += 0.2
        
        category_weights = {
            'equations': 0.3, 'functions': 0.25, 'integrals': 0.4,
            'derivatives': 0.35, 'greek_letters': 0.1, 'mathematical_symbols': 0.15
        }
        score += category_weights.get(category, 0.1)
        
        math_indicators = ['=', '+', '-', '*', '/', '^', '(', ')']
        indicator_count = sum(1 for indicator in math_indicators if indicator in formula_text)
        score += min(indicator_count * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def extract_text_with_metadata(self, pdf_path: str) -> str:
        """Extract text with metadata"""
        doc = fitz.open(pdf_path)
        formatted_text = ""
        
        metadata = doc.metadata
        self.extracted_content['metadata'] = {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'pages': len(doc)
        }
        
        for page_num, page in enumerate(doc):
            formatted_text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
            formatted_text += page.get_text()
        
        doc.close()
        self.extracted_content['text'] = formatted_text
        return formatted_text
    
    def process_pdf(self, pdf_path: str, llm_config: Optional[Dict] = None, vision_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Process PDF with optional LLM and vision enhancement"""
        
        # Setup models if config provided
        if llm_config:
            self.setup_llm_connection(llm_config)
        if vision_config:
            self.setup_vision_model(vision_config)
        
        print("Extracting tables...")
        self.extract_tables_with_llm(pdf_path)
        
        print("Extracting mathematical content...")
        self.extract_mathematical_content_with_llm(pdf_path)
        
        print("Extracting and analyzing images...")
        self.extract_images_and_analyze(pdf_path)
        
        print("Extracting text with metadata...")
        self.extract_text_with_metadata(pdf_path)
        
        return self.extracted_content

class IntelligentChatbotWithLLM:
    """Intelligent chatbot using LLM for superior question answering"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.content_vectors = None
        self.content_chunks = []
        self.content_metadata = []
        self.llm = None
        self.connection_status = "disconnected"
    
    def setup_llm_connection(self, config: Dict) -> bool:
        """Setup LLM connection for chatbot"""
        try:
            self.llm = ChatOpenAI(
                base_url=config.get('base_url', "http://localhost:8123/v1"),
                api_key=config.get('api_key', "dummy"),
                model=config.get('model', "llama3"),
                temperature=config.get('temperature', 0.3),
                max_tokens=config.get('max_tokens', 4000),
                streaming=config.get('streaming', False)
            )
            
            test_response = self.llm.invoke([
                SystemMessage(content="You are a PDF analysis assistant. Say 'Ready' if connected."),
                HumanMessage(content="Test")
            ])
            
            if test_response and test_response.content:
                self.connection_status = "connected"
                return True
            else:
                self.connection_status = "disconnected"
                return False
                
        except Exception as e:
            st.error(f"Chatbot LLM Connection Error: {str(e)}")
            self.connection_status = "disconnected"
            return False
    
    def prepare_content_for_search(self, extracted_content: Dict) -> List[str]:
        """Prepare content for search with enhanced metadata"""
        chunks = []
        metadata = []
        
        # Add text content
        if extracted_content.get('text'):
            paragraphs = [p.strip() for p in extracted_content['text'].split('\n\n') 
                         if p.strip() and len(p.strip()) > 20]
            for para in paragraphs:
                chunks.append(para)
                metadata.append({'type': 'text', 'source': 'document'})
        
        # Add enhanced table information
        for table in extracted_content.get('tables', []):
            if 'data' in table:
                table_df = table['data']
                table_text = f"Table from page {table['page']}. "
                table_text += f"Columns: {', '.join(table_df.columns.tolist())}. "
                
                # Add LLM insights if available
                if 'llm_title' in table:
                    table_text += f"Title: {table['llm_title']}. "
                if 'llm_insights' in table:
                    table_text += f"Insights: {table['llm_insights']}. "
                
                # Add table data
                for _, row in table_df.iterrows():
                    row_text = ' '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    table_text += row_text + ". "
                
                chunks.append(table_text)
                metadata.append({'type': 'table', 'page': table['page'], 'table_id': table['table_id']})
        
        # Add mathematical formulas with LLM analysis
        for formula_page in extracted_content.get('formulas', []):
            for formula in formula_page['formulas']:
                formula_text = f"Mathematical {formula['category']} from page {formula_page['page']}: {formula['text']}"
                chunks.append(formula_text)
                metadata.append({'type': 'formula', 'page': formula_page['page'], 'category': formula['category']})
            
            # Add LLM mathematical analysis if available
            if 'llm_math_analysis' in formula_page:
                chunks.append(f"Mathematical analysis from page {formula_page['page']}: {formula_page['llm_math_analysis']}")
                metadata.append({'type': 'math_analysis', 'page': formula_page['page']})
        
        # Add image analysis
        for image in extracted_content.get('images', []):
            if 'vision_analysis' in image:
                analysis = image['vision_analysis']
                image_text = f"Image from page {image['page']}: {analysis.get('content_type', 'unknown')}. "
                image_text += f"Description: {analysis.get('description', '')}. "
                image_text += f"Extracted text: {analysis.get('extracted_text', '')}."
                chunks.append(image_text)
                metadata.append({'type': 'image', 'page': image['page']})
        
        self.content_chunks = chunks
        self.content_metadata = metadata
        return chunks
    
    def build_search_index(self, content_chunks: List[str]):
        """Build search index"""
        if not content_chunks:
            return
        self.content_vectors = self.vectorizer.fit_transform(content_chunks)
    
    def find_relevant_content(self, question: str, top_k: int = 5) -> List[Dict]:
        """Find relevant content"""
        if self.content_vectors is None or not self.content_chunks:
            return []
        
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.content_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({
                    'content': self.content_chunks[idx],
                    'metadata': self.content_metadata[idx],
                    'similarity': similarities[idx]
                })
        
        return results
    
    def answer_question_with_llm(self, question: str) -> str:
        """Answer question using LLM with retrieved context"""
        if self.llm and self.connection_status == "connected":
            # Get relevant content
            relevant_results = self.find_relevant_content(question, top_k=5)
            
            if not relevant_results:
                return "I couldn't find relevant information in the PDF to answer your question."
            
            # Prepare context for LLM
            context = "\n\n".join([result['content'] for result in relevant_results])
            
            # Create LLM prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """You are an expert PDF document analyst. Answer questions based on the provided context from the PDF document. 
                    
Guidelines:
- Use only the information provided in the context
- Be specific and cite page numbers when available
- If the context doesn't contain enough information, say so
- For tables, provide clear data summaries
- For mathematical content, explain formulas clearly
- Be concise but comprehensive"""
                ),
                HumanMessagePromptTemplate.from_template(
                    """Context from PDF:
{context}

Question: {question}

Answer based on the context above:"""
                )
            ])
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run(context=context, question=question)
            
            return response
        else:
            # Fallback to basic method
            return self.answer_question_basic(question)
    
    def answer_question_basic(self, question: str) -> str:
        """Fallback basic question answering"""
        relevant_results = self.find_relevant_content(question, top_k=3)
        
        if not relevant_results:
            return "I couldn't find relevant information in the PDF to answer your question."
        
        answer = "Based on the content I found:\n\n"
        for i, result in enumerate(relevant_results, 1):
            similarity_percent = int(result['similarity'] * 100)
            answer += f"**Result {i}** (Relevance: {similarity_percent}%):\n"
            answer += f"{result['content'][:400]}...\n\n"
        
        return answer

def main():
    """Enhanced Streamlit app with LLM integration"""
    st.set_page_config(page_title="Enhanced PDF Chatbot", page_icon="🚀", layout="wide")
    
    st.title("🚀 Enhanced PDF Chatbot with LLM & Vision")
    st.write("Advanced PDF processing with LLM and Vision model integration!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'extracted_content' not in st.session_state:
        st.session_state.extracted_content = None
    if 'extractor' not in st.session_state:
        st.session_state.extractor = None
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # LLM Configuration
        st.subheader("LLM Settings")
        llm_enabled = st.checkbox("Enable LLM Enhancement", value=True)
        
        if llm_enabled:
            llm_base_url = st.text_input("LLM Base URL", value="http://localhost:8123/v1")
            llm_api_key = st.text_input("LLM API Key", value="dummy", type="password")
            llm_model = st.text_input("LLM Model", value="llama3")
            llm_temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
            llm_max_tokens = st.number_input("Max Tokens", 1000, 8000, 4000)
        
        # Vision Model Configuration
        st.subheader("Vision Model Settings")
        vision_enabled = st.checkbox("Enable Vision Analysis", value=True)
        
        if vision_enabled:
            vision_base_url = st.text_input("Vision Base URL", value="http://localhost:8123/v1")
            vision_api_key = st.text_input("Vision API Key", value="dummy", type="password")
            vision_model = st.text_input("Vision Model", value="llava")
        
        # Test connections
        if st.button("🧪 Test Connections"):
            if llm_enabled:
                extractor = EnhancedPDFExtractor()
                llm_config = {
                    'base_url': llm_base_url,
                    'api_key': llm_api_key,
                    'model': llm_model,
                    'temperature': llm_temperature,
                    'max_tokens': llm_max_tokens
                }
                if extractor.setup_llm_connection(llm_config):
                    st.success("✅ LLM Connected!")
                else:
                    st.error("❌ LLM Connection Failed")
            
            if vision_enabled:
                st.info("Vision model connection will be tested during processing")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Save uploaded file
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🚀 Process PDF with Enhanced Models", type="primary"):
                with st.spinner("Processing PDF with advanced AI models... This may take several minutes."):
                    try:
                        # Prepare configurations
                        llm_config = None
                        vision_config = None
                        
                        if llm_enabled:
                            llm_config = {
                                'base_url': llm_base_url,
                                'api_key': llm_api_key,
                                'model': llm_model,
                                'temperature': llm_temperature,
                                'max_tokens': llm_max_tokens
                            }
                        
                        if vision_enabled:
                            vision_config = {
                                'vision_base_url': vision_base_url,
                                'vision_api_key': vision_api_key,
                                'vision_model': vision_model
                            }
                        
                        # Extract content with enhanced models
                        extractor = EnhancedPDFExtractor()
                        extracted_content = extractor.process_pdf("temp_upload.pdf", llm_config, vision_config)
                        st.session_state.extracted_content = extracted_content
                        st.session_state.extractor = extractor
                        
                        # Initialize enhanced chatbot
                        chatbot = IntelligentChatbotWithLLM()
                        if llm_enabled and llm_config:
                            chatbot.setup_llm_connection(llm_config)
                        
                        content_chunks = chatbot.prepare_content_for_search(extracted_content)
                        chatbot.build_search_index(content_chunks)
                        st.session_state.chatbot = chatbot
                        
                        st.success("✅ PDF processed successfully with AI enhancement!")
                        
                        # Display enhanced processing summary
                        col1_sum, col2_sum, col3_sum, col4_sum, col5_sum = st.columns(5)
                        with col1_sum:
                            st.metric("📊 Tables", len(extracted_content['tables']))
                        with col2_sum:
                            formula_count = sum(len(fp['formulas']) for fp in extracted_content['formulas'])
                            st.metric("🧮 Formulas", formula_count)
                        with col3_sum:
                            st.metric("🖼️ Images", len(extracted_content['images']))
                        with col4_sum:
                            st.metric("📄 Pages", extracted_content['metadata'].get('pages', 0))
                        with col5_sum:
                            llm_status = "🟢 Connected" if extractor.connection_status == "connected" else "🔴 Offline"
                            st.metric("🤖 LLM", llm_status)
                    
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        st.exception(e)
    
    with col2:
        # Quick stats and status
        if st.session_state.extracted_content:
            st.subheader("📈 Processing Status")
            extracted_content = st.session_state.extracted_content
            
            # LLM Enhancement Status
            llm_enhanced_tables = sum(1 for t in extracted_content['tables'] if 'llm_title' in t)
            st.write(f"🤖 LLM Enhanced Tables: {llm_enhanced_tables}/{len(extracted_content['tables'])}")
            
            # Vision Analysis Status
            vision_analyzed_images = sum(1 for img in extracted_content['images'] if 'vision_analysis' in img)
            st.write(f"👁️ Vision Analyzed Images: {vision_analyzed_images}/{len(extracted_content['images'])}")
            
            # Formula Analysis
            formula_pages_with_llm = sum(1 for fp in extracted_content['formulas'] if 'llm_math_analysis' in fp)
            st.write(f"🧮 LLM Analyzed Formulas: {formula_pages_with_llm}/{len(extracted_content['formulas'])}")
    
    # Display extracted content with enhancements
    if st.session_state.extracted_content:
        extracted_content = st.session_state.extracted_content
        
        # Enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Enhanced Tables", "🧮 Smart Formulas", "🖼️ Vision Analysis", "📋 Structure", "ℹ️ Metadata"])
        
        with tab1:
            if extracted_content['tables']:
                st.subheader(f"Enhanced Tables ({len(extracted_content['tables'])})")
                for i, table in enumerate(extracted_content['tables']):
                    with st.expander(f"Table {i+1} - Page {table['page']} (Quality: {table['quality_score']:.2f})"):
                        
                        # Show LLM enhancements if available
                        if 'llm_title' in table:
                            st.write(f"**🤖 AI Title:** {table['llm_title']}")
                        if 'llm_insights' in table:
                            st.write(f"**🤖 AI Insights:** {table['llm_insights']}")
                        if 'llm_trends' in table:
                            st.write(f"**🤖 AI Trends:** {table['llm_trends']}")
                        
                        st.dataframe(table['data'], use_container_width=True)
                        st.caption(f"Rows: {table['row_count']}, Columns: {table['col_count']}")
            else:
                st.info("No tables found in the PDF.")
        
        with tab2:
            if extracted_content['formulas']:
                st.subheader("Smart Mathematical Analysis")
                for formula_page in extracted_content['formulas']:
                    st.write(f"**Page {formula_page['page']}:**")
                    
                    # Show LLM mathematical analysis if available
                    if 'llm_math_analysis' in formula_page:
                        st.write("**🤖 AI Mathematical Analysis:**")
                        st.write(formula_page['llm_math_analysis'])
                        st.divider()
                    
                    # Show individual formulas
                    for formula in formula_page['formulas']:
                        confidence_color = "🟢" if formula['confidence'] > 0.7 else "🟡" if formula['confidence'] > 0.4 else "🔴"
                        st.write(f"{confidence_color} **{formula['category'].title()}** (Confidence: {formula['confidence']:.2f})")
                        st.code(formula['text'], language='text')
            else:
                st.info("No mathematical formulas found in the PDF.")
        
        with tab3:
            if extracted_content['images']:
                st.subheader("Vision Model Analysis")
                for image in extracted_content['images']:
                    with st.expander(f"Image from Page {image['page']} - {image['image_id']}"):
                        
                        # Display image
                        try:
                            img_data = base64.b64decode(image['image_data'])
                            img = Image.open(BytesIO(img_data))
                            st.image(img, caption=f"Image from page {image['page']}", width=300)
                        except:
                            st.write("Image preview not available")
                        
                        # Show vision analysis
                        if 'vision_analysis' in image:
                            analysis = image['vision_analysis']
                            st.write(f"**👁️ Content Type:** {analysis.get('content_type', 'Unknown')}")
                            st.write(f"**👁️ Description:** {analysis.get('description', 'No description')}")
                            st.write(f"**👁️ Extracted Text:** {analysis.get('extracted_text', 'No text found')}")
                            st.write(f"**👁️ Insights:** {analysis.get('insights', 'No insights')}")
                        elif 'vision_error' in image:
                            st.error(f"Vision analysis error: {image['vision_error']}")
                        else:
                            st.info("Vision analysis not performed")
                        
                        st.write(f"**Size:** {image['size'][0]} x {image['size'][1]} pixels")
            else:
                st.info("No images found in the PDF.")
        
        with tab4:
            # Same structure analysis as before
            if extracted_content.get('structured_content'):
                st.subheader("Document Structure")
                for page_content in extracted_content['structured_content']:
                    with st.expander(f"Page {page_content['page']} Structure"):
                        structure = page_content['structure']
                        
                        if structure.get('headers'):
                            st.write("**Headers:**")
                            for header in structure['headers']:
                                level_emoji = "🔵" if header.get('level', 1) == 1 else "🟡" if header.get('level', 1) == 2 else "⚪"
                                st.write(f"{level_emoji} {header['text']}")
                        
                        if structure.get('equations'):
                            st.write("**Equations:**")
                            for eq in structure['equations']:
                                st.code(eq['text'])
            else:
                st.info("No structured content analysis available.")
        
        with tab5:
            metadata = extracted_content.get('metadata', {})
            if metadata:
                st.subheader("Document Metadata")
                for key, value in metadata.items():
                    if value:
                        st.write(f"**{key.title()}:** {value}")
            else:
                st.info("No metadata available.")
    
    # Enhanced Chatbot Interface
    if st.session_state.chatbot:
        st.divider()
        st.subheader("🤖 Intelligent PDF Assistant")
        
        # Show chatbot capabilities
        col_chat1, col_chat2 = st.columns([3, 1])
        
        with col_chat1:
            # Enhanced sample questions
            with st.expander("💡 Enhanced Sample Questions"):
                st.write("**Table Questions:**")
                st.write("• Analyze the trends in the first table")
                st.write("• What insights can you derive from the financial data?")
                st.write("• Compare the values across different columns")
                
                st.write("**Formula Questions:**")
                st.write("• Explain the mathematical formulas in detail")
                st.write("• What do these equations represent?")
                st.write("• How are these formulas applied?")
                
                st.write("**Image Questions:**")
                st.write("• What do the charts and diagrams show?")
                st.write("• Describe the visual content")
                st.write("• What data is represented in the images?")
                
                st.write("**General Questions:**")
                st.write("• Provide a comprehensive summary")
                st.write("• What are the main findings?")
                st.write("• How do all the elements relate?")
            
            question = st.text_input("Ask your enhanced AI assistant:", placeholder="What would you like to know about the PDF?")
            
            if question:
                with st.spinner("🤔 AI is analyzing your question..."):
                    try:
                        # Use LLM-enhanced answering if available
                        if hasattr(st.session_state.chatbot, 'llm') and st.session_state.chatbot.connection_status == "connected":
                            answer = st.session_state.chatbot.answer_question_with_llm(question)
                            st.write("**🤖 AI-Enhanced Answer:**")
                        else:
                            answer = st.session_state.chatbot.answer_question_basic(question)
                            st.write("**🤖 Standard Answer:**")
                        
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        
        with col_chat2:
            # Chatbot status
            st.write("**🤖 AI Status:**")
            if hasattr(st.session_state.chatbot, 'connection_status'):
                if st.session_state.chatbot.connection_status == "connected":
                    st.success("🟢 LLM Connected")
                else:
                    st.warning("🟡 Basic Mode")
            
            # Model info
            if st.session_state.extractor and hasattr(st.session_state.extractor, 'llm'):
                st.write("**Model Info:**")
                st.write(f"LLM: {llm_model if 'llm_model' in locals() else 'N/A'}")
                st.write(f"Vision: {vision_model if 'vision_model' in locals() else 'N/A'}")
        
        # Enhanced chat history
        if 'enhanced_chat_history' not in st.session_state:
            st.session_state.enhanced_chat_history = []
        
        if question and question not in [chat['question'] for chat in st.session_state.enhanced_chat_history]:
            try:
                if hasattr(st.session_state.chatbot, 'llm') and st.session_state.chatbot.connection_status == "connected":
                    answer = st.session_state.chatbot.answer_question_with_llm(question)
                    answer_type = "AI-Enhanced"
                else:
                    answer = st.session_state.chatbot.answer_question_basic(question)
                    answer_type = "Standard"
                
                st.session_state.enhanced_chat_history.append({
                    'question': question,
                    'answer': answer,
                    'type': answer_type
                })
            except:
                pass
        
        # Display enhanced chat history
        if st.session_state.enhanced_chat_history:
            with st.expander("📝 Enhanced Chat History"):
                for i, chat in enumerate(reversed(st.session_state.enhanced_chat_history[-5:]), 1):
                    type_emoji = "🤖" if chat['type'] == "AI-Enhanced" else "📝"
                    st.write(f"**{type_emoji} Q{i} ({chat['type']}):** {chat['question']}")
                    st.write(f"**A{i}:** {chat['answer'][:300]}...")
                    st.divider()
    
    # Clean up temp file
    if uploaded_file and os.path.exists("temp_upload.pdf"):
        os.remove("temp_upload.pdf")

def demo_command_line_enhanced():
    """Enhanced command line demo with LLM integration"""
    print("=== Enhanced PDF Chatbot with LLM & Vision ===")
    print("Advanced AI-powered PDF analysis!")
    
    pdf_path = input("Enter PDF file path: ").strip()
    
    if not os.path.exists(pdf_path):
        print("❌ File not found!")
        return
    
    # Get LLM configuration
    use_llm = input("Use LLM enhancement? (y/n): ").lower().startswith('y')
    llm_config = None
    
    if use_llm:
        base_url = input("LLM Base URL (default: http://localhost:8123/v1): ").strip()
        if not base_url:
            base_url = "http://localhost:8123/v1"
        
        model = input("Model name (default: llama3): ").strip()
        if not model:
            model = "llama3"
        
        llm_config = {
            'base_url': base_url,
            'api_key': 'dummy',
            'model': model,
            'temperature': 0.3,
            'max_tokens': 4000
        }
    
    print("\n🚀 Processing PDF with enhanced AI models...")
    
    # Extract content with enhancements
    extractor = EnhancedPDFExtractor()
    content = extractor.process_pdf(pdf_path, llm_config)
    
    # Print enhanced summary
    print(f"\n📊 Enhanced Processing Summary:")
    print(f"  Tables found: {len(content['tables'])}")
    print(f"  Formulas found: {sum(len(fp['formulas']) for fp in content['formulas'])}")
    print(f"  Images analyzed: {len(content['images'])}")
    print(f"  Pages: {content['metadata'].get('pages', 0)}")
    print(f"  LLM Status: {'🟢 Connected' if extractor.connection_status == 'connected' else '🔴 Offline'}")
    
    # Initialize enhanced chatbot
    chatbot = IntelligentChatbotWithLLM()
    if llm_config:
        chatbot.setup_llm_connection(llm_config)
    
    chunks = chatbot.prepare_content_for_search(content)
    chatbot.build_search_index(chunks)
    
    print(f"  Searchable content chunks: {len(chunks)}")
    print("\n🤖 Enhanced AI Assistant ready!")
    print("💡 Try: 'Analyze the main table' or 'Explain the mathematical formulas'")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            try:
                if chatbot.connection_status == "connected":
                    answer = chatbot.answer_question_with_llm(question)
                    print(f"\n🤖 AI-Enhanced Answer:\n{answer}")
                else:
                    answer = chatbot.answer_question_basic(question)
                    print(f"\n📝 Standard Answer:\n{answer}")
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        demo_command_line_enhanced()
    else:
        # Streamlit mode
        main()
