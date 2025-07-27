#!/usr/bin/env python3
"""
Legal Document Compliance Analyzer with Legal-BERT

This application reads a PDF document from a specified path and performs
legal compliance analysis using Legal-BERT for enhanced semantic understanding
of legal clauses and calculation of various risk and completeness scores.
"""

import os
import json
import argparse
import time
import psutil
import platform
from datetime import datetime
from PyPDF2 import PdfReader
import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Global variables for Legal-BERT model
legal_tokenizer = None
legal_model = None
nlp = None

# Performance metrics storage
performance_metrics = {
    'system_info': {},
    'timing': {},
    'memory': {},
    'model_info': {},
    'analysis_stats': {}
}

def get_system_info():
    """Get comprehensive system information for benchmarking"""
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    # Get GPU memory info if available
    if torch.cuda.is_available():
        system_info['gpu_memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        system_info['gpu_memory_allocated_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
    
    return system_info

def start_timer(operation_name):
    """Start timing an operation"""
    performance_metrics['timing'][operation_name] = {
        'start_time': time.time(),
        'start_memory': psutil.Process().memory_info().rss / (1024**2)  # MB
    }

def end_timer(operation_name):
    """End timing an operation and calculate metrics"""
    if operation_name in performance_metrics['timing']:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        timing_info = performance_metrics['timing'][operation_name]
        duration = end_time - timing_info['start_time']
        memory_delta = end_memory - timing_info['start_memory']
        
        performance_metrics['timing'][operation_name].update({
            'end_time': end_time,
            'duration_seconds': duration,
            'end_memory_mb': end_memory,
            'memory_delta_mb': memory_delta
        })

def load_legal_bert():
    """Load Legal-BERT model for specialized legal text analysis"""
    global legal_tokenizer, legal_model
    
    start_timer('legal_bert_loading')
    
    try:
        print("Loading Legal-BERT model...")
        model_name = "nlpaueb/legal-bert-base-uncased"
        
        # Time tokenizer loading
        start_timer('tokenizer_loading')
        legal_tokenizer = AutoTokenizer.from_pretrained(model_name)
        end_timer('tokenizer_loading')
        
        # Time model loading
        start_timer('model_loading')
        legal_model = AutoModel.from_pretrained(model_name)
        end_timer('model_loading')
        
        # Set model to evaluation mode
        legal_model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        legal_model.to(device)
        
        # Store model info
        performance_metrics['model_info'] = {
            'model_name': model_name,
            'device': str(device),
            'model_parameters': sum(p.numel() for p in legal_model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in legal_model.parameters()) / (1024**2)
        }
        
        print(f"Legal-BERT loaded successfully on {device}")
        end_timer('legal_bert_loading')
        return True
    except Exception as e:
        print(f"Error loading Legal-BERT: {e}")
        print("Falling back to spaCy model...")
        end_timer('legal_bert_loading')
        return False

def load_spacy_model():
    """Load spaCy model as fallback"""
    global nlp
    start_timer('spacy_loading')
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model for fallback analysis")
        end_timer('spacy_loading')
        return True
    except OSError:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        end_timer('spacy_loading')
        return True

def get_legal_bert_embedding(text, max_length=512):
    """Get Legal-BERT embedding for text"""
    global legal_tokenizer, legal_model
    
    if legal_tokenizer is None or legal_model is None:
        return None
    
    start_timer('embedding_generation')
    try:
        # Tokenize and prepare input
        inputs = legal_tokenizer(
            text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True, 
            padding=True
        )
        
        # Move inputs to same device as model
        device = next(legal_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = legal_model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        end_timer('embedding_generation')
        return embedding
    except Exception as e:
        print(f"Error getting Legal-BERT embedding: {e}")
        end_timer('embedding_generation')
        return None

def get_semantic_similarity_legal_bert(text1, text2, threshold=0.6):
    """Calculate semantic similarity between texts using Legal-BERT"""
    start_timer('semantic_similarity')
    # Get embeddings
    emb1 = get_legal_bert_embedding(text1)
    emb2 = get_legal_bert_embedding(text2)
    
    if emb1 is None or emb2 is None:
        end_timer('semantic_similarity')
        return 0.0, False
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    end_timer('semantic_similarity')
    return similarity, similarity >= threshold

def get_semantic_similarity_spacy(text1, text2, threshold=0.6):
    """Calculate semantic similarity using spaCy as fallback"""
    global nlp
    if nlp is None:
        return 0.0, False
    
    doc1 = nlp(text1.lower())
    doc2 = nlp(text2.lower())
    similarity = doc1.similarity(doc2)
    return similarity, similarity >= threshold

def analyze_clause_with_semantics(text, clause_info, document_type, use_legal_bert=True):
    """Analyze clause presence using semantic similarity"""
    text_lower = text.lower()
    exact_matches = []
    semantic_matches = []
    sentences_processed = 0  # Add counter
    
    # First, try exact keyword matching (faster)
    for alt in clause_info["alternatives"]:
        if alt in text_lower:
            exact_matches.append(alt)
    
    # If no exact matches or few matches, try semantic similarity
    if len(exact_matches) < 2:
        clause_description = clause_info["description"]
        
        # Split text into sentences for better semantic analysis
        if use_legal_bert and legal_tokenizer is not None:
            # Use Legal-BERT for semantic analysis
            doc = nlp(text) if nlp else None
            if doc:
                sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
            else:
                # Simple sentence splitting as fallback
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            
            # Analyze each sentence
            for sentence in sentences[:15]:  # Limit for performance
                sentences_processed += 1  # Increment counter
                # Increase threshold from 0.5 to 0.7 for better precision
                similarity, is_match = get_semantic_similarity_legal_bert(
                    sentence, clause_description, threshold=0.7  # Was 0.5
                )
                if is_match:
                    semantic_matches.append({
                        'sentence': sentence,
                        'similarity': similarity,
                        'clause': clause_description
                    })
        else:
            # Fallback to spaCy
            sentences_processed = 1  # Count as one sentence processed
            similarity, is_match = get_semantic_similarity_spacy(
                text[:1000], clause_description, threshold=0.5
            )
            if is_match:
                semantic_matches.append({
                    'text': text[:200] + "...",
                    'similarity': similarity,
                    'clause': clause_description
                })
    
    return exact_matches, semantic_matches, sentences_processed  # Return the counter

def load_legal_clauses():
    """Load legal clause definitions from JSON file"""
    start_timer('clauses_loading')
    try:
        with open('legal_clauses.json', 'r') as f:
            clauses_data = json.load(f)
        end_timer('clauses_loading')
        return clauses_data
    except FileNotFoundError:
        print("Error: legal_clauses.json not found. Please ensure the file exists in the same directory.")
        end_timer('clauses_loading')
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in legal_clauses.json: {e}")
        end_timer('clauses_loading')
        return None

def load_clause_translations():
    """Load clause letter to name translations from JSON file"""
    start_timer('translations_loading')
    try:
        with open('clause_translations.json', 'r') as f:
            translations = json.load(f)
        end_timer('translations_loading')
        return translations
    except FileNotFoundError:
        print("Error: clause_translations.json not found. Please ensure the file exists in the same directory.")
        end_timer('translations_loading')
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in clause_translations.json: {e}")
        end_timer('translations_loading')
        return None

def translate_clause_letters(letters, document_type, translations):
    """Translate clause letters to human-readable names"""
    if not letters or not translations or document_type not in translations:
        return letters
    
    translated = []
    for letter in letters:
        if letter in translations[document_type]:
            translated.append(f"{letter}: {translations[document_type][letter]}")
        else:
            translated.append(letter)
    
    return ", ".join(translated)

def detect_document_type(text):
    """Automatically detect document type based on content"""
    text_lower = text.lower()
    
    # NDA indicators
    nda_indicators = [
        'non-disclosure agreement', 'nda', 'confidentiality agreement', 
        'receiving party', 'disclosing party', 'confidential information',
        'non-disclosure', 'confidentiality obligations', 'trade secrets'
    ]
    nda_score = sum(1 for indicator in nda_indicators if indicator in text_lower)
    
    # Employment contract indicators
    employment_indicators = [
        'employment agreement', 'employee', 'employer', 'salary', 
        'compensation', 'job duties', 'work schedule', 'employment contract',
        'probation period', 'termination of employment', 'workplace policies'
    ]
    employment_score = sum(1 for indicator in employment_indicators if indicator in text_lower)
    
    # Service agreement indicators
    service_indicators = [
        'service agreement', 'services provided', 'service description',
        'consulting services', 'professional services', 'service provider'
    ]
    service_score = sum(1 for indicator in service_indicators if indicator in text_lower)
    
    # Lease agreement indicators
    lease_indicators = [
        'lease agreement', 'rental agreement', 'tenant', 'landlord',
        'premises', 'rent', 'security deposit', 'lease term'
    ]
    lease_score = sum(1 for indicator in lease_indicators if indicator in text_lower)
    
    # Partnership agreement indicators
    partnership_indicators = [
        'partnership agreement', 'partners', 'partnership', 'capital contributions',
        'profit sharing', 'partnership structure'
    ]
    partnership_score = sum(1 for indicator in partnership_indicators if indicator in text_lower)
    
    # Purchase agreement indicators
    purchase_indicators = [
        'purchase agreement', 'purchase price', 'buyer', 'seller',
        'goods', 'products', 'delivery terms'
    ]
    purchase_score = sum(1 for indicator in purchase_indicators if indicator in text_lower)
    
    # Consulting agreement indicators
    consulting_indicators = [
        'consulting agreement', 'consultant', 'consulting services',
        'independent contractor', 'consulting fees'
    ]
    consulting_score = sum(1 for indicator in consulting_indicators if indicator in text_lower)
    
    # Licensing agreement indicators
    licensing_indicators = [
        'licensing agreement', 'license', 'licensor', 'licensee',
        'royalties', 'intellectual property license'
    ]
    licensing_score = sum(1 for indicator in licensing_indicators if indicator in text_lower)
    
    # Privacy policy indicators
    privacy_indicators = [
        'privacy policy', 'data collection', 'personal information',
        'data protection', 'privacy rights'
    ]
    privacy_score = sum(1 for indicator in privacy_indicators if indicator in text_lower)
    
    # Terms of service indicators
    tos_indicators = [
        'terms of service', 'terms and conditions', 'user agreement',
        'service terms', 'acceptable use'
    ]
    tos_score = sum(1 for indicator in tos_indicators if indicator in text_lower)
    
    # Find the highest scoring document type
    scores = {
        'nda': nda_score,
        'employment_contract': employment_score,
        'service_agreement': service_score,
        'lease_agreement': lease_score,
        'partnership_agreement': partnership_score,
        'purchase_agreement': purchase_score,
        'consulting_agreement': consulting_score,
        'licensing_agreement': licensing_score,
        'privacy_policy': privacy_score,
        'terms_of_service': tos_score
    }
    
    # Get the document type with the highest score
    detected_type = max(scores, key=scores.get)
    
    # Only return the detected type if it has a reasonable score
    if scores[detected_type] >= 2:  # At least 2 indicators found
        return detected_type
    else:
        return 'other'  # Default to 'other' if no clear indicators

def parse_pdf(pdf_path):
    """Parse a PDF file and extract its content"""
    start_timer('pdf_parsing')
    try:
        reader = PdfReader(pdf_path)
        content = []
        
        for page in reader.pages:
            content.append(page.extract_text())
        
        text_content = '\n'.join(content)
        end_timer('pdf_parsing')
        return text_content
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        end_timer('pdf_parsing')
        return None
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        end_timer('pdf_parsing')
        return None

def analyze_legal_compliance_enhanced(text, document_type, clauses_data, use_legal_bert=True):
    """Enhanced legal compliance analysis using Legal-BERT semantic understanding"""
    start_timer('compliance_analysis')
    
    if not text or not document_type or not clauses_data:
        end_timer('compliance_analysis')
        return {
            'compliance_percentage': 0.00,
            'missing_clauses': 'ABCDE',
            'present_clauses': '',
            'partial_clauses': '',
            'mandatory_missing': 'ABC',
            'error': 'Missing text, document type, or clauses data',
            'riskScore': 100.00,
            'missingMandatoryCount': 3,
            'sensitivityLevel': 3,
            'clauseCompletenessScore': 0.00,
            'hasCriticalGaps': 1,
            'documentConfidenceScore': 0.00,
            'hasSalaryOrCompensation': 0
        }
    
    # Add document type validation
    detected_type = detect_document_type(text)
    if detected_type != document_type:
        print(f"   WARNING: Document appears to be a {detected_type.upper()}, not {document_type.upper()}")
        print(f"   Detected indicators: {detected_type}")
        print(f"   Requested analysis: {document_type}")
        print("   Consider re-running with correct document type for accurate results")
    
    # Extract clause weights and required clauses from loaded data
    NDA_CLAUSE_WEIGHTS = clauses_data.get('nda_CLAUSE_WEIGHTS', {})
    REQUIRED_CLAUSES = clauses_data.get('REQUIRED_CLAUSES', {})
    
    # Map document type to letter code
    doc_type_mapping = {
        'nda': 'A',
        'employment_contract': 'B',
        'service_agreement': 'C',
        'lease_agreement': 'D',
        'partnership_agreement': 'E',
        'purchase_agreement': 'F',
        'consulting_agreement': 'G',
        'licensing_agreement': 'H',
        'privacy_policy': 'I',
        'terms_of_service': 'J',
        'other': 'K'
    }
    
    document_type = document_type.lower()
    doc_type_letter = doc_type_mapping.get(document_type, 'K')
    
    if document_type not in REQUIRED_CLAUSES:
        end_timer('compliance_analysis')
        return {
            'compliance_percentage': 0.00,
            'missing_clauses': 'ABCDE',
            'present_clauses': '',
            'partial_clauses': '',
            'mandatory_missing': 'ABC',
            'error': f'Unknown document type: {document_type}',
            'riskScore': 100.00,
            'missingMandatoryCount': 3,
            'sensitivityLevel': 3,
            'clauseCompletenessScore': 0.00,
            'hasCriticalGaps': 1,
            'documentConfidenceScore': 0.00,
            'hasSalaryOrCompensation': 0
        }
    
    # Convert text to lowercase for matching
    text_lower = text.lower()
    
    # Initialize results
    present_clauses = set()
    missing_clauses = set()
    partial_clauses = set()
    mandatory_missing = set()
    total_weight = 0
    weighted_score = 0
    
    # Analysis statistics
    analysis_stats = {
        'total_clauses_analyzed': 0,
        'exact_matches_found': 0,
        'semantic_matches_found': 0,
        'sentences_processed': 0
    }
    
    # Check for salary/compensation terms (only relevant for employment contracts)
    salary_terms = ['salary', 'compensation', 'pay', 'remuneration', 'wage', 'bonus', 'commission']
    has_salary = any(term in text_lower for term in salary_terms) if document_type == 'employment_contract' else False
    
    # Get clauses for this document type
    clauses = REQUIRED_CLAUSES[document_type]
    
    # Create letter mapping for clauses
    if document_type in ['nda', 'employment_contract']:
        max_clauses = 10  # A-J
    else:
        max_clauses = 5   # A-E
    
    # Map first max_clauses to letters
    clause_letters = {}
    clause_list = list(clauses.keys())[:max_clauses]  # Take only first max_clauses
    for i, name in enumerate(clause_list):
        clause_letters[name] = chr(65 + i)
    
    # Initialize all possible letters as missing
    all_letters = set(chr(65 + i) for i in range(len(clause_list)))
    missing_clauses = all_letters.copy()
    
    # Analyze each clause with enhanced semantic understanding
    for clause_name, clause_info in clauses.items():
        if clause_name not in clause_letters:
            continue
            
        analysis_stats['total_clauses_analyzed'] += 1
        letter = clause_letters[clause_name]
        weight = NDA_CLAUSE_WEIGHTS[clause_info["weight"]]
        total_weight += weight
        
        # Enhanced clause detection with Legal-BERT
        exact_matches, semantic_matches, sentences_processed = analyze_clause_with_semantics(
            text, clause_info, document_type, use_legal_bert
        )
        
        # Update statistics
        if exact_matches:
            analysis_stats['exact_matches_found'] += len(exact_matches)
        if semantic_matches:
            analysis_stats['semantic_matches_found'] += len(semantic_matches)
        analysis_stats['sentences_processed'] += sentences_processed  # Add sentence count
        
        # Determine clause presence based on both exact and semantic matches
        if exact_matches:
            if len(exact_matches) >= 2:
                present_clauses.add(letter)
                weighted_score += weight
                print(f"✓ Clause {letter} ({clause_name}) - EXACT MATCH: {exact_matches}")
            else:
                partial_clauses.add(letter)
                weighted_score += weight * 0.5
                print(f"~ Clause {letter} ({clause_name}) - PARTIAL EXACT: {exact_matches}")
        elif semantic_matches:
            # Use semantic matches with confidence scoring
            best_similarity = max(sm['similarity'] for sm in semantic_matches)
            if best_similarity >= 0.6:
                present_clauses.add(letter)
                weighted_score += weight
                print(f"✓ Clause {letter} ({clause_name}) - SEMANTIC MATCH: {best_similarity:.3f}")
            elif best_similarity >= 0.4:
                partial_clauses.add(letter)
                weighted_score += weight * 0.5
                print(f"~ Clause {letter} ({clause_name}) - PARTIAL SEMANTIC: {best_similarity:.3f}")
        
        # Update missing clauses tracking
        if letter in present_clauses or letter in partial_clauses:
            if letter in missing_clauses:
                missing_clauses.remove(letter)
        
        if clause_info["mandatory"] and letter not in present_clauses:
            mandatory_missing.add(letter)
    
    # Store analysis statistics
    performance_metrics['analysis_stats'] = analysis_stats
    
    # Calculate final percentage
    compliance_percentage = (weighted_score / total_weight * 100) if total_weight > 0 else 0
    
    # Calculate missing mandatory count
    missing_mandatory_count = len(mandatory_missing)
    
    # Calculate risk score (0-100, higher is riskier)
    risk_factors = [
        missing_mandatory_count * 20,  # Each missing mandatory clause adds 20 points
        (100 - compliance_percentage) * 0.5,  # Lower compliance adds risk
        50 if has_salary and document_type == 'employment_contract' else 0,  # Presence of salary information increases risk only for employment contracts
    ]
    risk_score = min(100, sum(risk_factors) / len(risk_factors))
    
    # Determine sensitivity level (1: low, 2: medium, 3: high)
    sensitivity_level = 3 if risk_score > 70 else (2 if risk_score > 40 else 1)
    
    # Calculate clause completeness
    total_clauses = len(clause_letters)
    complete_clauses = len(present_clauses)
    partial_weight = len(partial_clauses) * 0.5
    clause_completeness = ((complete_clauses + partial_weight) / total_clauses * 100) if total_clauses > 0 else 0
    
    # Check for critical gaps
    critical_clauses = set('ABC')  # First three clauses are always critical
    has_critical_gaps = bool(critical_clauses & missing_clauses)  # Only count fully missing clauses
    
    # Calculate document confidence score (0-1)
    confidence_factors = [
        compliance_percentage / 100,  # Base on compliance
        1 - (missing_mandatory_count / max(1, len([c for c in clauses.values() if c['mandatory']]))),  # Mandatory clauses
        0 if has_critical_gaps else 1,  # Critical clauses presence
    ]
    document_confidence = max(0, min(1, sum(confidence_factors) / len(confidence_factors)))
    
    # Convert sets to sorted strings
    present_str = ''.join(sorted(present_clauses))
    missing_str = ''.join(sorted(missing_clauses))
    partial_str = ''.join(sorted(partial_clauses))
    mandatory_missing_str = ''.join(sorted(mandatory_missing))
    
    end_timer('compliance_analysis')
    
    return {
        'compliance_percentage': round(compliance_percentage, 2),
        'missing_clauses': missing_str,
        'present_clauses': present_str,
        'partial_clauses': partial_str,
        'mandatory_missing': mandatory_missing_str,
        'document_type': doc_type_letter,
        'riskScore': round(risk_score, 2),
        'missingMandatoryCount': missing_mandatory_count,
        'sensitivityLevel': sensitivity_level,
        'clauseCompletenessScore': round(clause_completeness, 2),
        'hasCriticalGaps': 1 if has_critical_gaps else 0,
        'documentConfidenceScore': round(document_confidence, 2),
        'hasSalaryOrCompensation': 1 if has_salary else 0
    }

def print_performance_metrics():
    """Print comprehensive performance metrics"""
    print("\n" + "="*80)
    print("PERFORMANCE METRICS & BENCHMARKING")
    print("="*80)
    
    # System Information
    print("SYSTEM INFORMATION:")
    system_info = performance_metrics['system_info']
    print(f"  Platform: {system_info.get('platform', 'N/A')}")
    print(f"  Processor: {system_info.get('processor', 'N/A')}")
    print(f"  Python Version: {system_info.get('python_version', 'N/A')}")
    print(f"  CPU Cores: {system_info.get('cpu_count', 'N/A')}")
    print(f"  Total Memory: {system_info.get('memory_total_gb', 'N/A')} GB")
    print(f"  PyTorch Version: {system_info.get('torch_version', 'N/A')}")
    print(f"  CUDA Available: {system_info.get('cuda_available', 'N/A')}")
    
    if system_info.get('cuda_available'):
        print(f"  CUDA Devices: {system_info.get('cuda_device_count', 'N/A')}")
        print(f"  GPU: {system_info.get('cuda_device_name', 'N/A')}")
        print(f"  GPU Memory: {system_info.get('gpu_memory_total_gb', 'N/A')} GB")
    
    # Model Information
    print("\nMODEL INFORMATION:")
    model_info = performance_metrics['model_info']
    if model_info:
        print(f"  Model: {model_info.get('model_name', 'N/A')}")
        print(f"  Device: {model_info.get('device', 'N/A')}")
        print(f"  Parameters: {model_info.get('model_parameters', 'N/A'):,}")
        print(f"  Model Size: {model_info.get('model_size_mb', 'N/A'):.1f} MB")
    
    # Timing Information
    print("\nTIMING BREAKDOWN:")
    timing = performance_metrics['timing']
    total_time = 0
    
    for operation, metrics in timing.items():
        if 'duration_seconds' in metrics:
            duration = metrics['duration_seconds']
            total_time += duration
            memory_delta = metrics.get('memory_delta_mb', 0)
            print(f"  {operation.replace('_', ' ').title()}: {duration:.3f}s (Memory: {memory_delta:+.1f} MB)")
    
    print(f"\n  Total Analysis Time: {total_time:.3f}s")
    
    # Memory Information
    print("\nMEMORY USAGE:")
    final_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
    print(f"  Final Memory Usage: {final_memory:.1f} MB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
        print(f"  GPU Memory Allocated: {gpu_memory:.1f} MB")
    
    # Analysis Statistics
    print("\nANALYSIS STATISTICS:")
    analysis_stats = performance_metrics['analysis_stats']
    if analysis_stats:
        print(f"  Clauses Analyzed: {analysis_stats.get('total_clauses_analyzed', 0)}")
        print(f"  Exact Matches: {analysis_stats.get('exact_matches_found', 0)}")
        print(f"  Semantic Matches: {analysis_stats.get('semantic_matches_found', 0)}")
        print(f"  Sentences Processed: {analysis_stats.get('sentences_processed', 0)}")
    
    # Performance Recommendations
    print("\nPERFORMANCE RECOMMENDATIONS:")
    if total_time > 30:
        print("     Analysis time is high. Consider:")
        print("     - Using GPU acceleration")
        print("     - Reducing sentence analysis limit")
        print("     - Using fallback analysis mode")
    elif total_time < 5:
        print("     Analysis time is excellent")
    
    if final_memory > 2000:  # 2GB
        print("     High memory usage. Consider:")
        print("     - Using smaller model")
        print("     - Processing in batches")
    
    print("="*80)

def print_analysis_results(results, document_type, pdf_path, translations):
    """Print formatted analysis results"""
    print("\n" + "="*60)
    print("LEGAL DOCUMENT COMPLIANCE ANALYSIS (Legal-BERT Enhanced)")
    print("="*60)
    print(f"Document: {pdf_path}")
    print(f"Document Type: {document_type.upper()}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    if 'error' in results and results['error']:
        print(f"ERROR: {results['error']}")
        return
    
    # Compliance Score
    compliance = results['compliance_percentage']
    if compliance >= 80:
        status = "EXCELLENT"
    elif compliance >= 60:
        status = "GOOD"
    elif compliance >= 40:
        status = "FAIR"
    else:
        status = "POOR"
    
    print(f"COMPLIANCE SCORE: {compliance}% ({status})")
    
    # Risk Assessment
    risk_score = results['riskScore']
    if risk_score <= 30:
        risk_level = "LOW"
    elif risk_score <= 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    print(f"RISK SCORE: {risk_score}/100 ({risk_level})")
    
    # Sensitivity Level
    sensitivity = results['sensitivityLevel']
    sensitivity_text = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}[sensitivity]
    print(f"SENSITIVITY LEVEL: {sensitivity_text}")
    
    # Document Confidence
    confidence = results['documentConfidenceScore']
    print(f"DOCUMENT CONFIDENCE: {confidence:.2f}")
    
    print("-"*60)
    
    # Clause Analysis with translations
    print("CLAUSE ANALYSIS:")
    
    # Present clauses
    present_clauses = results['present_clauses']
    if present_clauses:
        translated_present = translate_clause_letters(present_clauses, document_type, translations)
        print(f"   Present Clauses: {translated_present}")
    else:
        print("   Present Clauses: None")
    
    # Partial clauses
    partial_clauses = results['partial_clauses']
    if partial_clauses:
        translated_partial = translate_clause_letters(partial_clauses, document_type, translations)
        print(f"   Partial Clauses: {translated_partial}")
    else:
        print("   Partial Clauses: None")
    
    # Missing clauses
    missing_clauses = results['missing_clauses']
    if missing_clauses:
        translated_missing = translate_clause_letters(missing_clauses, document_type, translations)
        print(f"   Missing Clauses: {translated_missing}")
    else:
        print("   Missing Clauses: None")
    
    # Missing mandatory clauses
    mandatory_missing = results['mandatory_missing']
    if mandatory_missing:
        translated_mandatory = translate_clause_letters(mandatory_missing, document_type, translations)
        print(f"   Missing Mandatory: {translated_mandatory}")
    else:
        print("   Missing Mandatory: None")
    
    # Critical Gaps
    if results['hasCriticalGaps']:
        print("   CRITICAL GAPS DETECTED!")
    else:
        print("   No critical gaps detected")
    
    # Salary/Compensation Detection
    if results['hasSalaryOrCompensation']:
        print("   Salary/Compensation information detected")
    
    print("-"*60)
    
    # Recommendations
    print("RECOMMENDATIONS:")
    if compliance < 60:
        print("   • Document needs significant improvements")
        print("   • Consider adding missing mandatory clauses")
    elif compliance < 80:
        print("   • Document is acceptable but could be improved")
        print("   • Review missing optional clauses")
    else:
        print("   • Document appears to be well-structured")
        print("   • Minor improvements may be optional")
    
    if results['missingMandatoryCount'] > 0:
        print(f"   • Add {results['missingMandatoryCount']} missing mandatory clause(s)")
    
    if results['hasCriticalGaps']:
        print("   • Address critical gaps immediately")
    
    print("="*60)

def main():
    """Main function to run the legal document analyzer"""
    # Initialize performance tracking
    start_timer('total_execution')
    performance_metrics['system_info'] = get_system_info()
    
    parser = argparse.ArgumentParser(
        description='Analyze legal document compliance from PDF file using Legal-BERT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py document.pdf --type nda
  python app.py contract.pdf --type employment_contract
  python app.py agreement.pdf --type service_agreement
        """
    )
    
    parser.add_argument(
        'pdf_path',
        help='Path to the PDF document to analyze'
    )
    
    parser.add_argument(
        '--type', '-t',
        choices=[
            'nda', 'employment_contract', 'service_agreement', 'lease_agreement',
            'partnership_agreement', 'purchase_agreement', 'consulting_agreement',
            'licensing_agreement', 'privacy_policy', 'terms_of_service', 'other'
        ],
        default='other',
        help='Type of legal document (default: other)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for JSON results (optional)'
    )
    
    parser.add_argument(
        '--no-legal-bert', '-n',
        action='store_true',
        help='Disable Legal-BERT and use fallback analysis'
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Show detailed performance metrics'
    )
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found at '{args.pdf_path}'")
        return 1
    
    # Initialize models
    use_legal_bert = not args.no_legal_bert
    if use_legal_bert:
        legal_bert_loaded = load_legal_bert()
        if not legal_bert_loaded:
            use_legal_bert = False
            print("Falling back to spaCy analysis...")
    
    # Always load spaCy as fallback
    load_spacy_model()
    
    # Load legal clauses
    print("Loading legal clause definitions...")
    clauses_data = load_legal_clauses()
    if not clauses_data:
        return 1
    
    # Load clause translations
    print("Loading clause translations...")
    translations = load_clause_translations()
    if not translations:
        return 1
    
    # Parse PDF
    print(f"Parsing PDF: {args.pdf_path}")
    text_content = parse_pdf(args.pdf_path)
    if not text_content:
        return 1
    
    print(f"Extracted {len(text_content)} characters of text")
    
    # Analyze compliance with enhanced model
    analysis_type = "Legal-BERT" if use_legal_bert else "spaCy"
    print(f"Analyzing compliance for {args.type} document using {analysis_type}...")
    results = analyze_legal_compliance_enhanced(text_content, args.type, clauses_data, use_legal_bert)
    
    # Print results
    print_analysis_results(results, args.type, args.pdf_path, translations)
    
    # Print performance metrics if requested
    if args.benchmark:
        print_performance_metrics()
    
    # Save results to JSON if requested
    if args.output:
        try:
            # Add metadata to results
            output_data = {
                'metadata': {
                    'pdf_path': args.pdf_path,
                    'document_type': args.type,
                    'analysis_date': datetime.now().isoformat(),
                    'text_length': len(text_content),
                    'analysis_model': analysis_type
                },
                'results': results,
                'performance_metrics': performance_metrics
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    end_timer('total_execution')
    return 0

if __name__ == "__main__":
    exit(main()) 