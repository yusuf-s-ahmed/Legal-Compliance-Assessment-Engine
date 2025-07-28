# Legal Compliance Assessment Engine

This legal document analysis engine powered by Legal-BERT, delivers semantic understanding of legal clauses and compliance requirements. Built with advanced transformer architecture specifically trained on legal corpora, it provides deep contextual analysis that transcends traditional keyword matching approaches.

The engine supports comprehensive analysis across multiple document types including Non-Disclosure Agreements (NDAs), employment contracts, lease agreements, service agreements, and partnership contracts. It leverages Legal-BERT's specialised legal-domain knowledge combined with spaCy's advanced natural language processing for precise sentence segmentation and semantic similarity scoring.

The system employs a hybrid approach combining Legal-BERT's semantic embeddings with spaCy's linguistic analysis, enabling precise clause identification through cosine similarity scoring and contextual understanding. The FastAPI-based REST API ensures high-performance, scalable document processing with persistent model loading for optimal throughput.

This engine represents a significant advancement over traditional legal document analysis tools, providing the depth and accuracy required for enterprise legal compliance operations while maintaining the performance and scalability needed for high-volume document processing environments.

## Table of Contents

- [Quick start](#quick-start)
- [Usage](#usage)
- [Output](#output)
- [Command Usage Example](#command-usage-example)
- [Supported Document Types](#supported-document-types)
- [Benchmarking Details](#benchmarking-details)
- [Architecture](#architecture)
- [Attribution](#attribution)

## Quick Start

1. Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Or download the packages independently
```
PyPDF2
spacy
torch
transformers
scikit-learn
fastapi
uvicorn
python-multipart
```

## Usage

1. Run the API Server
```
python api_server.py
```

2. Call the API Using Command Prompt


```
# Health check

curl http://localhost:8000/health
```

```
API endpoint commands (using /analyse endpoint)

curl -X POST "http://localhost:8000/analyse" -F "file=@document.pdf" -F "document_type=nda" -F "enable_benchmark=true"
curl -X POST "http://localhost:8000/analyse" -F "file=@document2.pdf" -F "document_type=lease_agreement" -F "enable_benchmark=true"
curl -X POST "http://localhost:8000/analyse" -F "file=@document3.pdf" -F "document_type=service_agreement" -F "enable_benchmark=true"
```

3. Or optionally, analyse the main application directly with benchmarking
```
python app.py document.pdf --type nda --benchmark
```

## Output 
```
Loading Legal-BERT model...
Legal-BERT loaded successfully on cpu
Loaded spaCy model for fallback analysis
Loading legal clause definitions...
Loading clause translations...
Parsing PDF: document.pdf
Extracted 6803 characters of text

Analysing compliance for nda document using Legal-BERT...

✓ Clause A (definition of confidential information) - EXACT MATCH: ['information disclosed', 'confidential information', 'confidential information does not include']
✓ Clause B (scope of confidentiality) - EXACT MATCH: ['receiving party shall', 'obligations of confidentiality']
✓ Clause C (return of confidential materials) - SEMANTIC MATCH: 0.970
✓ Clause D (term of confidentiality) - SEMANTIC MATCH: 0.921
~ Clause E (permitted disclosures) - PARTIAL EXACT: ['may disclose']
~ Clause F (remedies for breach) - PARTIAL EXACT: ['indemnification']
✓ Clause G (jurisdiction) - EXACT MATCH: ['governing law', 'jurisdiction']
✓ Clause H (exclusions from confidential information) - EXACT MATCH: ['confidential information does not include', 'not include']
~ Clause I (survival clause) - PARTIAL EXACT: ['survive the']
✓ Clause J (non-circumvention) - SEMANTIC MATCH: 0.929

LEGAL DOCUMENT COMPLIANCE ANALYSIS (Legal-BERT Enhanced)

Document: document.pdf
Document Type: NDA
Analysis Date: 2025-07-26 17:26:43

COMPLIANCE SCORE: 86.71% (EXCELLENT)
RISK SCORE: 2.22/100 (LOW)
SENSITIVITY LEVEL: LOW
DOCUMENT CONFIDENCE: 0.96

CLAUSE ANALYSIS:
   Present Clauses: A: definition of confidential information, B: scope of confidentiality, C: return of confidential materials, D: term of confidentiality, G: jurisdiction, H: exclusions from confidential information, J: non-circumvention
   Partial Clauses: E: permitted disclosures, F: remedies for breach, I: survival clause
   Missing Clauses: None
   Missing Mandatory: None
   No critical gaps detected

RECOMMENDATIONS:
   • Document appears to be well-structured
   • Minor improvements may be optional

PERFORMANCE METRICS & BENCHMARKING

SYSTEM INFORMATION:
  Platform: Windows-10-10.0.26100-SP0
  Processor: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
  Python Version: 3.10.0
  CPU Cores: 16
  Total Memory: 15.73 GB
  PyTorch Version: 2.7.1+cpu
  CUDA Available: False

MODEL INFORMATION:
  Model: nlpaueb/legal-bert-base-uncased
  Device: cpu
  Parameters: 109,482,240
  Model Size: 417.6 MB

TIMING BREAKDOWN:
  Legal Bert Loading: 3.916s (Memory: +451.0 MB)
  Tokenizer Loading: 0.658s (Memory: +12.5 MB)
  Model Loading: 3.252s (Memory: +438.5 MB)
  Spacy Loading: 0.411s (Memory: +46.9 MB)
  Clauses Loading: 0.000s (Memory: +0.0 MB)
  Translations Loading: 0.000s (Memory: +0.0 MB)
  Pdf Parsing: 0.006s (Memory: +0.1 MB)
  Compliance Analysis: 15.118s (Memory: +11.3 MB)
  Semantic Similarity: 0.207s (Memory: -1.2 MB)
  Embedding Generation: 0.090s (Memory: +0.0 MB)

  Total Analysis Time: 23.657s

MEMORY USAGE:
  Final Memory Usage: 859.8 MB

ANALYSIS STATISTICS:
  Clauses Analyzed: 10
  Exact Matches: 12
  Semantic Matches: 84
  Sentences Processed: 90
```

## Command Usage Example
```
python app.py <document.pdf> --type <document_type>
```

## Supported Document Types
- Non-Disclosure Agreements (NDA)
- Employment Contracts
- Service Agreements
- Lease Agreements
- Partnership Agreements
- Purchase Agreements
- Consulting Agreements
- Licensing Agreements
- Privacy Policies
- Terms of Service
- Performance Metrics

## The system provides detailed benchmarking including:

- System information (CPU, GPU, memory)
- Model loading and inference times
- Memory usage tracking
- Analysis statistics (clauses analysed, matches found)
- Performance recommendations

## Architecture
- Legal-BERT: Specialised legal domain transformer model
- spaCy: Natural language processing and sentence segmentation
- PyTorch: Deep learning framework for model inference
- JSON Configuration: Flexible clause definitions and translations

## Attribution
This product includes Legal-BERT, developed by Ioannis Chalkidis et al. at the National and Kapodistrian University of Athens, released under the MIT License.
https://huggingface.co/nlpaueb/legal-bert-base-uncased
