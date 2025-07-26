# Legal Compliance Analytics Engine

A sophisticated legal document compliance analyzer powered by Legal-BERT for enhanced semantic understanding of legal clauses.

## Features

- **Legal-BERT Integration**: Uses specialized legal domain model for accurate clause detection  
- **Multi-Document Support**: Analyzes NDAs, employment contracts, service agreements, and more  
- **Semantic Analysis**: Goes beyond keyword matching with contextual understanding  
- **Performance Benchmarking**: Comprehensive metrics for system optimization  
- **Risk Assessment**: Calculates compliance scores and identifies critical gaps  

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Usage

# Analyze NDA with benchmarking
```
python app.py document.pdf --type nda --benchmark
```

# Analyze other document types
```
python app.py document.pdf --type employment_contract
python app.py document.pdf --type service_agreement
```

# Supported Document Types
Non-Disclosure Agreements (NDA)
Employment Contracts
Service Agreements
Lease Agreements
Partnership Agreements
Purchase Agreements
Consulting Agreements
Licensing Agreements
Privacy Policies
Terms of Service
Performance Metrics

The system provides detailed benchmarking including:

System information (CPU, GPU, memory)
Model loading and inference times
Memory usage tracking
Analysis statistics (clauses analyzed, matches found)

Performance recommendations

Architecture
Legal-BERT: Specialized legal domain transformer model
spaCy: Natural language processing and sentence segmentation
PyTorch: Deep learning framework for model inference
JSON Configuration: Flexible clause definitions and translations

Attribution
This product includes Legal-BERT, developed by Ioannis Chalkidis et al. at the National and Kapodistrian University of Athens, released under the MIT License.
https://huggingface.co/nlpaueb/legal-bert-base-uncased
