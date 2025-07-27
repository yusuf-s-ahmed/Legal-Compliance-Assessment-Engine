#!/usr/bin/env python3
"""
Test script for the Legal Compliance Analytics API
"""

import requests
import json

def test_api():
    """Test the API with a sample document"""
    
    # API endpoint
    url = "http://localhost:8000/analyze"
    
    # Test file
    files = {
        'file': ('document.pdf', open('document.pdf', 'rb'), 'application/pdf')
    }
    
    # Parameters
    data = {
        'document_type': 'nda',
        'enable_benchmark': 'true'
    }
    
    try:
        print("Sending request to API...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("Analysis completed successfully!")
            print(f"Compliance Score: {result.get('compliance_score', 'N/A')}%")
            print(f"Risk Score: {result.get('risk_score', 'N/A')}/100")
            print(f"Document Confidence: {result.get('document_confidence', 'N/A')}")
            
            # Save full result to file
            with open('api_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("Full result saved to api_result.json")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api() 