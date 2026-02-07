# Data Contracts Assignment - Sahil Kamboj (B23CS1061)

[![Validation](https://img.shields.io/badge/validation-passing-brightgreen)](https://github.com/sps1001/MLOps-Sahilpreet_Singh-B23CS1061)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)

## 📌 Overview

Implementation of data contracts for four real-world scenarios following the Open Data Contract Standard (ODCS). This project demonstrates data quality enforcement, schema governance, and pipeline reliability patterns.

### Scenarios

1. **Ride-Sharing** - Logical schema mapping with PII governance
2. **E-Commerce** - Enum validation and invalid code rejection
3. **IoT Sensors** - Range-based quality checks for sensor data
4. **FinTech** - Regex pattern enforcement with circuit breakers

## 🚀 Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Validate
yamllint datacontracts/
python validate_contracts.py
```

## 📁 Structure

```
├── datacontracts/           # YAML data contracts (4 files)
├── validate_contracts.py    # Validation script
├── README.md
├── report.tex
├── report.pdf
└── requirements.txt
```

## ✅ Validation

All contracts pass:

- ✓ YAML syntax (0 errors)
- ✓ ODCS structure compliance
- ✓ Scenario-specific requirements (18/18 checks)

## 📄 Documentation

- **Report**: [report.pdf](report.pdf)
- **Repository**: [GitHub](https://github.com/sps1001/MLOps-Sahilpreet_Singh-B23CS1061/tree/Assignment-2)

## 👤 Author

**Sahil Kamboj** (B23CS1061)  
ML-DL-Ops (CSL7120) • IIT Jodhpur
