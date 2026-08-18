# AI-Financial-Statement-Analyzer
Donut + Gemini + Deep Learning + Document AI


## 📌 Overview

An AI-powered financial document analysis application that extracts
bank transactions from PDF statements and converts them into
structured financial data.

The application combines traditional PDF text extraction,
Generative AI, and Deep Learning-based document understanding.

## 🎯 Problem

Bank statements are often provided as PDF documents, making it
difficult to analyze transactions automatically.

This project automates the process of:

- Extracting transactions from bank statements
- Categorizing transactions
- Identifying income and expenses
- Detecting recurring subscriptions
- Identifying money transfers
- Visualizing financial activity

## 🤖 AI & Deep Learning

The project uses two AI pipelines.

### Pipeline 1 — PDFPlumber + Gemini

PDF → Text Extraction → Gemini → Structured Transactions

### Pipeline 2 — Donut + Gemini

PDF → Document Image → Donut Deep Learning Model → Gemini → Structured Transactions

Donut is a Transformer-based document understanding model used for
processing scanned or image-based documents.

## 🛠️ Technologies

- Python
- Streamlit
- Pandas
- PyTorch
- Hugging Face Transformers
- Donut
- Gemini
- PDFPlumber
- PDF2Image
- Plotly

## ✨ Features

- PDF bank statement upload
- AI-powered transaction extraction
- Automatic transaction categorization
- Income and expense analysis
- Net cash flow calculation
- Recurring subscription detection
- Money transfer detection
- Interactive financial dashboard
- Transaction data export to CSV

## 🔄 Architecture

Bank Statement PDF
        ↓
PDF Text Extraction / Document Image Processing
        ↓
AI Document Understanding
        ↓
Gemini Structured Extraction
        ↓
Transaction Data
        ↓
Financial Analysis
        ↓
Interactive Streamlit Dashboard

## 📊 Dashboard

The application provides:

- Income
- Expenses
- Net cash flow
- Transaction count
- Daily spending trends
- Expense category breakdown
- Recurring subscriptions
- Money transfers

## 🚀 Future Improvements

- Improve extraction accuracy
- Add financial anomaly detection
- Add spending predictions
- Add personalized financial recommendations
- Support additional document formats
- Deploy the application as a production service

## 👩‍💻 Author

**Tejaswini Kochar**

GitHub: [TejaswiniKochar558](https://github.com/TejaswiniKochar558)
