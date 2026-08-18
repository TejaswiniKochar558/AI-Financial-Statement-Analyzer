import streamlit as st
import pandas as pd
import json
import os
import time
import pdfplumber
import torch
import re
from PIL import Image, ImageChops
from pdf2image import convert_from_path
from transformers import DonutProcessor, VisionEncoderDecoderModel
from google import genai
from google.genai import types
import plotly.express as px
import plotly.graph_objects as go



# =========================
# 1. Configuration & Setup
# =========================

st.set_page_config(
    page_title="FinExtract AI Dashboard",
    page_icon="腸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()

DONUT_MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Backend Logic
# =========================

@st.cache_resource
def load_donut_model():
    """Loads model once and caches it in memory."""
    processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return processor, model

def query_gemini(text_input, schema_def):
    """Helper to query Gemini with retry logic and model fallbacks."""
    # Initialize client
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    sys_prompt = """
    Extract all bank transactions from the provided text.
    - Infer 'category' (e.g., Food, Transport, Transfer, Subscription, Salary) from the description.
    - Ensure 'debit' and 'credit' are numbers (no currency symbols).
    - If a transaction is a transfer, categorize it as 'Transfer'.
    - Output strictly valid JSON.
    """
    
    full_prompt = f"{sys_prompt}\n\n--- INPUT DATA ---\n{text_input}"
    
    # List of models to try in order (Flash is faster/cheaper, Pro is fallback)
    models_to_try = ['gemini-1.5-flash-001', 'gemini-1.5-flash-002', 'gemini-1.5-pro-001']
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name, 
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_def,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            # If it's a 404 (Model Not Found), continue to next model in list
            if "404" in str(e) or "NOT_FOUND" in str(e):
                continue
            
            # If it's a real error (like parsing), log it and stop
            st.error(f"Error with model {model_name}: {e}")
            return {"transactions": []}

    # If all models failed
    st.error("All model attempts failed. Please check your API Key permissions.")
    return {"transactions": []}
    

# --- Pipeline 1: PDFPlumber (Chunked) ---
def run_pipeline_1_chunked(pdf_file, schema):
    all_transactions = []
    
    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        chunk_size = 5 # Process 5 pages at a time to avoid Output Token Limits
        
        # Create a progress bar
        progress_text = "Processing PDF pages..."
        my_bar = st.progress(0, text=progress_text)
        
        for i in range(0, total_pages, chunk_size):
            # Get text for current chunk
            chunk_pages = pdf.pages[i : i + chunk_size]
            chunk_text = "\n".join([p.extract_text() for p in chunk_pages if p.extract_text()])
            
            if not chunk_text.strip():
                continue

            # Query Gemini for this chunk
            data = query_gemini(chunk_text, schema)
            transactions = data.get("transactions", [])
            all_transactions.extend(transactions)
            
            # Update Progress
            percent_complete = min((i + chunk_size) / total_pages, 1.0)
            my_bar.progress(percent_complete, text=f"Processed pages {i+1} to {min(i+chunk_size, total_pages)}")
            
            # Sleep to respect Free Tier Rate Limits (approx 15 RPM allowed)
            time.sleep(2) 

        my_bar.empty() # Clear bar when done

    return all_transactions

# --- Pipeline 2: Donut (Existing Logic - simplified for brevity) ---
def trim_whitespace(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image

def run_pipeline_2(pdf_path, schema):
    # NOTE: Running Vision models on 117 pages will take very long.
    # We leave this as is, but users should be warned.
    processor, model = load_donut_model()
    images = convert_from_path(pdf_path, dpi=150)
    
    raw_texts = []
    task_prompt = "<s_cord-v2>"
    
    progress_bar = st.progress(0, text="Analyzing Images (Donut)...")
    
    # Process images in batches for LLM, but OCR one by one
    chunk_size = 5
    current_text_batch = ""
    all_transactions = []

    for i, img in enumerate(images):
        # ... (Resize and OCR logic remains same) ...
        img = trim_whitespace(img)
        max_width = 1280
        if img.width > max_width:
            w_percent = (max_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((max_width, h_size), Image.LANCZOS)

        pixel_values = processor(img, return_tensors="pt").pixel_values.to(DEVICE)
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=1024)
        
        seq = processor.batch_decode(outputs)[0]
        seq = re.sub(r"<.*?>", "", seq).replace(processor.tokenizer.eos_token, "").strip()
        current_text_batch += seq + "\n"
        
        # Trigger Gemini every 'chunk_size' pages
        if (i + 1) % chunk_size == 0 or (i + 1) == len(images):
             data = query_gemini(current_text_batch, schema)
             all_transactions.extend(data.get("transactions", []))
             current_text_batch = "" # Reset batch
             time.sleep(2) # Rate limit

        progress_bar.progress((i + 1) / len(images))

    return all_transactions

def identify_subscriptions(df):
    """Identify potential subscriptions based on recurring amounts and descriptions."""
    if df.empty: return pd.DataFrame()
    
    # Normalize description for grouping
    df['desc_norm'] = df['description'].str.lower().str.replace(r'\d+', '', regex=True).str.strip()
    
    # Group by description and amount (Expenses only)
    expenses = df[df['debit'] > 0]
    counts = expenses.groupby(['desc_norm', 'debit']).size().reset_index(name='count')
    
    # Filter for recurring (e.g., appearing more than once)
    subs = counts[counts['count'] > 1].sort_values(by='count', ascending=False)
    
    # Join back to get original names
    subs = subs.merge(expenses[['desc_norm', 'description', 'category']].drop_duplicates('desc_norm'), on='desc_norm', how='left')
    return subs

def identify_transfers(df):
    """Filter for transfer-related transactions."""
    if df.empty: return pd.DataFrame()
    keywords = ['transfer', 'zelle', 'wire', 'sent', 'received', 'venmo', 'paypal', 'cash app']
    pattern = '|'.join(keywords)
    mask = df['description'].str.contains(pattern, case=False, na=False) | (df['category'].str.lower() == 'transfer')
    return df[mask]

# =========================
# 3. Frontend UI
# =========================

st.title("頂 Financial Statement Analyzer")
st.markdown("Extract, Compare, and Visualize your Bank Statement data using **Hybrid AI (OCR + LLM)**.")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF bank statement", type=["pdf"])
    
    st.divider()
    st.subheader("Pipeline Settings")
    enable_donut = st.checkbox("Run Pipeline 2 (Donut OCR)", value=False, help="Enable this for scanned/image-based PDFs. Note: Very slow for large files.")
    
    run_btn = st.button("噫 Process Statement", type="primary", disabled=not uploaded_file)

# --- Main Logic ---
if run_btn and uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Schema definition
    schema_structure = {
        "type": "OBJECT",
        "properties": {
            "transactions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "date": {"type": "STRING", "description": "YYYY-MM-DD"},
                        "description": {"type": "STRING"},
                        "category": {"type": "STRING", "enum": ["Food", "Transport", "Shopping", "Transfer", "Utilities", "Subscription", "Income", "Other"]},
                        "debit": {"type": "NUMBER"},
                        "credit": {"type": "NUMBER"},
                        "balance": {"type": "NUMBER"}
                    },
                    "required": ["date", "description", "category", "debit", "credit"]
                }
            }
        }
    }

    # 1. Run Pipeline 1 (Chunked)
    with st.spinner("Running Pipeline 1 (PDFPlumber + Gemini)..."):
        p1_data = run_pipeline_1_chunked(temp_path, schema_structure)

    # 2. Run Pipeline 2
    p2_data = []
    if enable_donut:
        st.warning("Pipeline 2 is running on a large file. This may take a while.")
        p2_data = run_pipeline_2(temp_path, schema_structure)
    
    st.session_state['p1_data'] = p1_data
    st.session_state['p2_data'] = p2_data
    st.session_state['processed'] = True
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

# =========================
# 4. Results & Dashboard
# =========================

if st.session_state.get('processed'):
    p1_data = st.session_state['p1_data']
    
    df = pd.DataFrame(p1_data)
    if not df.empty:
        # Data Cleaning
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['debit'] = pd.to_numeric(df['debit'], errors='coerce').fillna(0)
        df['credit'] = pd.to_numeric(df['credit'], errors='coerce').fillna(0)
        df['amount'] = df.apply(lambda x: x['credit'] if x['credit'] > 0 else -x['debit'], axis=1)
    
    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["投 Analytics Dashboard", "使 Subscriptions & Transfers", "統 Data Table"])
    
    # TAB 1: Overview
    with tab1:
        if df.empty:
            st.warning("No data extracted.")
        else:
            # Date Filter
            min_d, max_d = df['date'].min(), df['date'].max()
            c1, c2 = st.columns(2)
            s_date = c1.date_input("Start", min_d)
            e_date = c2.date_input("End", max_d)
            
            mask = (df['date'] >= pd.to_datetime(s_date)) & (df['date'] <= pd.to_datetime(e_date))
            f_df = df.loc[mask]

            # KPI
            inc = f_df['credit'].sum()
            exp = f_df['debit'].sum()
            net = inc - exp
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Income", f"${inc:,.2f}")
            k2.metric("Expenses", f"${exp:,.2f}")
            k3.metric("Net Flow", f"${net:,.2f}", delta_color="normal" if net > 0 else "inverse")
            k4.metric("Transactions", len(f_df))

            st.divider()
            
            # Charts
            g1, g2 = st.columns([2, 1])
            with g1:
                # Daily Spending Bar
                daily_spend = f_df[f_df['debit'] > 0].groupby('date')['debit'].sum().reset_index()
                fig_bar = px.bar(daily_spend, x='date', y='debit', title="Daily Spending Trend")
                st.plotly_chart(fig_bar, use_container_width=True)
            with g2:
                # Category Pie
                cat_spend = f_df[f_df['debit'] > 0].groupby('category')['debit'].sum().reset_index()
                fig_pie = px.pie(cat_spend, values='debit', names='category', title="Expense Breakdown", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

    # TAB 2: Insights (Subs & Transfers)
    with tab2:
        if df.empty:
            st.warning("No data.")
        else:
            col_sub, col_trans = st.columns(2)
            
            # 1. Subscriptions
            with col_sub:
                st.subheader("詩 Potential Subscriptions")
                st.caption("Recurring payments (same amount & description)")
                subs_df = identify_subscriptions(df)
                if not subs_df.empty:
                    # Calculate estimated monthly cost
                    est_monthly = subs_df['debit'].sum()
                    st.metric("Est. Recurring/Month", f"${est_monthly:,.2f}")
                    st.dataframe(
                        subs_df[['description', 'debit', 'count', 'category']].rename(columns={'debit': 'Amount', 'count': 'Occurrences'}),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No recurring payments detected.")

            # 2. Transfers
            with col_trans:
                st.subheader("款 Money Transfers")
                trans_df = identify_transfers(df)
                if not trans_df.empty:
                    in_trans = trans_df['credit'].sum()
                    out_trans = trans_df['debit'].sum()
                    
                    tc1, tc2 = st.columns(2)
                    tc1.metric("Transfers In", f"${in_trans:,.2f}")
                    tc2.metric("Transfers Out", f"${out_trans:,.2f}")
                    
                    st.dataframe(
                        trans_df[['date', 'description', 'debit', 'credit']].sort_values(by='date', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No transfer transactions found.")

    # TAB 3: Raw Data
    with tab3:
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "transactions.csv", "text/csv")