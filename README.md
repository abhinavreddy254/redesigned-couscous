# Streamlit and visualization                         # Line 1-20 : Importing Libraries
import streamlit as st                                # Line 21-23: Seeting up Tesseract Path
import pandas as pd                                   # Line 25-110 : Wells Fargo processing function
import matplotlib.pyplot as plt                       # Line 112-184: Amex Business Processing Function
import time
import logging
                                                      # Line 186-254: Amex Checking Processing Function
# File handling and OCR                               # Line 256-313: Jan Citizens Processing Function
from pathlib import Path                              # Line 315-419: USAA Checking Function
import fitz  # PyMuPDF                                # Line 421-505: Bank of West Processing Function
import pytesseract                                    # Line 507-511 : Mapping Bank Types to Processing Function
from PIL import Image                                 # Line 513-537 : Open AI Categorization Function
                                                     # Line 539-566:CSV Transactions Processing Function
# Other utilities                                     #Line 568-640 : Streamlit App Structure
import os
from datetime import datetime
import re

# LangChain for AI-based categorization
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import numpy as np


# Set up Tesseract executable path (update this path based on your local installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\tesseract.exe"



def process_wells_fargo(pdf_file):
    structured_data = []

    # Define coordinate ranges and regex patterns specific to Wells Fargo
    scanned_ranges = {
        "Date": (240, 335),
        "Description": (350, 1280),
        "Deposits/Credits": (1300, 1748),
        "Withdrawals/Debits": (1720, 2070),
        "Ending Daily Balance": (2075, 2300),
    }
    selectable_ranges = {
        "Date": (61.5, 78.4),
        "Description": (125, 195),
        "Deposits/Credits": (400.25, 436.83),
        "Withdrawals/Debits": (458.25, 504.78),
        "Ending Daily Balance": (525, 567.12),
    }
    date_pattern = r"^(\d{1,2}/\d{1,2}|/\d{1,2})$"  # Matches full or partial dates
    amount_pattern = r"^-?\$?\d{1,3}(,\d{3})*(\.\d{2})?\$?$"  # Matches valid amounts

    def get_ranges(page_type):
        return scanned_ranges if page_type == "Scanned" else selectable_ranges

    # Helper function to infer missing months from partial dates based only on the next valid date
    def infer_date(partial_date, next_valid_date):
        day = partial_date.strip("/")
        if next_valid_date:
            next_month = int(next_valid_date.split("/")[0])
            inferred_date = f"{next_month}/{day}"
            return inferred_date
        else:
            return f"/{day}"  # Return partial if no context is available

    try:
        doc = fitz.open(pdf_file)
        print(f"PDF opened successfully. Total pages: {len(doc)}")
    except Exception as e:
        print(f"Failed to open PDF: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]  # Load the page
            print(f"Processing page {page_num + 1}...")

            # Check if the page has selectable text
            selectable_text = page.get_text("text")
            page_type = "Selectable" if selectable_text.strip() else "Scanned"
            print(f"Page {page_num + 1}: Detected as {page_type}.")

            ranges = get_ranges(page_type)
            current_row = {"Date": "", "Description": "", "Deposits/Credits": "", "Withdrawals/Debits": "", "Ending Daily Balance": ""}
            transaction_started = False
            next_valid_date = None  # Initialize the next valid date

            if page_type == "Selectable":
                page_dict = page.get_text("dict")
                for block in page_dict["blocks"]:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                x0 = span["bbox"][0]

                                # Check for the word "Totals" in the Date field
                                if ranges["Date"][0] <= x0 <= ranges["Date"][1] and text.lower() == "totals":
                                    print("Detected 'Totals' in Date field. Finalizing the last transaction.")
                                    if transaction_started:
                                        structured_data.append(current_row)
                                    return pd.DataFrame(structured_data)  # Stop processing

                                # Update the next valid date for inference
                                if ranges["Date"][0] <= x0 <= ranges["Date"][1] and re.match(r"^\d{1,2}/\d{1,2}$", text):
                                    next_valid_date = text

                                # Check which field the text belongs to
                                if ranges["Date"][0] <= x0 <= ranges["Date"][1] and re.match(date_pattern, text):
                                    if transaction_started:
                                        structured_data.append(current_row)
                                        current_row = {"Date": "", "Description": "", "Deposits/Credits": "", "Withdrawals/Debits": "", "Ending Daily Balance": ""}
                                    if text.startswith("/"):
                                        text = infer_date(text, next_valid_date)
                                    current_row["Date"] = text
                                    transaction_started = True
                                elif transaction_started and ranges["Description"][0] <= x0 <= ranges["Description"][1]:
                                    current_row["Description"] += " " + text
                                elif transaction_started and ranges["Deposits/Credits"][0] <= x0 <= ranges["Deposits/Credits"][1]:
                                    if re.match(amount_pattern, text):
                                        current_row["Deposits/Credits"] = text
                                elif transaction_started and ranges["Withdrawals/Debits"][0] <= x0 <= ranges["Withdrawals/Debits"][1]:
                                    if re.match(amount_pattern, text):
                                        current_row["Withdrawals/Debits"] = text
                                elif transaction_started and ranges["Ending Daily Balance"][0] <= x0 <= ranges["Ending Daily Balance"][1]:
                                    if re.match(amount_pattern, text):
                                        current_row["Ending Daily Balance"] = text

                if transaction_started:
                    structured_data.append(current_row)

            elif page_type == "Scanned":
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

                for i in range(len(ocr_result["text"])):
                    text = ocr_result["text"][i].strip()
                    if text:
                        x0 = ocr_result["left"][i]

                        # Check for the word "Totals" in the Date field
                        if ranges["Date"][0] <= x0 <= ranges["Date"][1] and text.lower() == "totals":
                            print("Detected 'Totals' in Date field. Finalizing the last transaction.")
                            if transaction_started:
                                structured_data.append(current_row)
                            return pd.DataFrame(structured_data)  # Stop processing

                        # Update the next valid date for inference
                        if ranges["Date"][0] <= x0 <= ranges["Date"][1] and re.match(r"^\d{1,2}/\d{1,2}$", text):
                            next_valid_date = text

                        if ranges["Date"][0] <= x0 <= ranges["Date"][1] and re.match(date_pattern, text):
                            if transaction_started:
                                structured_data.append(current_row)
                                current_row = {"Date": "", "Description": "", "Deposits/Credits": "", "Withdrawals/Debits": "", "Ending Daily Balance": ""}
                            if text.startswith("/"):
                                text = infer_date(text, next_valid_date)
                            current_row["Date"] = text
                            transaction_started = True
                        elif transaction_started and ranges["Description"][0] <= x0 <= ranges["Description"][1]:
                            current_row["Description"] += " " + text
                        elif transaction_started and ranges["Deposits/Credits"][0] <= x0 <= ranges["Deposits/Credits"][1]:
                            if re.match(amount_pattern, text) and not current_row["Deposits/Credits"]:
                                current_row["Deposits/Credits"] = text
                        elif transaction_started and ranges["Withdrawals/Debits"][0] <= x0 <= ranges["Withdrawals/Debits"][1]:
                            if re.match(amount_pattern, text) and not current_row["Withdrawals/Debits"]:
                                current_row["Withdrawals/Debits"] = text
                        elif transaction_started and ranges["Ending Daily Balance"][0] <= x0 <= ranges["Ending Daily Balance"][1]:
                            if re.match(amount_pattern, text) and not current_row["Ending Daily Balance"]:
                                current_row["Ending Daily Balance"] = text

                if transaction_started:
                    structured_data.append(current_row)

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            continue

    return pd.DataFrame(structured_data)




def process_amex_business(pdf_file):
    structured_data = []

    date_x_range = (200, 360)
    description_x_range = (400, 1100)
    amount_x_range = (2100, 3000)
    location_x_range = (1240, 1450)

    date_pattern = r"^\d{1,2}/\d{1,2}/\d{2}[*\"]?$"
    amount_pattern = r"^-?\$?\(?-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\)?$"
    end_section_keywords = ["Ending Balance", "Checks Paid Summary", "Important Information"]

    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        row = {"Date": "", "Description": "", "Amount": "", "Location": ""}
        capturing_description = False
        capturing_amount = False
        capturing_location = False

        for i in range(len(ocr_result["text"])):
            text = ocr_result["text"][i].strip()
            if text:
                x0 = ocr_result["left"][i]

                if any(keyword.lower() in text.lower() for keyword in end_section_keywords):
                    break

                if date_x_range[0] <= x0 < date_x_range[1] and re.match(date_pattern, text):
                    if row["Date"]:
                        structured_data.append(row)
                        row = {"Date": "", "Description": "", "Amount": "", "Location": ""}
                    row["Date"] = text
                    capturing_description = True
                    capturing_amount = True
                    capturing_location = True

                elif capturing_description and description_x_range[0] <= x0 < description_x_range[1]:
                    row["Description"] += " " + text

                elif capturing_amount and amount_x_range[0] <= x0 < amount_x_range[1] and re.match(amount_pattern, text):
                    if not row["Amount"]:
                        row["Amount"] = text
                        capturing_amount = False

                elif capturing_location and location_x_range[0] <= x0 < location_x_range[1]:
                    row["Location"] += " " + text

        if row["Date"]:
            structured_data.append(row)

    df_structured = pd.DataFrame(structured_data)
    return df_structured




def process_amex_checking(pdf_file):
    structured_data = []

    date_x_range = (257, 500)
    description_x_range = (557, 1400)
    credits_x_range = (1561, 1782)
    debits_x_range = (1866, 2114)
    balance_x_range = (2221, 2443)

    date_pattern = r"^\d{1,2}/\d{1,2}/\d{4}$"
    amount_pattern = r"^\(?\$?-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\)?$"
    end_section_keywords = ["Ending Balance", "Checks Paid Summary", "Important Information"]

    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        row = {"Date": "", "Description": "", "Credits": "", "Debits": "", "Ending Daily Balance": ""}
        description_header_found = False

        for i in range(len(ocr_result["text"])):
            text = ocr_result["text"][i].strip()
            if text:
                x0 = ocr_result["left"][i]

                if not description_header_found and "Description" in text:
                    description_header_found = True
                    continue

                if not description_header_found:
                    continue

                if any(keyword.lower() in text.lower() for keyword in end_section_keywords):
                    break

                if date_x_range[0] <= x0 < date_x_range[1] and re.match(date_pattern, text):
                    if row["Date"]:
                        structured_data.append(row)
                        row = {"Date": "", "Description": "", "Credits": "", "Debits": "", "Ending Daily Balance": ""}
                    row["Date"] = text

                elif description_x_range[0] <= x0 < description_x_range[1]:
                    row["Description"] += " " + text

                elif credits_x_range[0] <= x0 < credits_x_range[1] and re.match(amount_pattern, text):
                    row["Credits"] = text

                elif debits_x_range[0] <= x0 < debits_x_range[1] and re.match(amount_pattern, text):
                    row["Debits"] = text

                elif balance_x_range[0] <= x0 < balance_x_range[1] and re.match(amount_pattern, text):
                    row["Ending Daily Balance"] = text

        if row["Date"]:
            structured_data.append(row)

    df_structured = pd.DataFrame(structured_data)
    return df_structured





def process_jan_citizens_bank(pdf_file):
    structured_data = []

    date_x_range = (280, 513)
    description_x_range = (540, 840)
    debits_x_range = (1250, 1400)
    credits_x_range = (1810, 2000)
    balance_x_range = (2110, 2400)

    date_pattern = r"^\d{2}/\d{2}/\d{4}$"
    amount_pattern = r"^\(?\$?-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\)?$"

    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        row = {"Date": "", "Description": "", "Debits": "", "Credits": "", "Balance": ""}

        for i in range(len(ocr_result["text"])):
            text = ocr_result["text"][i].strip()
            if text:
                x0 = ocr_result["left"][i]

                if date_x_range[0] <= x0 < date_x_range[1] and re.match(date_pattern, text):
                    if row["Date"]:
                        structured_data.append(row)
                        row = {"Date": "", "Description": "", "Debits": "", "Credits": "", "Balance": ""}
                    row["Date"] = text

                elif description_x_range[0] <= x0 < description_x_range[1] and row["Date"] and not row["Debits"]:
                    row["Description"] += " " + text

                elif debits_x_range[0] <= x0 < debits_x_range[1] and re.match(amount_pattern, text):
                    row["Debits"] = text

                elif credits_x_range[0] <= x0 < credits_x_range[1] and re.match(amount_pattern, text):
                    row["Credits"] = text

                elif balance_x_range[0] <= x0 < balance_x_range[1] and re.match(amount_pattern, text):
                    row["Balance"] = text

        if row["Date"]:
            structured_data.append(row)

    df_structured = pd.DataFrame(structured_data)
    return df_structured


def process_usaa_checking(directory_path):
    # Helper function to load tables from directory
    def load_tables(directory_path):
        tables = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Ignore JSON files, only load tables (CSV or Excel)
            if filename.endswith(".csv"):
                table = pd.read_csv(file_path)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                table = pd.read_excel(file_path)
            else:
                continue  # Skip non-table files
            
            tables.append(table)
        return tables
    
    # Step 1: Load tables
    tables = load_tables(directory_path)
    if not tables:
        raise ValueError("No valid tables found in the directory.")
    print(f"Loaded {len(tables)} tables from directory.")
    
    # Step 2: Clean tables by removing rows below 'Confidence Scores % (Table Cell)'
    def clean_tables(tables):
        cleaned_tables = []
        for table in tables:
            # Find the index of the row containing 'Confidence Scores % (Table Cell)'
            index_to_remove = table[table.apply(lambda row: row.astype(str).str.contains("Confidence Scores % \(Table Cell\)").any(), axis=1)].index
            
            # If the index exists, remove that row and all rows below it
            if not index_to_remove.empty:
                first_index_to_remove = index_to_remove[0]  # Get the first occurrence
                table = table[:first_index_to_remove]  # Keep rows only above this index
            
            # Append the cleaned table to the list
            cleaned_tables.append(table.reset_index(drop=True))
        
        return cleaned_tables

    # Apply initial cleaning
    cleaned_tables = clean_tables(tables)
    print(f"Cleaned {len(cleaned_tables)} tables.")

    # Step 3: Remove all ' symbols from the tables
    def remove_all_symbols(tables):
        cleaned_tables = []
        for table in tables:
            # Remove all instances of the ' symbol in the entire DataFrame
            table = table.replace({"'": ""}, regex=True)
            cleaned_tables.append(table)
        return cleaned_tables

    # Apply symbol removal
    cleaned_tables = remove_all_symbols(cleaned_tables)

    # Step 4: Process each table to extract Date, Description, Debits, Credits, and Balance columns
    date_pattern = r"^\d{1,2}/\d{1,2}$"  # Define date pattern in mm/dd format

    def process_date_rows(tables):
        processed_tables = []
        for table in tables:
            # Initialize an empty DataFrame to store processed rows
            processed_table = pd.DataFrame(columns=["Date", "Description", "Debits", "Credits", "Balance"])
            rows_to_add = []  # Temporary list to collect rows before concatenating
            
            for _, row in table.iterrows():
                # Extract the date cell and check if it matches the mm/dd format
                date_cell = str(row.iloc[0]).strip()
                
                if re.match(date_pattern, date_cell):
                    # Format the date with leading zeroes if necessary
                    date_parts = date_cell.split('/')
                    month = date_parts[0].zfill(2)
                    day = date_parts[1].zfill(2)
                    formatted_date = f"{month}/{day}"
                    
                    # Extract values for other columns in the expected order
                    description = str(row.iloc[1]).strip() if len(row) > 1 else ""
                    debits = str(row.iloc[2]).strip() if len(row) > 2 else ""
                    credits = str(row.iloc[3]).strip() if len(row) > 3 else ""
                    balance = str(row.iloc[4]).strip() if len(row) > 4 else ""
                    
                    # Add the processed row to the list
                    rows_to_add.append({
                        "Date": formatted_date,
                        "Description": description,
                        "Debits": debits,
                        "Credits": credits,
                        "Balance": balance
                    })
            
            # Concatenate all rows to create the final DataFrame for each table
            processed_table = pd.concat([processed_table, pd.DataFrame(rows_to_add)], ignore_index=True)
            processed_tables.append(processed_table)
        
        return processed_tables

    # Process tables to extract date rows
    processed_tables = process_date_rows(cleaned_tables)

    # Step 5: Merge all processed tables into a single DataFrame
    final_combined_table = pd.concat(processed_tables, ignore_index=True)
    print("Final Combined Table:")
    print(final_combined_table)
    
    # Return the final combined DataFrame
    return final_combined_table


def process_bankofwest(directory_path):
    """
    Process Bank of the West tables:
    1. Load and sort tables from the specified directory.
    2. Clean tables by removing rows below 'Confidence Scores % (Table Cell)'.
    3. Remove all instances of the ' symbol.
    4. Extract rows with valid 'Date', 'Amount', and 'Description' columns with valid dates.
    5. Combine and return the final cleaned DataFrame.
    """
    def load_bank_tables_sorted(directory_path):
        tables = []
        file_list = []

        # Collect only CSV or Excel files with names like table-1, table-2
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_list.append(filename)

        # Sort files based on numeric suffix (e.g., table-1, table-2)
        sorted_files = sorted(file_list, key=lambda x: int(x.split('-')[-1].split('.')[0]))

        # Load the sorted files
        for filename in sorted_files:
            file_path = os.path.join(directory_path, filename)
            if filename.endswith(".csv"):
                table = pd.read_csv(file_path)
            elif filename.endswith((".xlsx", ".xls")):
                table = pd.read_excel(file_path)
            else:
                continue  # Skip non-table files

            tables.append(table)  # Append table
        return tables

    def clean_bank_tables(tables):
        cleaned_tables = []
        for table in tables:
            index_to_remove = table[
                table.apply(
                    lambda row: row.astype(str).str.contains("Confidence Scores % \(Table Cell\)", na=False).any(),
                    axis=1
                )
            ].index

            if not index_to_remove.empty:
                first_index_to_remove = index_to_remove[0]
                table = table.iloc[:first_index_to_remove]  # Keep rows only above this index

            cleaned_tables.append(table.reset_index(drop=True))
        return cleaned_tables

    def remove_all_symbols(tables):
        cleaned_tables = []
        for table in tables:
            table = table.replace({"'": ""}, regex=True)
            cleaned_tables.append(table)
        return cleaned_tables

    def extract_valid_data_with_dates(tables):
        extracted_data = []  # Initialize the list here
        date_pattern = r"^\d{1,2}/\d{1,2}$"  # Define mm/dd date pattern

        for table in tables:
            if "'Date" in table.columns and "'Amount" in table.columns and "'Description" in table.columns:
                valid_data = table[table["'Date"].astype(str).str.match(date_pattern, na=False)]
                valid_data = valid_data[["'Date", "'Amount", "'Description"]].dropna(how='all')
                if not valid_data.empty:
                    extracted_data.append(valid_data)  # Append the valid data to the list

        if extracted_data:
            combined_data = pd.concat(extracted_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()  # Return an empty DataFrame if nothing was found

        return combined_data

    # Step 1: Load and sort tables
    tables = load_bank_tables_sorted(directory_path)
    if not tables:
        raise ValueError("No valid CSV or Excel files found in the specified directory.")
    print(f"Loaded {len(tables)} tables from directory in sorted order.")

    # Step 2: Clean tables
    cleaned_tables = clean_bank_tables(tables)
    print(f"Cleaned {len(cleaned_tables)} tables.")

    # Step 3: Remove all symbols
    cleaned_tables = remove_all_symbols(cleaned_tables)

    # Step 4: Extract valid data with dates
    final_combined_table = extract_valid_data_with_dates(cleaned_tables)

    if final_combined_table.empty:
        print("No valid data found after processing.")
    else:
        print("Final Combined Table:")
        print(final_combined_table)

    return final_combined_table

def process_bmo_checking(directory_path):
    consolidated_data = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Load only CSV and Excel files
        if filename.endswith(".csv"):
            table = pd.read_csv(file_path)
        elif filename.endswith((".xlsx", ".xls")):
            table = pd.read_excel(file_path)
        else:
            continue  # Skip non-table files
        
        # Clean the table
        index_to_remove = table[table.apply(lambda row: row.astype(str).str.contains("Confidence Scores % \(Table Cell\)").any(), axis=1)].index
        
        if not index_to_remove.empty:
            first_index_to_remove = index_to_remove[0]
            table = table[:first_index_to_remove]  # Keep rows only above this index
        
        # Check for required columns
        required_columns = ["'Date", "'Amount", "'Description"]
        if all(col in table.columns for col in required_columns):
            table = table[required_columns].dropna(subset=required_columns)
        else:
            continue  # Skip tables missing required columns
        
        consolidated_data.append(table)

    # Return consolidated data
    if consolidated_data:
        return pd.concat(consolidated_data, ignore_index=True)  # Combine all tables into one DataFrame
    else:
        print("No valid data processed.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid data

    


import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import streamlit as st

# AWS Credentials and S3 Configuration
AWS_ACCESS_KEY = "AKIAYEKP5VJZW3NZZNOU"
AWS_SECRET_KEY = "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"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "probuckettt"

# Function to upload file to S3
def upload_to_s3(file, file_name, folder_name=None):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
        if folder_name:
            file_name = f"{folder_name}/{file_name}"
        
        s3.upload_fileobj(file, S3_BUCKET_NAME, file_name)
        s3_url = f"s3://{S3_BUCKET_NAME}/{file_name}"
        return f"File uploaded successfully to {s3_url}"
    
    except NoCredentialsError:
        return "AWS credentials not available. Please provide valid credentials."
    except PartialCredentialsError:
        return "Incomplete AWS credentials provided. Ensure both access key and secret key are specified."
    except ClientError as e:
        return f"AWS Client Error: {e.response['Error']['Message']}"
    except Exception as e:
        return f"An error occurred: {e}"













# Define mapping of bank types to their respective processing functions
bank_functions = {
    "Wells Fargo": process_wells_fargo,
    "Amex Business": process_amex_business,
    "Amex Checking": process_amex_checking,
    "Jan Citizens Bank": process_jan_citizens_bank,
    "USAA Checking": process_usaa_checking,
    "Bank Of West": process_bankofwest,
    "BMO Checking": process_bmo_checking,
}




import pyperclip
def run_automation(folder_path,file_name):
    import os
    import time
    import zipfile
    import shutil
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    logging.basicConfig(level=logging.INFO)

    # Read the folder path from the temporary file
    if not folder_path.startswith("s3://"):
        raise ValueError("Invalid folder path format. Must start with 's3://'.")
    logging.info(f"Using provided folder path: {folder_path}")


    # Initialize WebDriver with custom download folder
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)

    try:
        driver.maximize_window()
        print("Browser window maximized!")

        # Step 1: Open the URL
        url = "https://aws.amazon.com/textract/"
        driver.get(url)
        print("Amazon Textract URL opened successfully!")

        # Step 2: Close the pop-up (if any)
        try:
            close_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Close') or @aria-label='Close']"))
            )
            close_button.click()
            print("Pop-up closed successfully!")
        except Exception as e:
            print("No pop-up found or couldn't close pop-up:", e)

        # Step 3: Click the "Sign In" button
        try:
            sign_in_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'lb-txt-none') and contains(text(), 'Sign In')]"))
            )
            sign_in_button.click()
            print("Clicked on the 'Sign In' button successfully!")
        except Exception as e:
            print("Could not find or click the 'Sign In' button:", e)

        # Step 4: Enter the account ID
        try:
            account_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#account"))
            )
            account_input.clear()
            account_input.send_keys("559050238579")
            print("Account ID entered successfully!")
        except Exception as e:
            print("Could not find or interact with the account ID input box:", e)

        # Step 5: Enter the IAM username
        try:
            username_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#username.awsui_input_2rhyz_7n7ue_101"))
            )
            username_input.clear()
            username_input.send_keys("textract")
            print("IAM user name entered successfully!")
        except Exception as e:
            print("Could not find or interact with the IAM username input box:", e)

        # Step 6: Enter the password
        try:
            password_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#password.awsui_input_2rhyz_7n7ue_101"))
            )
            password_input.clear()
            password_input.send_keys("Natasha@black")
            print("Password entered successfully!")
        except Exception as e:
            print("Could not find or interact with the password input box:", e)

        # Step 7: Click the final "Sign In" button
        try:
            final_sign_in_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(@class, 'awsui_content_vjswe_1wo5s_101') and text()='Sign in']"))
            )
            final_sign_in_button.click()
            print("Clicked the 'Sign In' button successfully!")
        except Exception as e:
            print("Could not find or click the 'Sign In' button:", e)

        # Step 8: Navigate to "Amazon Textract" link in Recently Visited
        try:
            textract_recently_visited = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//div[@aria-label='Recently visited']//a[contains(text(), 'Amazon Textract')]")
                )
            )
            textract_recently_visited.click()
            print("Clicked the 'Amazon Textract' link in Recently Visited successfully!")
        except Exception as e:
            print("Could not find or click the 'Amazon Textract' link in Recently Visited:", e)

        try:
            hamburger_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[.//span[contains(@class, 'awsui_icon_h11ix_o4x4v_185')]]")
                    
                )
            )
            hamburger_button.click()
            print("Clicked the hamburger button successfully!")
        except Exception as e:
            print("Could not find or click the hamburger button:", e)

    # Step 10: Click the "Analyze Document" button
        try:
            analyze_document_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//span[contains(@class, 'awsui_link-text_eymn4_rug8v_6') and text()='Analyze Document']")
                )
            )
            analyze_document_button.click()
            print("Clicked the 'Analyze Document' button successfully!")
        except Exception as e:
            print("Could not find or click the 'Analyze Document' button:", e)

    
    # Step 12: Click the "Upload Documents" button without scrolling
        try:
            upload_documents_button = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//span[contains(@class, 'awsui_content_vjswe_1ekr7_149') and text()='Upload documents']")
                    
                )
            )
        # Use JavaScript to click the button directly
            driver.execute_script("arguments[0].click();", upload_documents_button)
            print("Clicked the 'Upload Documents' button successfully using JavaScript!")
        except Exception as e:
            print("Could not find or click the 'Upload Documents' button:", e)

# Step 13: Input the S3 path dynamically
        try:
            # Scroll to the appropriate section
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.4);")
            time.sleep(2)
            s3_input_box = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='s3://bucket/prefix/']"))
            )
    
    # Focus, clear, and input the dynamically loaded folder path
            driver.execute_script("arguments[0].focus();", s3_input_box)
            s3_input_box.clear()

    # Dynamically read the folder path from a temporary file
            # Input the folder path into the S3 input box
            s3_input_box.send_keys(folder_path)
            print(f"S3 folder path entered successfully: {folder_path}")
        except Exception as e:
            print("Could not find or interact with the S3 URL input box:", e)


    # Step 14: Scroll down slightly and click "Analyze Document - Tables"
        try:
            
        
    # Scroll slightly more to make sure the element is visible
            driver.execute_script("window.scrollBy(0, 300);")
            time.sleep(2)  # Allow the page to adjust

    # Locate and click the "Analyze Document - Tables" option
            analyze_tables_button = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "span#\\:r6p\\:-label.awsui_label_1wepg_1vmnx_143.awsui_label_13tpe_9w8pd_5")
                )
            )
            analyze_tables_button.click()
            print("Clicked on 'Analyze Document - Tables' successfully!")
        except Exception as e:
            print(f"Could not find or click 'Analyze Document - Tables': {e}")
        try:
    # Scroll down to make sure the button is visible
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Allow time for scrolling to complete

    # Locate the button by its data-testid attribute
            start_processing_button = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='start-processing-button']"))
            )

    # Click the button using JavaScript (more reliable for complex UIs)
            driver.execute_script("arguments[0].click();", start_processing_button)
            print("Clicked on 'Start Processing' successfully!")
        except Exception as e:
            print(f"Could not find or click 'Start Processing': {e}")

        try:
    # Locate the first document name
            first_document_name = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "//table/tbody/tr[1]/td[2]/div"))
            )
            print(f"First document found: {first_document_name.text}")
        except Exception as e:
            print("Document not found.")

        try:
    # Scroll down by 30% of the page height
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight * 0.3);")
            time.sleep(6)  # Allow time for the UI to adjust
            print("Scrolled down by 30% of the page height successfully!")
        except Exception as e:
            print(f"Could not scroll down the page: {e}")

        try:
    # Locate the checkbox
            first_row_checkbox = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, f"(//label[contains(@aria-label, '{file_name}')]//input[@type='checkbox'])[1]"))
            )

    # Debugging: Check visibility and enabled state
            print(f"Checkbox is displayed: {first_row_checkbox.is_displayed()}")
            print(f"Checkbox is enabled: {first_row_checkbox.is_enabled()}")

    # Highlight and click the checkbox
            driver.execute_script("arguments[0].style.border='3px solid red'", first_row_checkbox)
            driver.execute_script("arguments[0].click();", first_row_checkbox)
            print("First checkbox for the document clicked successfully!")
        except Exception as e:
            print("Could not locate or interact with the document checkbox:", e)

    # Perform actions to download file
# Perform actions to download file
        print("Waiting for 40 seconds to allow processing to complete...")
        time.sleep(100)  # Allow time for the processing to complete


# **Automated Download Process**
        try:
            
    # Wait for the button to be present
            download_button = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id=":r7i:"]/div[1]/div/div/div[1]/div/div[2]/div/div[2]/button/span')
                    )
            )
            print("Download button found!")

    # Click the download button (automatically trigger download)
            driver.execute_script("arguments[0].click();", download_button)
            print("Download button clicked successfully!")

        except Exception as e:
            print(f"Error while clicking the download button: {e}")
            driver.save_screenshot("download_button_error.png")
            print("Screenshot saved for debugging!")


            print("Waiting for 15 seconds to allow processing to complete...")
            time.sleep(15)  # Adjust this time as necessary, based on file size
    

        # Additional steps...
        print("Automation completed successfully!")

    except Exception as e:
        print(f"An error occurred during automation: {e}")

    input("Press Enter to close the browser...")

# Clean up
    driver.quit()

    print("Automation process completed up to downloading.")

    

    



import os
import zipfile
import pyperclip

def process_downloaded_file(download_dir="C:/Users/HP/Downloads/"):
    try:
        # Locate the most recent file in the download directory
        downloaded_file = max(
            [os.path.join(download_dir, f) for f in os.listdir(download_dir)], 
            key=os.path.getctime
        )

        # Check if the downloaded file is a zip file
        if downloaded_file.endswith('.zip'):
            print(f"Found zip file: {downloaded_file}")

            # Define extraction folder
            extraction_folder = downloaded_file.replace('.zip', '')

            # Unzip the file
            try:
                with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
                    zip_ref.extractall(extraction_folder)
                print(f"Unzipped the file to: {extraction_folder}")
            except zipfile.BadZipFile:
                print("Error: The downloaded file is not a valid zip file.")
                return None

            # Copy the extraction folder path to the clipboard
            extraction_folder_path = os.path.abspath(extraction_folder)
            pyperclip.copy(extraction_folder_path)
            print(f"Unzipped folder path copied to clipboard: {extraction_folder_path}")
            return extraction_folder_path
        else:
            print("No zip file found in the Downloads folder.")
            return None

    except Exception as e:
        print(f"An error occurred while processing the downloaded file: {e}")
        return None












import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

def process_and_visualize_transactions(file_path, api_key, custom_categories=None):
    """
    Process a CSV file of transactions by detecting its structure, categorizing transactions,
    and visualizing transaction totals by category. Supports custom categories.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Function to clean and format the 'Amount' columns (same as before)
        def clean_and_format_amount(df, column_name):
            if column_name in df.columns:
                df[column_name] = df[column_name].astype(str)
                df[column_name] = df[column_name].str.replace('[\$,]', '', regex=True)
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
            return df

        # Apply cleaning logic (same as before)
        if 'Deposits/Credits' in df.columns and 'Withdrawals/Debits' in df.columns:
            df = clean_and_format_amount(df, 'Deposits/Credits')
            df = clean_and_format_amount(df, 'Withdrawals/Debits')
            df['Amount_CD'] = df.apply(lambda x: x['Deposits/Credits'] - x['Withdrawals/Debits'], axis=1)

        elif 'Debits' in df.columns and 'Credits' in df.columns:
            df = clean_and_format_amount(df, 'Debits')
            df = clean_and_format_amount(df, 'Credits')
            df['Amount_CD'] = df.apply(lambda x: x['Credits'] - x['Debits'], axis=1)

        # Case 3: Amount (used in a separate CSV format)
        elif 'Amount' in df.columns:
            df = clean_and_format_amount(df, 'Amount')
           
            # Convert all positives to negatives and negatives to positives
            df['Amount'] = df['Amount'].apply(lambda x: -x)

        # Case 4: 'Amount (another format with different naming)
        elif "'Amount" in df.columns:
            df = clean_and_format_amount(df, "'Amount")
           
            # Convert all positives to negatives and negatives to positives
            df["'Amount"] = df["'Amount"].apply(lambda x: -x)

        

        # Ensure 'Description' column exists
        if 'Description' not in df.columns:
            raise KeyError("Description column missing.")
        df['combined_text'] = df['Description'].fillna('')

        # Inner function for categorization logic
        def get_chat_category(description):
            try:
                chat = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key)
                custom_instructions = ""
                if custom_categories:
                    for category, keywords in custom_categories.items():
                        custom_instructions += f"- If the description includes keywords like {', '.join(keywords)}, categorize as '{category}'.\n"

                prompt_template = ChatPromptTemplate.from_template(
                    f"""
                    Categorize the following transaction description into appropriate categories based on these rules:
                    {custom_instructions}
                    If no rule applies, use one of these predefined categories:
                    Food & Dining, Transportation, Healthcare, Entertainment, Hotel, Shopping, Online Payments, Money Transfer, Subscription Service,
                    Fees & Charges, Clothing, Uber Eats, Repairs & Maintenance, Gas, Internet & Usage, Amazon Purchase, Spa, Groceries,
                    Carrier, Banking & Finance, Utilities, Education, Personal Services, Charity & Donations, Home & Garden, Technology,
                    Pets, Gaming, Streaming, Cinemas, Automotive, Events, Gifts, Business Expenses, Miscellaneous, Emergencies,
                    Cultural, Nature & Outdoors, Luxury, Seasonal, Social & Networking, DIY & Crafts, Delivery Services, Sales and Finance, Other.
                    Specific Instructions:
                    - Respond **only** with the category name, without any additional text, comments, or explanations.
                    - If the transaction description contains keywords like "Deposit", "eDeposit", "Add Money", or "Deposit IN Branch", categorize it as "Deposit".
                    - If the transaction description contains keywords like "ELECTRONIC DEP", categorize it as "Sales and Finance".
                    - If the transaction description is unclear or doesn't match any category, categorize it as "Other".

                    Transaction Description: '{description}'.
                    """
                )
                llm_chain = LLMChain(llm=chat, prompt=prompt_template)
                category = llm_chain.run(description=description)
                return category.strip()
            except Exception as e:
                print(f"Error querying the model: {e}")
                return "Uncategorized"

        # Categorization with custom categories prioritized
        def categorize_transaction(description):
            # Check custom categories first
            if custom_categories:
                for category, keywords in custom_categories.items():
                    if any(keyword.lower() in description.lower() for keyword in keywords):
                        return category
            # Fallback to OpenAI categorization
            return get_chat_category(description)

        df['category'] = df['combined_text'].apply(categorize_transaction)

        # Visualization Logic
        # Visualization for the 'Amount' column
        if 'Amount' in df.columns:
            transaction_totals = df.groupby('category')['Amount'].sum()
            transaction_totals = pd.to_numeric(transaction_totals, errors='coerce').fillna(0)

            plt.figure(figsize=(9, 6))
            plt.bar(
                transaction_totals.index,
                transaction_totals,
                color=['skyblue' if x > 0 else 'lightcoral' for x in transaction_totals],
                edgecolor='black'
            )
            plt.axhline(0, color='black', linewidth=1, linestyle='--')
            plt.title("Total Transaction Amount by Category", fontsize=16)
            plt.xlabel('Category', fontsize=14)
            plt.ylabel('Total Transaction Amount ($)', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.tight_layout()
            st.pyplot()  # Use st.pyplot to display the plot in Streamlit

        # Visualization for the "'Amount'" column
        if "'Amount" in df.columns:
            transaction_totals = df.groupby('category')["'Amount"].sum()
            transaction_totals = pd.to_numeric(transaction_totals, errors='coerce').fillna(0)

            plt.figure(figsize=(9, 6))
            plt.bar(
                transaction_totals.index,
                transaction_totals,
                color=['skyblue' if x > 0 else 'lightcoral' for x in transaction_totals],
                edgecolor='black'
            )
            plt.axhline(0, color='black', linewidth=1, linestyle='--')
            plt.title("Total Transaction Amount by Category", fontsize=16)
            plt.xlabel('Category', fontsize=14)
            plt.ylabel('Total Transaction Amount ($)', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.tight_layout()
            st.pyplot()  # Use st.pyplot to display the plot in Streamlit

        # Visualization for the 'Amount_CD' column
        if 'Amount_CD' in df.columns:
            transaction_totals = df.groupby('category')['Amount_CD'].sum()

            plt.figure(figsize=(10, 6))
            plt.bar(
                transaction_totals.index,
                transaction_totals,
                color='skyblue',
                edgecolor='black'
            )
            plt.title("Total Transaction Amount by Category", fontsize=16)
            plt.xlabel("Category", fontsize=14)
            plt.ylabel("Total Amount ($)", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot()  # Use st.pyplot to display the plot in Streamlit

        return df

    except Exception as e:
        print(f"Error processing or visualizing transactions: {e}")
        return None


    


import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def clean_and_preprocess(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.lower()

    def clean_numeric(value):
        if isinstance(value, str):
            value = value.replace(',', '').replace('$', '').replace('(', '-').replace(')', '').strip()
            value = re.sub(r'\.(?=\d{3}\.)', '', value)
        return pd.to_numeric(value, errors='coerce')

    if 'debits' in df.columns and 'credits' in df.columns:
        df['debits'] = df['debits'].apply(clean_numeric).fillna(0)
        df['credits'] = df['credits'].apply(clean_numeric).fillna(0)
        df['amount'] = df['credits'] - df['debits']
    elif 'deposits/credits' in df.columns and 'withdrawals/debits' in df.columns:
        df['deposits/credits'] = df['deposits/credits'].apply(clean_numeric).fillna(0)
        df['withdrawals/debits'] = df['withdrawals/debits'].apply(clean_numeric).fillna(0)
        df['amount'] = df['deposits/credits'] - df['withdrawals/debits']
    elif 'amount' in df.columns:
        df['amount'] = df['amount'].apply(clean_numeric)
    else:
        raise ValueError(f"Unable to determine Amount column in {file_path}")

    df['labelled category'] = df.get('labelled category', '').str.strip().str.lower()
    df['true category'] = df.get('true category', '').str.strip().str.lower()

    return df[['date', 'description', 'amount', 'labelled category', 'true category']]


def process_transaction_data(training_files, testing_file):
    model = SentenceTransformer('paraphrase-mpnet-base-v2')

    training_data = pd.concat([clean_and_preprocess(path) for path in training_files], ignore_index=True)
    testing_data = clean_and_preprocess(testing_file)

    def generate_embeddings(data, batch_size=500):
        embeddings = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            embeddings.extend(model.encode(batch['description'].tolist(), show_progress_bar=True))
        return embeddings

    training_data['embedding'] = generate_embeddings(training_data)
    testing_data['embedding'] = generate_embeddings(testing_data)

    similarity_matrix = cosine_similarity(
        np.array(testing_data['embedding'].tolist()),
        np.array(training_data['embedding'].tolist())
    )

    most_similar_indices = similarity_matrix.argmax(axis=1)
    testing_data['predicted_category'] = training_data.iloc[most_similar_indices]['labelled category'].values

    testing_data.drop(columns=['embedding'], inplace=True)

    summary_report = testing_data.groupby('predicted_category', as_index=False)['amount'].sum()
    summary_report.rename(columns={'amount': 'total_spent'}, inplace=True)

    correct_predictions = (testing_data['predicted_category'] == testing_data['true category']).sum()
    total_predictions = len(testing_data)
    accuracy = correct_predictions / total_predictions * 100

    total_income = testing_data[testing_data['amount'] > 0]['amount'].sum()
    total_expenses = abs(testing_data[testing_data['amount'] < 0]['amount'].sum())
    net_profit_loss = total_income - total_expenses

    pnl_statement = {
        'Total Income': f"${total_income:,.2f}",
        'Total Expenses': f"${total_expenses:,.2f}",
        'Net Profit/Loss': f"${net_profit_loss:,.2f}"
    }

    return {
        'categorized_transactions': testing_data,
        'summary_report': summary_report,
        #'accuracy': accuracy,
        #'pnl_statement': pnl_statement
    }










import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

def detect_date_column(df):
    """
    Detect the date column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        str: The name of the date column if found, otherwise None.
        str: The date format.
    """
    for col in df.columns:
        try:
            # Try parsing as format 1: DD-MMM
            sample1 = pd.to_datetime(df[col].head(10), format='%d-%b', errors='coerce')
            if sample1.notna().sum() > 0:
                return col, '%d-%b'  # Return column name and format
           
            # Try parsing as format 2: MM/DD
            sample2 = pd.to_datetime(df[col].head(10), format='%m/%d', errors='coerce')
            if sample2.notna().sum() > 0:
                return col, '%m/%d'  # Return column name and format
           
            # Try parsing as format 3: MM/DD/YY* or MM-DD-YYYY
            sample3 = pd.to_datetime(df[col].str.replace('*', '', regex=False), errors='coerce')
            if sample3.notna().sum() > 0:
                return col, 'custom'  # Return column name and custom format
        except Exception:
            continue
    return None, None

def calculate_profit_loss_from_file(file_path):
    """
    Calculate and display a profit and loss statement from a file containing transactions data.

    Args:
        file_path (str): The path to the file containing transactions data.

    Returns:
        tuple: (output_text, start_date, end_date) if successful, otherwise None.
    """
    try:
        # Load the file into a DataFrame
        df = pd.read_csv(file_path)
        df.columns = [col.strip() for col in df.columns]  # Clean column names

        # Detect the date column and its format
        date_column, date_format = detect_date_column(df)
        if date_column and date_format:
            # Parse the date column assuming the current year for formats
            current_year = pd.Timestamp.now().year
            if date_format == '%d-%b':
                df[date_column] = pd.to_datetime(
                    df[date_column] , format='%d-%b', errors='coerce'
                )
            elif date_format == '%m/%d':
                df[date_column] = pd.to_datetime(
                    df[date_column] , format='%m/%d', errors='coerce'
                )
            elif date_format == 'custom':
                df[date_column] = pd.to_datetime(
                    df[date_column].str.replace('*', '', regex=False), errors='coerce'
                )
            df = df.dropna(subset=[date_column])  # Drop rows with invalid dates

            # Format dates to exclude the year if not necessary
            date_range = df[date_column]
            if not date_range.empty:
                min_date = date_range.min()
                max_date = date_range.max()
                start_date = min_date.strftime('%d-%b')
                end_date = max_date.strftime('%d-%b')
            else:
                start_date = "Unknown"
                end_date = "Unknown"
        else:
            print("No valid date column detected.")
            start_date = "Unknown"
            end_date = "Unknown"

        # Identify the relevant column for processing
        if 'Amount_CD' in df.columns:
            column = 'Amount_CD'
            category_totals = df.groupby('category')[column].sum()
            income_categories = category_totals[category_totals > 0]  # Income
            expense_categories = category_totals[category_totals < 0].abs()  # Expenses
        elif 'Amount' in df.columns:
            column = 'Amount'
            category_totals = df.groupby('category')[column].sum()
            income_categories = category_totals[category_totals < 0].abs()  # Income
            expense_categories = category_totals[category_totals > 0]  # Expenses
        elif "'Amount" in df.columns:
            column = "'Amount"
            category_totals = df.groupby('category')[column].sum()
            income_categories = category_totals[category_totals < 0].abs()  # Income
            expense_categories = category_totals[category_totals > 0]  # Expenses
        else:
            print("No recognized columns for calculating profit and loss.")
            return None

        # Calculate totals and net profit/loss
        total_income = income_categories.sum()
        total_expenses = expense_categories.sum()
        net_profit_loss = total_income - total_expenses
        status = "Profit" if net_profit_loss > 0 else "Loss"

        # Generate output text for the statement
        output_text = "\nProfit and Loss Statement\n" + "-" * 30 + f"\nDate Range: {start_date} to {end_date}\n"
        output_text += "\nIncome:\n"
        for category, amount in income_categories.items():
            output_text += f"{category}: ${amount:,.2f}\n"
        output_text += f"Sales: ${total_income:,.2f}\n"

        output_text += "\nExpenses:\n"
        for category, amount in expense_categories.items():
            output_text += f"{category}: ${amount:,.2f}\n"

        output_text += "\nSummary:\n"
        output_text += f"Total Income: ${total_income:,.2f}\n"
        output_text += f"Total Expenses: ${total_expenses:,.2f}\n"
        output_text += f"Net Profit/Loss: ${net_profit_loss:,.2f} ({status})\n"

        return output_text, start_date, end_date
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def export_calculated_output_to_pdf(output_text, file_path, start_date=None, end_date=None):
    """
    Export the profit and loss statement to a PDF with proper formatting, including the date range.

    Args:
        output_text (str): The text output from the calculation function.
        file_path (str): The path to save the PDF file.
        start_date (str): The start date of the transactions.
        end_date (str): The end date of the transactions.

    Returns:
        None
    """
    try:
        # Check if the directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create the PDF document
        pdf = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        style_heading = styles["Heading1"]
        style_normal = styles["Normal"]

        # Add the date range at the top
        elements = []
        if start_date and end_date:
            date_paragraph = Paragraph(
                f"<b>Date Range:</b> {start_date} to {end_date}", style_normal
            )
            elements.append(date_paragraph)
            elements.append(Spacer(1, 12))  # Add space below the date range

        # Split the output_text into lines and prepare data for the table
        lines = output_text.splitlines()
        data = [["Description", "Amount"]]
        for line in lines:
            if line.startswith("Date Range"):  # Skip "Date Range" from table data
                continue
            if ":" in line:
                key, value = map(str.strip, line.split(":", 1))
                data.append([key, value])

        # Create and style the table
        table = Table(data, colWidths=[300, 100])

        # Add conditional styling for "Income," "Expenses," and "Summary" rows
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  # Header row background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),       # Header row text color
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),               # Right-align the Amount column
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),    # Header row font
            ('FONTSIZE', (0, 0), (-1, -1), 10),                 # Font size for all rows
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),              # Padding for header row
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)        # Add grid lines
        ])

        # Highlight specific rows (Income, Expenses, Summary)
        for i, row in enumerate(data):
            if row[0] in {"Income", "Expenses", "Summary"}:
                table_style.add('BACKGROUND', (0, i), (-1, i), colors.beige)  # Subtle highlight
                table_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')  # Bold text

        table.setStyle(table_style)

        # Add table and heading to PDF
        elements.append(Paragraph("Profit and Loss Statement", style_heading))
        elements.append(Spacer(1, 12))
        elements.append(table)

        # Build the PDF
        pdf.build(elements)
        print(f"Profit and Loss Statement successfully saved at: {file_path}")
    except Exception as e:
        print(f"An error occurred while creating the PDF: {e}")






        





















# Hardcode OpenAI API key
api_key = "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"

# Start Streamlit App
st.set_page_config(page_title="Bank Statement Processor with Categorization", layout="wide")
st.title("🏦 Bank Statement Processor with Categorization")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Step 1: Bank Selection
st.header("1️⃣ Select Bank Type")
bank_options = list(bank_functions.keys())  # Use the keys from the bank_functions dictionary
selected_bank = st.radio("Choose the bank statement type:", bank_options)

# Initialize session state for custom categories
if "custom_categories" not in st.session_state:
    st.session_state["custom_categories"] = {}

# Step 2: Process File or Directory Based on Bank Type
extracted_data = None  # Initialize extracted_data globally for use across categorization

if selected_bank:
    st.header("2️⃣ Upload Bank Statement PDF or Directory")
    
    # For specific banks, upload and process PDF
    if selected_bank in ["Wells Fargo", "Amex Business", "Amex Checking", "Jan Citizens Bank"]:
        st.header(f"2️⃣ Upload and Process Statement for {selected_bank}")

    # Step 1: Upload PDF for Extraction
        st.subheader("a) Upload PDF of the Bank Statement")
        uploaded_file = st.file_uploader("Upload the PDF of the bank statement:", type=["pdf"])

        if uploaded_file:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            temp_pdf_path = Path(f"./{uploaded_file.name}")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Process the file
            processing_function = bank_functions[selected_bank]
            with st.spinner(f"Processing {selected_bank} statement..."):
                extracted_data = processing_function(temp_pdf_path)

        # Show extracted data
            if extracted_data is not None and not extracted_data.empty:
                st.subheader("Extracted Data")
                st.dataframe(extracted_data)
                csv = extracted_data.to_csv(index=False)
                st.download_button(
                    label="Download Extracted Data as CSV",
                    data=csv,
                    file_name=f"{selected_bank}_processed_data.csv",
                    mime="text/csv",
                )
            else:
                st.error("No data extracted. Please verify the PDF file.")

    # Step 2: Upload CSV for Categorization
        st.subheader("b) Upload CSV File to Categorize Transactions")
        uploaded_csv = st.file_uploader("Upload CSV file to categorize transactions:", type=["csv"])

        if uploaded_csv:
            temp_file_path = f"./{uploaded_csv.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_csv.getbuffer())

            with st.spinner("Processing and categorizing transactions..."):
                processed_data = process_and_visualize_transactions(temp_file_path, api_key)

            if processed_data is not None:
                st.subheader("Processed and Categorized Data")
                st.dataframe(processed_data)
            # Allow download of the processed and categorized data
                categorized_csv = processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Processed and Categorized Data as CSV",
                    data=categorized_csv,
                    file_name=f"{selected_bank}_processed_categorized_data.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to categorize transactions.")

    # Step 3: Add Custom Categories
        st.subheader("c) Add Custom Categories")
        def add_custom_category():
            category_name = st.text_input("Enter the category name:")
            example_keywords = st.text_area("Enter associated keywords (comma-separated):")
            if st.button("Add Category"):
                if not category_name or not example_keywords.strip():
                    st.error("Please provide both a category name and associated keywords.")
                else:
                    keywords = [kw.strip() for kw in example_keywords.split(",") if kw.strip()]
                    st.session_state["custom_categories"][category_name] = keywords
                    st.success(f"Added category '{category_name}' with keywords: {', '.join(keywords)}")

        add_custom_category()

        # Display existing custom categories
        if st.session_state["custom_categories"]:
            st.subheader("Existing Custom Categories")
            for category, keywords in st.session_state["custom_categories"].items():
                st.write(f"**{category}**: {', '.join(keywords)}")

        # Step 5: Re-categorize Transactions (if extracted data exists)
        if extracted_data is not None and not extracted_data.empty:
            st.subheader("e) Re-categorize Transactions")
            if st.button("Re-categorize Transactions"):
                with st.spinner("Re-categorizing transactions with OpenAI..."):
                    try:
                        custom_categories = st.session_state.get("custom_categories", {})
                        recategorized_data = process_and_visualize_transactions(temp_file_path, api_key, custom_categories)
                        st.subheader("Re-categorized Transactions")
                        st.dataframe(recategorized_data)

                        # Allow download of the re-categorized data
                        recategorized_csv = recategorized_data.to_csv(index=False)
                        st.download_button(
                            label="Download Re-categorized Data as CSV",
                            data=recategorized_csv,
                            file_name=f"{selected_bank}_recategorized_data.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Error during re-categorization: {e}")

    # Step G: Generate Profit and Loss Statement
        st.subheader("g) Generate Profit and Loss Statement")
        categorized_csv_file = st.file_uploader("Upload your categorized CSV file:", type=["csv"])

        if categorized_csv_file:
    # Save the uploaded file to a permanent directory
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)  # Create the directory if it doesn't exist

            permanent_file_path = os.path.join(upload_dir, categorized_csv_file.name)
            with open(permanent_file_path, "wb") as f:
                f.write(categorized_csv_file.getbuffer())

            print(f"File saved to: {permanent_file_path}")

    # Load the uploaded file
            df_temp = pd.read_csv(permanent_file_path)
            print(f"Uploaded DataFrame:\n{df_temp.head()}")

            if st.button("Generate Profit & Loss Statement"):
                with st.spinner("Calculating Profit & Loss Statement..."):
                    try:
                # Calculate P&L using the permanent file path
                        output_text, start_date, end_date = calculate_profit_loss_from_file(permanent_file_path)

                # Generate PDF
                        output_pdf_path = "Profit_and_Loss_Statement.pdf"
                        export_calculated_output_to_pdf(output_text, output_pdf_path, start_date, end_date)

                # Provide success message and download option
                        st.success(f"P&L Statement saved as {output_pdf_path}")
                        st.download_button(
                            label="Download P&L Statement PDF",
                            data=open(output_pdf_path, "rb").read(),
                            file_name=output_pdf_path,
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"An error occurred while generating the P&L Statement: {e}")


    # Step H: Categorize Transactions using BERT
    st.subheader("h) Categorize Transactions using BERT")
    testing_file = st.file_uploader("Upload your testing CSV file:", type=["csv"])

# Constant training files
    training_files = [
        "C:/Users/HP/Downloads/Final_Training (1).csv",  # Replace with actual path
        "C:/Users/HP/Downloads/Final_Training_2.csv"  # Replace with actual path
    ]

    if testing_file:
    # Save the uploaded testing file to a permanent directory
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        testing_file_path = os.path.join(upload_dir, testing_file.name)
        with open(testing_file_path, "wb") as f:
            f.write(testing_file.getbuffer())

        st.write(f"File saved to: {testing_file_path}")

        if st.button("Categorize Transactions using BERT"):
            with st.spinner("Processing with BERT model..."):
                try:
                # Process transactions using BERT
                    results = process_transaction_data(training_files, testing_file_path)

                # Display results
                    # Display the entire categorized data
                    st.subheader("Categorized Transactions:")
                    st.dataframe(results['categorized_transactions'])

# Add a download button for the categorized transactions
                    csv = results['categorized_transactions'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Categorized Transactions as CSV",
                        data=csv,
                        file_name='categorized_transactions.csv',
                        mime='text/csv'
                    )


                    st.subheader("Spending Summary by Predicted Category:")
                    st.table(results['summary_report'])

                    #st.subheader("Prediction Accuracy:")
                    #st.write(f"{results['accuracy']:.2f}%")

                   # st.subheader("Profit & Loss Statement:")
                    #for key, value in results['pnl_statement'].items():
                     #   st.write(f"**{key}:** {value}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    

    
    elif selected_bank in ["USAA Checking", "Bank Of West", "BMO Checking"]:
        st.header(f"2️⃣ Upload and Process Statement for {selected_bank}")
    
    # Step A: Upload PDF to S3 Bucket
        st.subheader("a) Upload PDF to Send to S3 Bucket")
        uploaded_pdf = st.file_uploader("Upload the PDF to send to the S3 bucket:", type=["pdf"])

        if uploaded_pdf:
            cleaned_file_name = uploaded_pdf.name.replace("'", "")
            st.success(f"File '{cleaned_file_name}' uploaded successfully!")

        # Initialize folder index in session state if not set
            if "folder_index" not in st.session_state:
                st.session_state.folder_index = 1  # Start folder numbering at 1

        # Create a dynamic folder name
            folder_name = f"single-document-{st.session_state.folder_index}"
            folder_path = f"s3://{S3_BUCKET_NAME}/{folder_name}"

        # Upload to S3
            s3_response = None
            with st.spinner("Uploading to AWS S3..."):
                file_name = uploaded_pdf.name
                s3_response = upload_to_s3(uploaded_pdf, file_name, folder_name)
                time.sleep(5) 

        # Show response for S3 upload
            if "successfully" in s3_response.lower():
                st.success(f"File uploaded successfully to S3: {s3_response}")
            # Increment folder index after successful upload
                st.session_state.folder_index += 1
                st.session_state["last_uploaded_folder"] = folder_path
                with open("folder_path.txt", "w") as f:
                    f.write(folder_path)

                st.write("### S3 Folder Path")
                st.text(folder_path)

    # Step B: Run Textract Automation
        st.subheader("b) Run Textract Automation")
        if st.button("Run Textract Automation"):
            if "last_uploaded_folder" in st.session_state:
                with st.spinner("Running Textract Automation..."):
                    automation_result = run_automation(st.session_state["last_uploaded_folder"], file_name)
                st.success("Textract Automation completed successfully!")
                st.write(automation_result)
            else:
                st.error("No folder path available. Please upload a PDF first.")

    # Step C: Process Downloaded File Section
        st.subheader("c) Process Downloaded File")
        if st.button("Process Download"):
            extracted_path = process_downloaded_file()  # Call your unzipping function
            if extracted_path:
                st.success(f"File processed and extracted at: {extracted_path}")
                st.text(f"Extracted Path: {extracted_path}")
            else:
                st.error("Failed to process the downloaded file.")

    # Step D: Directory Input for Tabular Data
        st.subheader("d) Directory Input for Tabular Data")
        directory_path = st.text_input("Enter the directory path containing the extracted tables (CSV files):")

        if directory_path:
            try:
                processing_function = bank_functions[selected_bank]
                with st.spinner(f"Processing directory '{directory_path}' for {selected_bank}..."):
                    extracted_data = processing_function(directory_path)

                if extracted_data is not None and not extracted_data.empty:
                    st.subheader("Extracted Data")
                    st.dataframe(extracted_data)
                    combined_csv = extracted_data.to_csv(index=False)
                    st.download_button(
                        label="Download Extracted Data as CSV",
                        data=combined_csv,
                        file_name=f"{selected_bank}_combined_data.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("No data extracted. Please verify the directory and its contents.")
            except Exception as e:
                st.error(f"Error processing the directory: {e}")

    # Step E: Upload CSV to Categorize Transactions
        st.subheader("e) Upload CSV to Categorize Transactions")
        uploaded_csv = st.file_uploader("Upload CSV file to categorize transactions:", type=["csv"])

        if uploaded_csv:
            temp_file_path = f"./{uploaded_csv.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_csv.getbuffer())

            with st.spinner("Processing and categorizing transactions..."):
                processed_data = process_and_visualize_transactions(temp_file_path, api_key)

            if processed_data is not None:
                st.subheader("Processed and Categorized Data")
                st.dataframe(processed_data)
            # Allow download of the processed and categorized data
                categorized_csv = processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Processed and Categorized Data as CSV",
                    data=categorized_csv,
                    file_name=f"{selected_bank}_processed_categorized_data.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to categorize transactions.")

    # Step 3: Add Custom Categories
        st.subheader("c) Add Custom Categories")
        def add_custom_category():
            category_name = st.text_input("Enter the category name:")
            example_keywords = st.text_area("Enter associated keywords (comma-separated):")
            if st.button("Add Category"):
                if not category_name or not example_keywords.strip():
                    st.error("Please provide both a category name and associated keywords.")
                else:
                    keywords = [kw.strip() for kw in example_keywords.split(",") if kw.strip()]
                    st.session_state["custom_categories"][category_name] = keywords
                    st.success(f"Added category '{category_name}' with keywords: {', '.join(keywords)}")

        add_custom_category()

        # Display existing custom categories
        if st.session_state["custom_categories"]:
            st.subheader("Existing Custom Categories")
            for category, keywords in st.session_state["custom_categories"].items():
                st.write(f"**{category}**: {', '.join(keywords)}")

        # Step 5: Re-categorize Transactions (if extracted data exists)
        if extracted_data is not None and not extracted_data.empty:
            st.subheader("e) Re-categorize Transactions")
            if st.button("Re-categorize Transactions"):
                with st.spinner("Re-categorizing transactions with OpenAI..."):
                    try:
                        custom_categories = st.session_state.get("custom_categories", {})
                        recategorized_data = process_and_visualize_transactions(temp_file_path, api_key, custom_categories)
                        st.subheader("Re-categorized Transactions")
                        st.dataframe(recategorized_data)

                        # Allow download of the re-categorized data
                        recategorized_csv = recategorized_data.to_csv(index=False)
                        st.download_button(
                            label="Download Re-categorized Data as CSV",
                            data=recategorized_csv,
                            file_name=f"{selected_bank}_recategorized_data.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Error during re-categorization: {e}")

    # Step G: Generate Profit and Loss Statement
        st.subheader("g) Generate Profit and Loss Statement")
        categorized_csv_file = st.file_uploader("Upload your categorized CSV file:", type=["csv"])

        if categorized_csv_file:
    # Save the uploaded file to a permanent directory
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)  # Create the directory if it doesn't exist

            permanent_file_path = os.path.join(upload_dir, categorized_csv_file.name)
            with open(permanent_file_path, "wb") as f:
                f.write(categorized_csv_file.getbuffer())

            print(f"File saved to: {permanent_file_path}")

    # Load the uploaded file
            df_temp = pd.read_csv(permanent_file_path)
            print(f"Uploaded DataFrame:\n{df_temp.head()}")

            if st.button("Generate Profit & Loss Statement"):
                with st.spinner("Calculating Profit & Loss Statement..."):
                    try:
                # Calculate P&L using the permanent file path
                        output_text, start_date, end_date = calculate_profit_loss_from_file(permanent_file_path)

                # Generate PDF
                        output_pdf_path = "Profit_and_Loss_Statement.pdf"
                        export_calculated_output_to_pdf(output_text, output_pdf_path, start_date, end_date)

                # Provide success message and download option
                        st.success(f"P&L Statement saved as {output_pdf_path}")
                        st.download_button(
                            label="Download P&L Statement PDF",
                            data=open(output_pdf_path, "rb").read(),
                            file_name=output_pdf_path,
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"An error occurred while generating the P&L Statement: {e}")
            


        

else:
    st.info("Please select a bank type to proceed.")


