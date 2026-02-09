import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import hashlib

# Scope for Google Sheets and Drive
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_NAME = "azundo_users"

def get_db_connection():
    """Establishes connection to Google Sheets."""
    try:
        if "gcp_service_account" not in st.secrets:
            return None, "Missing Google Cloud secrets. Please configure .streamlit/secrets.toml."
        
        # Load credentials from Streamlit secrets
        creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(credentials)
        
        # Open the spreadsheet
        sheet = client.open(SHEET_NAME).sheet1
        return sheet, None
    except FileNotFoundError:
        return None, "Secrets file not found. Please create .streamlit/secrets.toml."
    except Exception as e:
        return None, f"Connection Error: {e}"

def init_db():
    """Checks if the database (Sheet) is accessible and set up."""
    sheet, error = get_db_connection()
    if error:
        st.error(f"Database Error: {error}")
        return False
    
    try:
        # Check if headers exist
        headers = sheet.row_values(1)
        if headers != ["username", "password", "name"]:
            # If empty or wrong, set headers
            sheet.clear()
            sheet.append_row(["username", "password", "name"])
        return True
    except Exception as e:
        st.error(f"Failed to initialize DB: {e}")
        return False

def make_hashes(password):
    """Hashes a password with SHA256."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """Checks if a password matches the hash."""
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def register_user(username, password, name):
    """Registers a new user in Google Sheets."""
    sheet, error = get_db_connection()
    if not sheet:
        return False

    try:
        # Check if username already exists
        # Get all records map to check unicity efficiently
        records = sheet.get_all_records()
        for record in records:
            if record['username'] == username:
                return False
        
        # Append new user
        sheet.append_row([username, make_hashes(password), name])
        return True
    except Exception as e:
        print(f"Register Error: {e}")
        return False

def login_user(username, password):
    """Logs in a user. Returns tuple (username, password, name, progress_index)."""
    sheet, error = get_db_connection()
    if not sheet:
        return None

    try:
        records = sheet.get_all_records()
        hashed_pw = make_hashes(password)
        
        for i, record in enumerate(records):
            if record['username'] == username and str(record['password']) == hashed_pw:
                # Handle missing progress column gracefully
                progress = record.get('progress', 0)
                if progress == "": progress = 0
                return [(record['username'], record['password'], record['name'], int(progress))]
        return None
    except Exception as e:
        print(f"Login Error: {e}")
        return None

def update_progress(username, new_index):
    """Updates the progress index for a user."""
    sheet, error = get_db_connection()
    if not sheet:
        return False
        
    try:
        # specific cell update is more efficient but finding row is needed
        cell = sheet.find(username)
        if cell:
            # Update 'progress' column (assuming it's the 4th column, col D)
            # You can also find the header column index dynamically if preferred
            # For simplicity, let's assume it's column 4 based on instruction
            sheet.update_cell(cell.row, 4, new_index)
            return True
        return False
    except Exception as e:
        print(f"Update Progress Error: {e}")
        return False
