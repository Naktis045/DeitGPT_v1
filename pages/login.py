import streamlit as st
import sqlite3

# --- DATABASE FUNCTIONS ---
def create_user_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result

# --- APP LAYOUT ---
def signup():
    st.subheader("Sign Up")
    new_user = st.text_input("Choose a Username")
    new_pass = st.text_input("Choose a Password", type="password")

    if st.button("Create Account"):
        if new_user and new_pass:
            try:
                add_user(new_user, new_pass)
                st.success("Account created successfully!")
                st.info("You can now log in.")
            except sqlite3.IntegrityError:
                st.warning("Username already exists.")
        else:
            st.warning("Please fill in both fields.")

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.success(f"Welcome back, {username}!")
            st.balloons()
            st.write("🎉 You’re logged in.")
        else:
            st.error("Invalid credentials.")

def main():
    st.title("Login / Sign Up System")

    # Make sure the table exists
    create_user_table()

    menu = st.selectbox("Menu", ["Login", "Sign Up"])
    if menu == "Login":
        login()
    else:
        signup()

if __name__ == "__main__":
    main()
