import streamlit as st
import time

# Asumsikan Anda memiliki file app2.py yang berisi aplikasi utama.
# Import file utama di sini agar bisa dipanggil setelah login berhasil.
try:
    # Mengganti import dari app_main menjadi app2
    from app2 import main_app
except ImportError:
    # Ini hanya placeholder jika app2.py belum ada/gagal diimport
    def main_app():
        st.error(importError)
        st.stop()


def check_login():
    """Menangani logika otentikasi pengguna."""
    
    # Inisialisasi session state jika belum ada
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Tampilkan formulir login hanya jika belum login
    if not st.session_state.logged_in:
        st.title("Sistem Deteksi Kerusakan Dinding")
        st.subheader("Silakan login untuk melanjutkan")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                # --- LOGIKA OTENTIKASI (Contoh Sederhana) ---
                # Ganti dengan otentikasi yang lebih aman di lingkungan produksi!
                if username == "user" and password == "1234":
                    st.session_state.logged_in = True
                    st.success("Login Berhasil!")
                    # Rerun untuk masuk ke aplikasi utama
                    st.rerun() 
                else:
                    st.error("Username atau password salah.")
    
    # Jika sudah login, jalankan aplikasi utama
    if st.session_state.logged_in:
        main_app()

if __name__ == "__main__":

    check_login()
