"""
Script bantu untuk menjalankan dashboard Streamlit dan membuka tunnel
ngrok secara otomatis dalam satu proses. Cocok dipakai pada Google Colab
atau lingkungan lain yang tidak memiliki port publik langsung.

Cara pakai:
    1. Pastikan sudah menjalankan: pip install -r requirements.txt
    2. Daftar di https://dashboard.ngrok.com untuk mendapatkan authtoken
    3. Jalankan: python run_with_ngrok.py --token <NGROK_AUTHTOKEN>
"""

import argparse
import subprocess
import time

from pyngrok import ngrok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Ngrok authtoken dari dashboard.ngrok.com")
    parser.add_argument("--port", type=int, default=8501, help="Port lokal Streamlit")
    args = parser.parse_args()

    ngrok.set_auth_token(args.token)
    public_url = ngrok.connect(args.port, "http")
    print(f"Dashboard dapat diakses publik melalui: {public_url}")

    process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", str(args.port)]
    )

    try:
        process.wait()
    except KeyboardInterrupt:
        print("Menghentikan dashboard dan tunnel ngrok...")
        process.terminate()
        ngrok.disconnect(public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
