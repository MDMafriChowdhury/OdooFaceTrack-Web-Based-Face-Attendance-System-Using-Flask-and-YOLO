import os
import ssl
from cheroot import wsgi
from cheroot.ssl.builtin import BuiltinSSLAdapter  # ✅ updated import

# Import your app and setup functions
from app import app, init_db, load_recognizer

# --- Server Setup ---
print("[INFO] Initializing database...")
init_db()

print("[INFO] Loading face recognition models...")
load_recognizer()

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 5000
THREADS = 20

print(f"[INFO] Starting Cheroot production server on https://{HOST}:{PORT}")
print(f"[INFO] Running with {THREADS} threads.")

# --- Create the WSGI server ---
server = wsgi.Server(
    (HOST, PORT),
    app,
    numthreads=THREADS
)

# --- Configure SSL using BuiltinSSLAdapter ---
try:
    cert_file = 'cert.pem'
    key_file = 'key.pem'

    # Make sure certs exist
    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        raise FileNotFoundError("SSL certificate or key not found.")

    # Apply SSL
    server.ssl_adapter = BuiltinSSLAdapter(cert_file, key_file, None)

except FileNotFoundError:
    print("=" * 50)
    print("ERROR: 'cert.pem' and 'key.pem' not found in this directory.")
    print("Please run OpenSSL to generate them:")
    print("  openssl req -new -x509 -days 365 -nodes -out cert.pem -keyout key.pem")
    print("=" * 50)
    exit(1)
except Exception as e:
    print("=" * 50)
    print(f"ERROR: Could not configure SSL: {e}")
    print("Try installing pyOpenSSL: python.exe -m pip install pyOpenSSL")
    print("=" * 50)
    exit(1)

# --- Start the server ---
try:
    print("[INFO] Server started successfully. Press Ctrl+C to stop.")
    server.start()
except KeyboardInterrupt:
    print("[INFO] Server shutting down...")
    server.stop()