import os
import subprocess
import time
import urllib.request
import socket

def get_public_ip():
    try:
        return urllib.request.urlopen('https://ident.me').read().decode('utf8')
    except:
        return "Not found"

print("🚀 Starting MoE Workbench Kaggle One-Click Deployment...")
public_ip = get_public_ip()
print(f"📌 Your Session IP: {public_ip} (Use this as your Tunnel Password)")

# Step 1: Install dependencies
print("📦 Installing backend dependencies...")
# Added faiss-cpu, sentence-transformers, aiohttp, beautifulsoup4 for RAG and Web Search
os.system("pip install -q transformers peft bitsandbytes huggingface_hub fastapi uvicorn pydantic python-multipart faiss-cpu sentence-transformers aiohttp beautifulsoup4")

# Step 2: Install Node modules
print("📦 Installing frontend packages...")
os.system("npm install")

# Step 3: Start backend
print("🧠 Starting PyTorch Backend (Port 8080)...")
backend_process = subprocess.Popen(["python3", "backend/app_unified.py"])

# Step 4: Start frontend
print("🖥️ Starting React Frontend (Port 3000)...")
os.environ["VITE_API_URL"] = "http://localhost:8080"
# server.ts handles the proxying and Vite middleware
frontend_process = subprocess.Popen(["npm", "run", "dev"])

# Wait for servers to bind with health check
print("⏳ Waiting for systems to initialize (this may take a few minutes for 7B model)...")
max_retries = 30
for i in range(max_retries):
    try:
        with urllib.request.urlopen("http://localhost:3000/healthz") as f:
            if f.getcode() == 200:
                print("✅ Frontend is up!")
                break
    except:
        pass
    time.sleep(10)
    if i % 3 == 0:
        print(f"   Still waiting... (Retry {i}/{max_retries})")

# Step 5: Expose using localtunnel
print("🌐 Creating Public Tunnel to Frontend...")
# Use npx to avoid global install issues on Kaggle
tunnel_process = subprocess.Popen(
    ["npx", "localtunnel", "--port", "3000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for localtunnel to print the URL
url = None
start_time = time.time()
while time.time() - start_time < 30:
    line = tunnel_process.stdout.readline()
    if "your url is" in line:
        url = line.strip().split(" ")[-1]
        break
    if not line and tunnel_process.poll() is not None:
        break

if url:
    print(f"\n" + "="*50)
    print(f"✅ SUCCESS! Your MoE Workbench is live here:")
    print(f"👉 {url}")
    print(f"\n🔑 TUNNEL PASSWORD: {public_ip}")
    print(f"   (Paste this when prompted by the Localtunnel bridge)")
    print("="*50)
    print("\n⚠️ Note: When opening the link, you may need to click 'Click to Continue' or enter the password above.")
else:
    print("❌ Failed to create public tunnel. Please check the logs.")
    # Fallback info
    print(f"Manual tunnel: npx localtunnel --port 3000")

# Keep the script running
try:
    frontend_process.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    backend_process.terminate()
    frontend_process.terminate()
    tunnel_process.terminate()
