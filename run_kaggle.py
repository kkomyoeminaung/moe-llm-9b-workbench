import os
import subprocess
import time
import urllib.request
import socket

def get_public_ip():
    try:
        return urllib.request.urlopen('https://ident.me', timeout=5).read().decode('utf8')
    except:
        return "Not found"

print("🚀 Starting MoE Workbench Kaggle One-Click Deployment...")
public_ip = get_public_ip()

# Step 1: Install dependencies
print("📦 Installing backend & frontend dependencies...")
# Include node source update just in case for older environments
os.system("curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs &> /dev/null")
os.system("pip install -q transformers peft bitsandbytes huggingface_hub fastapi uvicorn pydantic python-multipart faiss-cpu sentence-transformers aiohttp beautifulsoup4")
os.system("npm install")
os.system("npm install -g localtunnel")

print(f"📌 Your Session IP: {public_ip} (Use this as your Tunnel Password)")

# Step 2: Start backend
print("🧠 Starting PyTorch Backend (Port 8080)...")
backend_process = subprocess.Popen(["python3", "backend/app_unified.py"])

# Step 3: Start frontend
print("🖥️ Starting React Frontend (Port 3000)...")
os.environ["VITE_API_URL"] = "http://localhost:8080"
# server.ts handles the proxying and Vite middleware
frontend_process = subprocess.Popen(["npm", "run", "dev", "--", "--host", "0.0.0.0"])

# Wait for servers to bind
print("⏳ Waiting for backend to initialize (may take a few minutes for weights to download)...")
max_retries = 60
for i in range(max_retries):
    try:
        with urllib.request.urlopen("http://127.0.0.1:3000/healthz", timeout=1) as f:
            if f.getcode() == 200:
                print("✅ Frontend is up and proxying!")
                break
    except:
        pass
    time.sleep(5)
    if i % 6 == 0 and i > 0:
        print(f"   Still loading model weights... (Seconds passed: {i * 5})")

# Step 4: Expose using localtunnel
print("🌐 Creating Public Tunnel to Frontend...")
tunnel_process = subprocess.Popen(
    ["lt", "--port", "3000", "--subdomain", f"moe-workbench-7b-v2"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

url = None
start_time = time.time()
while time.time() - start_time < 30:
    line = tunnel_process.stdout.readline()
    if "your url is" in line.lower():
        url = line.strip().split(" ")[-1]
        break
    if not line and tunnel_process.poll() is not None:
        # Fallback if specific subdomain fails
        tunnel_process = subprocess.Popen(["lt", "--port", "3000"], stdout=subprocess.PIPE, text=True)
        line = tunnel_process.stdout.readline()
        if "your url is" in line.lower():
            url = line.strip().split(" ")[-1]
        break

if url:
    print("\n" + "="*50)
    print("✅ SUCCESS! Your MoE Workbench is live here:")
    print(f"👉 {url}")
    print(f"\n🔑 TUNNEL PASSWORD: {public_ip}")
    print("   (Paste this when prompted by the Localtunnel bridge)")
    print("="*50)
    print("\n⚠️ Note: When opening the link, you may need to click 'Click to Continue' and enter the password above.")
else:
    print("❌ Failed to create public tunnel. Please check the logs.")

try:
    frontend_process.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    backend_process.terminate()
    frontend_process.terminate()
    tunnel_process.terminate()
