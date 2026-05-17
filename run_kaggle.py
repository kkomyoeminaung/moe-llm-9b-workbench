import os
import subprocess
import time
import urllib.request
import socket
import random
import string
import json
import sys

def get_public_ip():
    try:
        return urllib.request.urlopen('https://ident.me', timeout=5).read().decode('utf8')
    except:
        return "Not found"

def generate_random_id(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

print("==========================================================")
print("🚀 MoE Workbench - Automated One-Click Deployment")
print("==========================================================")

# Step 0: Context Initialization
repo_url = "https://github.com/kkomyoeminaung/moe-llm-9b-workbench.git"
repo_name = "moe-llm-9b-workbench"

# Detect if we are inside the repo or need to clone
if not os.path.exists("backend") and not os.path.exists("package.json"):
    if os.path.exists(repo_name):
        print(f"📂 Found existing folder '{repo_name}'. Switching...")
        os.chdir(repo_name)
    else:
        print(f"📥 Project files not found. Cloning from {repo_url}...")
        os.system(f"git clone {repo_url}")
        if os.path.exists(repo_name):
            os.chdir(repo_name)
        else:
            print("❌ Error: Repository cloning failed. Please check internet connection.")
            sys.exit(1)
else:
    print("✅ Already inside the project workspace.")

public_ip = get_public_ip()
session_id = generate_random_id()
python_cmd = sys.executable

# Step 1: Resource Preparation
print("📦 Preparing system resources (Node.js, PyTorch dependencies)...")

# --- Environment Configuration ---
if 'HF_TOKEN' not in os.environ:
    print("🔑 HF_TOKEN not found in environment.")
    # Check for Kaggle Secrets if available
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            print("   - HF_TOKEN securely loaded from Kaggle Secrets.")
    except:
        pass

# Persistent Path Configuration
KAGGLE_INPUT_DIR = "/kaggle/input" 
LOCAL_MODEL_DIR = "models"
CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "data"

# Create necessary directories FIRST to avoid crashes during restoration status writing
print("   - Creating workspace directories...")
os.makedirs("exports", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("data/rag_index", exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Auto-Restore from previous runs (Kaggle Dataset Input)
print("🔍 Checking for persistent data in Kaggle Input...")
found_persistence = False
restored_assets = []

if os.path.exists(KAGGLE_INPUT_DIR):
    print("🔍 Scanning Kaggle Input for specialized weights and brains...")
    for dataset in os.listdir(KAGGLE_INPUT_DIR):
        ds_path = os.path.join(KAGGLE_INPUT_DIR, dataset)
        if not os.path.isdir(ds_path): continue
        
        reproduction_found = False
        for sub in ["models", "checkpoints", "data"]:
            source = os.path.join(ds_path, sub)
            if os.path.exists(source) and os.path.isdir(source) and os.listdir(source):
                print(f"   ✨ AUTO-RESTORE: Found '{sub}' in {dataset}. Merging...")
                # Use rsync if available for efficiency, otherwise cp
                if os.system(f"rsync -av --ignore-existing {source}/ {sub}/ &>/dev/null") != 0:
                    os.system(f"cp -rn {source}/* {sub}/ 2>/dev/null")
                found_persistence = True
                restored_assets.append(sub)
                reproduction_found = True
        
        if reproduction_found:
            print(f"   ✅ Successfully linked {dataset} as active brain memory.")

if found_persistence:
    # Write a status file for the backend/UI to see
    try:
        with open("data/restore_status.json", "w") as f:
            json.dump({
                "restored": True, 
                "assets": list(set(restored_assets)), 
                "time": time.time()
            }, f)
    except Exception as e:
        print(f"⚠️ Warning: Could not write restore status: {e}")
else:
    print("   - No previous persistent data found. Starting fresh.")

print("   - Installing requirements.txt...")
os.system(f"{python_cmd} -m pip install -r requirements.txt &> /dev/null")

print("   - Installing npm packages...")
os.system("npm install &> /dev/null")
os.system("npm install -g localtunnel &> /dev/null")

print(f"✅ Session IP: {public_ip}")
print(f"📌 Use this as your Tunnel Password if prompted by localtunnel.")

# Step 2: Bootup Sequence
print("\n🔥 Igniting Neural Engine (Backend)...")
backend_process = subprocess.Popen([python_cmd, "backend/app_unified.py"])

print("🛰️  Starting Interface (Frontend Proxy)...")
frontend_process = subprocess.Popen(["npm", "run", "dev", "--", "--host", "0.0.0.0"])

# Step 3: Stabilization Check
print("\n⏳ Stabilizing system (Downloading weights & initializing)...")
max_retries = 150 # Up to 12.5 minutes
backend_ready = False
frontend_ready = False

for i in range(max_retries):
    if not frontend_ready:
        try:
            with urllib.request.urlopen("http://127.0.0.1:3000/healthz", timeout=1) as f:
                if f.getcode() == 200:
                    print("✅ Interface ready.")
                    frontend_ready = True
        except:
            pass
    
    if not backend_ready:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/stats", timeout=1) as f:
                if f.getcode() == 200:
                    data = json.loads(f.read().decode())
                    if data.get("status") != "loading":
                        print("🔥 Neural Engine Online! Weights stabilized.")
                        backend_ready = True
                    else:
                        if i % 6 == 0:
                            print(f"   [SYNC] Loading model weights... ({i * 5}s)")
        except:
            pass
            
    if frontend_ready and backend_ready:
        break
    time.sleep(5)

# Step 4: External Tunneling
subdomain = f"moe-workbench-{session_id}"
print(f"\n🌐 Creating Public Access Tunnel...")
# Using npx localtunnel is more reliable in many cloud environments
tunnel_process = subprocess.Popen(
    ["npx", "localtunnel", "--port", "3000", "--subdomain", subdomain],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

url = ""
# Wait for some output from lt
# Wait up to 30 seconds for localtunnel
start_time = time.time()
while time.time() - start_time < 30:
    line = tunnel_process.stdout.readline()
    if "your url is" in line.lower():
        url = line.split("is:")[1].strip()
        break
    if not line and tunnel_process.poll() is not None:
        break
    time.sleep(1)

if not url:
    # Try random subdomain fallback
    print("   - Retrying with random subdomain...")
    tunnel_process = subprocess.Popen(["npx", "localtunnel", "--port", "3000"], stdout=subprocess.PIPE, text=True)
    start_time = time.time()
    while time.time() - start_time < 30:
        line = tunnel_process.stdout.readline()
        if "your url is" in line.lower():
            url = line.split("is:")[1].strip()
            break
        time.sleep(1)

if url:
    print("\n" + "*"*60)
    print(f"🎉 SUCCESS! YOUR MOE WORKBENCH IS PUBLICLY ACCESSIBLE:")
    print(f"🔗 URL: {url}")
    print(f"🔑 PASSWORD: {public_ip}")
    print("*"*60 + "\n")
    print("📢 PERSISTENCE REMINDER:")
    print("   To save your QLoRA weights and RAG data for the next run:")
    print("   1. Click 'Save Version' in the top right of Kaggle.")
    print("   2. Choose 'Quick Save' or 'Save & Run All'.")
    print("   3. On the next run, add this run's output as an 'Input Dataset'.")
    print("   4. The system will automatically detect and restore your brain. 🧠")
    print("\nInstructions:")
    print("1. Click the URL above.")
    print("2. If prompted, enter the PASSWORD shown above.")
    print("3. Enjoy your 9B MoE Workbench!")
else:
    print("\n❌ Failed to create public tunnel. You can try running 'lt --port 3000' manually.")

try:
    while True:
        if frontend_process.poll() is not None: 
            print("\nFrontend process exited.")
            break
        if backend_process.poll() is not None: 
            print("\nBackend process exited.")
            break
        time.sleep(2)
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
finally:
    backend_process.terminate()
    frontend_process.terminate()
    if 'tunnel_process' in locals():
        tunnel_process.terminate()
