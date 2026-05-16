# 🚀 MoE LLM 9B Workbench (Pro Edition)

A high-performance workbench for managing, fine-tuning, and deploying Mixture of Experts (MoE) LLMs. Optimized for 7B/9B architectures with Kaggle/Colab support.

## ✨ Features

- **100% Own LLM Architecture**: Seamlessly integrate premerged weights (e.g., Qwen2.5-7B) with specialized MoE routing.
- **Smart Logic System**: 
    - **True Token Streaming**: Real-time generation output (not batch splits).
    - **Proper Chat Templates**: Full support for Hugging Face `apply_chat_template` ensuring accurate persona adherence.
    - **RAG Integration**: Integrated retrieval-augmented generation with FAISS.
- **Instructional Dashboard**: Full documentation in Myanmar language including step-by-step setup guides.
- **Advanced Controls**:
    - **System Persona Selection**: Define how your AI behaves in real-time.
    - **Temperature & Top-K**: Fine-tune the creativity of your model.
    - **Context Management**: 4096+ token window support for deep reasoning.

## 🛠️ Quick Start

### For Kaggle / Colab
1. Use the provided `moe_workbench_launcher.ipynb`.
2. Toggle **Internet ON** and **GPU (T4 x2)**.
3. Run the cells to clone and launch.

### For Local (Docker)
```bash
docker-compose up --build
```

---

## 🇲🇲 အသုံးပြုပုံ အနှစ်ချုပ် (Summary in Myanmar)

- **Chat**: ကိုယ်ပိုင် 7B model နှင့် တိုက်ရိုက် စကားပြောနိုင်ပါသည်။ Stream ကို ဖွင့်ထားခြင်းဖြင့် ပိုမိုမြန်ဆန်သော အဖြေများကို ရရှိပါမည်။
- **Architect**: လိုအပ်ချက်များကို ရိုက်ထည့်ကာ Software source code များအား အလိုအလျောက် ရေးသားခိုင်းနိုင်ပါသည်။
- **Knowledge**: မိမိဒေတာများကို PDF/TXT တင်ပြီး MoE Model ကို သင်ကြားပေးနိုင်ပါသည်။
- **Config**: Settings Dashboard တွင် Temperature နှင့် System Persona များကို စိတ်ကြိုက် ပြင်ဆင်နိုင်ပါသည်။
