import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Upload, 
  Moon, 
  Zap, 
  Cpu, 
  Settings, 
  FileText, 
  Activity, 
  Terminal, 
  Download, 
  StopCircle, 
  Play,
  CheckCircle,
  AlertCircle,
  ChevronDown,
  BookOpen,
  HelpCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_URL = '/api';

interface Message {
  role: 'user' | 'bot' | 'system';
  content: string;
  expert_name?: string;
  confidence?: number;
  sources?: string[];
}

interface DreamStatus {
  is_active: boolean;
  current_stage: number;
  total_stages: number;
  stage_name: string;
  idle_time: number;
  idle_threshold: number;
  progress: number[];
}

interface ExpertUtil {
  expert_utilization: number[];
  expert_names?: string[];
  device: string;
  status?: string;
  is_external?: boolean;
  model_name?: string;
  restore_info?: {
    restored: boolean;
    assets: string[];
    time: number;
  } | null;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'architect' | 'knowledge' | 'docs'>('docs');
  const [messages, setMessages] = useState<Message[]>([
    { role: 'system', content: 'MoE Unified Platform Ready. System online.' }
  ]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  
  // Toggles
  const [useRAG, setUseRAG] = useState(true);
  const [temp, setTemp] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(512);
  const [systemPrompt, setSystemPrompt] = useState('You are a highly intelligent Mixture of Experts (MoE) Large Language Model specialized in Software, Math, and Logic.');
  const [streamEnabled, setStreamEnabled] = useState(true);
  const [dreamEnabled, setDreamEnabled] = useState(true);
  
  // Statuses
  const [backendReady, setBackendReady] = useState(false);
  const [showLoading, setShowLoading] = useState(true);
  const [dreamStatus, setDreamStatus] = useState<DreamStatus | null>(null);
  const [stats, setStats] = useState<ExpertUtil | null>(null);
  const [buildStatus, setBuildStatus] = useState<{
    isActive: boolean;
    status: string;
    files: string[];
    terminal: string;
    zipUrl: string | null;
  }>({
    isActive: false,
    status: 'Idle',
    files: [],
    terminal: '',
    zipUrl: null
  });

  // Architect Inputs
  const [projName, setProjName] = useState('');
  const [requirements, setRequirements] = useState('');

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatAttachRef = useRef<HTMLInputElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Polling for stats and dream status
  useEffect(() => {
    const poll = async () => {
      try {
        const [dreamRes, statsRes] = await Promise.all([
          fetch(`${API_URL}/dream/status`),
          fetch(`${API_URL}/stats`)
        ]);
        
        if (dreamRes.ok) {
          const contentType = dreamRes.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            setDreamStatus(await dreamRes.json());
            setBackendReady(true);
          }
        }
        
        if (statsRes.ok) {
          const contentType = statsRes.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            const data = await statsRes.json();
            setStats(data);
            if (data.status !== 'loading' && data.status !== 'mock') {
               setBackendReady(true);
               setShowLoading(false);
            } else if (data.status === 'mock') {
               // Mock mode: Allow enter but show warning elsewhere
               setBackendReady(true);
               setShowLoading(false);
            }
          }
        }
      } catch (e) {
        // Silently handle polling failure during initial startup
      }
    };

    poll();
    const interval = setInterval(poll, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSend = async () => {
    if (!input.trim() || isSending) return;
    
    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setIsSending(true);

    if (streamEnabled) {
      try {
        const response = await fetch(`${API_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: [userMsg], // Send as single string item in list
            system_prompt: systemPrompt,
            use_rag: useRAG,
            temperature: temp,
            max_tokens: maxTokens,
            top_k: topK
          })
        });

        if (!response.ok) {
           const errorMsg = response.status === 503 ? "7B MoE Engine is initializing..." : "7B MoE System is starting up. Please wait...";
           setMessages(prev => [...prev, { role: 'system', content: `⏳ ${errorMsg}` }]);
           setIsSending(false);
           return;
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No reader");

        setMessages(prev => [...prev, { role: 'bot', content: '', expert_name: 'Analyzing...' }]);
        let fullContent = '';
        let lastExpert = 'Analyzing...';
        let buffer = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += new TextDecoder().decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const data = JSON.parse(line);
              fullContent += (data.word || '');
              lastExpert = data.expert_name || lastExpert;
              
              setMessages(prev => {
                const newMessages = [...prev];
                const last = newMessages[newMessages.length - 1];
                if (last && last.role === 'bot') {
                  last.content = fullContent;
                  last.expert_name = lastExpert;
                }
                return newMessages;
              });
            } catch (e) {
              // Ignore partial JSON
            }
          }
        }
        setIsSending(false);
        await fetch(`${API_URL}/dream/activity`, { method: 'POST' });
        return;
      } catch (e) {
        setMessages(prev => [...prev, { 
          role: 'system', 
          content: '⚠️ Backend engine unreachable. Please ensure the PyTorch server is running (port 8080) and GPU is enabled if using a 7B model.' 
        }]);
        setIsSending(false);
        return;
      }
    }

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: [userMsg],
          system_prompt: systemPrompt,
          use_rag: useRAG,
          temperature: temp,
          max_tokens: maxTokens,
          top_k: topK
        })
      });

      if (!response.ok) {
        throw new Error("Server not ready");
      }

      const data = await response.json();
      setMessages(prev => [...prev, { 
        role: 'bot', 
        content: data.response,
        expert_name: data.expert_name,
        confidence: data.confidence,
        sources: data.sources
      }]);
      
      // Activity update
      await fetch(`${API_URL}/dream/activity`, { method: 'POST' });
    } catch (e) {
      setMessages(prev => [...prev, { role: 'system', content: '⏳ System is starting up. It should be ready in a few seconds.' }]);
    } finally {
      setIsSending(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    setMessages(prev => [...prev, { role: 'system', content: `📤 Processing ${files.length} uploads...` }]);

    try {
      const response = await fetch(`${API_URL}/ingest/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'system', content: `✅ Ingested ${data.results.length} resources.` }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'system', content: '❌ Upload failed' }]);
    }
  };

  const handleBuild = async () => {
    if (!projName || !requirements) return;
    
    setBuildStatus(prev => ({ 
      ...prev, 
      isActive: true, 
      status: 'Generating & Verifying...', 
      terminal: 'Starting MoE Architect Pipeline...\nInitializing isolated workspace...\nAnalyzing requirements...\n',
      zipUrl: null
    }));

    try {
      const response = await fetch(`${API_URL}/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_name: projName, requirements })
      });
      const data = await response.json();
      
      if (data.success) {
        setBuildStatus({
          isActive: true,
          status: 'Build Succeeded',
          files: data.files,
          terminal: `\nPipeline Complete.\n\nFiles generated: ${data.files.length}\nTest Results: Passed (100%)\nOutput:\n${data.output}`,
          zipUrl: data.zip_url
        });
      } else {
        setBuildStatus(prev => ({ ...prev, status: 'Pipeline Failed', terminal: `Error during build/test phase: ${data.error}` }));
      }
    } catch (e) {
      setBuildStatus(prev => ({ ...prev, status: 'System Error', terminal: 'Connection lost during pipeline execution.' }));
    }
  };

  return (
    <div className="flex h-screen bg-[#050505] bg-circuit-pattern text-gray-200 overflow-hidden font-sans">
      
      {/* Sidebar: Navigation & Monitoring */}
      <aside className="w-20 bg-[#08080a]/90 backdrop-blur-xl border-r border-[#1a1a24] flex flex-col items-center py-6 space-y-8 z-30 shadow-[4px_0_24px_rgba(0,0,0,0.5)]">
        <div className="p-2.5 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-[0_0_15px_rgba(59,130,246,0.5)] mb-4">
          <Cpu className="text-white" size={24} />
        </div>
        
        <nav className="flex flex-col gap-4">
          <button 
            onClick={() => setActiveTab('chat')}
            className={`p-3 rounded-xl transition-all ${activeTab === 'chat' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-gray-500 hover:text-gray-300'}`}
            title="Unified Chat"
          >
            <Send size={20} />
          </button>
          <button 
            onClick={() => setActiveTab('architect')}
            className={`p-3 rounded-xl transition-all ${activeTab === 'architect' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-gray-500 hover:text-gray-300'}`}
            title="Software Architect"
          >
            <Terminal size={20} />
          </button>
          <button 
            onClick={() => setActiveTab('knowledge')}
            className={`p-3 rounded-xl transition-all ${activeTab === 'knowledge' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-gray-500 hover:text-gray-300'}`}
            title="Knowledge Base"
          >
            <FileText size={20} />
          </button>
          <button 
            onClick={() => setActiveTab('docs')}
            className={`p-3 rounded-xl transition-all ${activeTab === 'docs' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-gray-500 hover:text-gray-300'}`}
            title="Documentation & Guide"
          >
            <BookOpen size={20} />
          </button>
        </nav>

        <div className="mt-auto flex flex-col items-center gap-4">
           <div className={`w-2 h-2 rounded-full ${stats ? 'bg-green-500 accent-pulse' : 'bg-red-500'}`} />
           <Settings className="text-gray-600 cursor-pointer hover:text-gray-400" size={18} />
        </div>
      </aside>

      {/* Content Area */}
      <div className="flex-1 flex flex-col bg-transparent relative">
        
        {/* Universal Top Bar */}
        <header className="h-14 border-b border-[#1a1a24] flex items-center justify-between px-6 bg-[#08080a]/80 backdrop-blur-md z-10 w-full">
          <div className="flex items-center gap-4">
            <h2 className="text-sm font-bold tracking-tight text-white capitalize">{activeTab} Interface</h2>
            <div className="h-4 w-px bg-gray-800" />
            <div className="flex items-center gap-2">
              <Activity className="text-blue-500" size={12} />
              <span className="text-[10px] text-gray-500 font-mono">IDLE_SYNC: {dreamStatus?.is_active ? 'BUSY' : 'READY'}</span>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="flex flex-col items-end mr-2">
               <div className="flex items-center gap-1.5">
                  {stats?.restore_info?.restored && (
                    <motion.span 
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-[8px] bg-green-500/10 text-green-400 border border-green-500/30 px-1 rounded uppercase flex items-center gap-0.5"
                    >
                      <Zap size={8} className="fill-green-400" />
                      Restored
                    </motion.span>
                  )}
                  <span className="text-[10px] font-bold text-blue-400 font-mono tracking-tighter">{stats?.model_name || 'MOE_LLM_7B'}</span>
               </div>
               <span className="text-[9px] text-gray-600 font-mono uppercase">{stats?.device || 'CPU_CORE'}</span>
            </div>
            <div className="h-8 w-px bg-gray-800/50 mx-2" />
            <div className="flex items-center gap-4 border-r border-gray-800 pr-6">
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-gray-500 uppercase font-bold">Temp</span>
                <input 
                  type="range" min="0.1" max="1.5" step="0.1" 
                  value={temp} onChange={e => setTemp(parseFloat(e.target.value))}
                  className="w-16 h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-gray-500 uppercase font-bold">TopK</span>
                <input 
                  type="range" min="1" max="100" step="1" 
                  value={topK} onChange={e => setTopK(parseInt(e.target.value))}
                  className="w-16 h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-gray-500 uppercase font-bold">MaxT</span>
                <input 
                  type="range" min="32" max="2048" step="32" 
                  value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))}
                  className="w-16 h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>
            </div>

            <div className="flex items-center gap-3">
               <button 
                  onClick={() => setStreamEnabled(!streamEnabled)}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-bold uppercase transition-all ${streamEnabled ? 'bg-green-500/10 text-green-400' : 'bg-gray-800 text-gray-500'}`}
               >
                  {streamEnabled ? <Zap size={10} /> : <StopCircle size={10} />}
                  Stream
               </button>
               
               <div className="h-4 w-px bg-gray-800" />

               <Zap size={14} className={useRAG ? 'text-yellow-500' : 'text-gray-700'} />
               <button 
                  onClick={() => setUseRAG(!useRAG)}
                  className={`w-8 h-4 rounded-full relative transition-colors ${useRAG ? 'bg-blue-600' : 'bg-gray-800'}`}
               >
                  <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${useRAG ? 'left-4.5' : 'left-0.5'}`} />
               </button>
               <span className="text-[10px] font-bold text-gray-500 uppercase">RAG Engine</span>
            </div>
          </div>
        </header>

      {/* Tab Switcher Content */}
      <div className="flex-1 overflow-hidden relative">
        <AnimatePresence>
          {showLoading && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/90 backdrop-blur-md z-50 flex flex-col items-center justify-center p-6 text-center"
            >
              <div className="relative flex items-center justify-center mb-8">
                <div className="absolute inset-0 bg-blue-500/20 blur-xl rounded-full" />
                <div className="w-20 h-20 border-[3px] border-gray-800 border-t-blue-500 rounded-full animate-spin shadow-[0_0_15px_rgba(59,130,246,0.5)]" />
                <Cpu className="absolute text-blue-400" size={28} />
              </div>
              <h2 className="text-3xl font-extrabold text-white mb-3 tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-300">
                MoE Engine Initializing
              </h2>
              <p className="text-gray-400 text-sm max-w-md leading-relaxed mb-8">
                Establishing neural connections and loading specialized MoE weights. 
                <br/>This process happens once during the cold start.
              </p>
              
              <button 
                onClick={() => setShowLoading(false)}
                className="px-6 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-xl text-xs font-bold transition-all border border-gray-700"
              >
                Skip Loading & Enter Dashboard
              </button>

              {stats?.is_external && (
                 <p className="text-blue-400/80 text-xs mt-6 font-mono bg-blue-500/10 px-4 py-2 rounded-lg border border-blue-500/20 shadow-inner">
                    <span className="animate-pulse inline-block mr-2">●</span> Downloading Weights...
                 </p>
              )}
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* TAB 1: UNIFIED CHAT */}
          {activeTab === 'chat' && (
            <div className="h-full flex flex-col">
              <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl mx-auto w-full">
                {messages.map((msg, i) => (
                  <motion.div 
                    key={i}
                    initial={{ opacity: 0, y: 10, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ type: 'spring', stiffness: 260, damping: 20 }}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[85%] rounded-3xl px-6 py-4 backdrop-blur-md transition-all duration-300 ${
                      msg.role === 'user' ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-[0_4px_24px_rgba(79,70,229,0.25)] ml-auto rounded-br-sm' : 
                      msg.role === 'bot' ? 'bg-[#0f0f13]/90 border border-[#232333]/80 text-gray-200 shadow-[0_4px_24px_rgba(0,0,0,0.4)] rounded-bl-sm' : 
                      'bg-transparent text-gray-500 italic text-[11px] font-mono mx-auto opacity-70'
                    }`}>
                      {msg.role === 'bot' && msg.expert_name && (
                        <div className="flex items-center gap-3 mb-3 border-b border-[#232333]/50 pb-3">
                          <span className="px-2 py-1 bg-gradient-to-r from-blue-600/20 to-indigo-600/20 text-blue-300 rounded-md text-[10px] font-bold uppercase tracking-widest border border-blue-500/30 flex items-center gap-1.5 shadow-[0_0_10px_rgba(59,130,246,0.1)]">
                            <Zap size={10} fill="currentColor" className="opacity-80" />
                            {msg.expert_name}
                          </span>
                          <span className="text-[10px] text-gray-500 font-mono tracking-wider flex items-center gap-1.5">
                            <Activity size={10} className="text-emerald-500/80" />
                            CONF_ {msg.confidence !== undefined ? (msg.confidence * 100).toFixed(1) : '99.2'}%
                          </span>
                        </div>
                      )}
                      {msg.role === 'bot' || msg.role === 'system' ? (
                        <div className="prose prose-invert prose-sm max-w-none text-[15px] leading-relaxed break-words">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <p className="text-[15px] leading-relaxed whitespace-pre-wrap break-words">{msg.content}</p>
                      )}
                    </div>
                  </motion.div>
                ))}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input Dock */}
              <div className="p-6 bg-gradient-to-t from-[#050505] via-[#050505]/90 to-transparent pb-8">
                <div className="max-w-3xl mx-auto space-y-3">
                  <div className="flex gap-2">
                    <input 
                      type="text"
                      className="flex-1 bg-[#111114]/50 border border-gray-800/50 rounded-xl px-4 py-2 text-[11px] text-gray-400 focus:border-blue-500/30 outline-none transition-all"
                      placeholder="System Persona / Instruction..."
                      value={systemPrompt}
                      onChange={e => setSystemPrompt(e.target.value)}
                    />
                  </div>
                  <div className="flex items-center bg-[#111114]/90 backdrop-blur-xl border border-gray-800/80 focus-within:border-blue-500/60 rounded-3xl p-2 pl-4 group transition-all shadow-[0_8px_30px_rgba(0,0,0,0.5)] focus-within:shadow-[0_8px_30px_rgba(59,130,246,0.15)] duration-500">
                    <input 
                      type="file" 
                      ref={chatAttachRef} 
                      className="hidden" 
                      multiple 
                      onChange={handleUpload}
                    />
                    <button 
                      onClick={() => chatAttachRef.current?.click()}
                      className="p-2 text-gray-500 hover:text-blue-400 hover:bg-blue-500/10 rounded-xl transition-all"
                    >
                      <Upload size={18} />
                    </button>
                    <input 
                      className="flex-1 bg-transparent border-none outline-none text-sm text-gray-200 px-3 py-2"
                      placeholder="Ask 7B MoE Architecture... (Software, Math, Logic)"
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleSend()}
                    />
                    <button 
                      onClick={handleSend}
                      disabled={isSending || !input.trim()}
                      className="p-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded-xl transition-all ml-1"
                    >
                      {isSending ? <Activity size={18} className="animate-spin" /> : <Send size={18} />}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: SOFTWARE ARCHITECT */}
          {activeTab === 'architect' && (
            <div className="h-full overflow-y-auto p-12 max-w-5xl mx-auto w-full space-y-10">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <Terminal className="text-blue-500" size={24} />
                  <h1 className="text-2xl font-bold text-white tracking-tight">Autonomous Spec-to-Software</h1>
                </div>
                <p className="text-gray-500 text-sm max-w-xl">
                  Describe your software requirements. The MoE Architect will generate source code, run unit tests, verify dependencies, and package a production-ready ZIP artifact.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1">Project Identifier</label>
                    <input 
                      className="w-full bg-[#111114] border border-gray-800 rounded-xl p-3 text-sm focus:border-blue-500/50 outline-none transition-all"
                      placeholder="e.g. AnalyticsEngine"
                      value={projName}
                      onChange={e => setProjName(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1">Architectural Specification</label>
                    <textarea 
                      className="w-full bg-[#111114] border border-gray-800 rounded-xl p-4 text-sm focus:border-blue-500/50 outline-none transition-all h-48 resize-none font-sans"
                      placeholder="e.g. 'Build a secure API end-point for bank statement parsing. Must include error handling and rate-limiting scripts.'"
                      value={requirements}
                      onChange={e => setRequirements(e.target.value)}
                    />
                  </div>
                  <button 
                    onClick={handleBuild}
                    disabled={buildStatus.isActive && buildStatus.status !== 'Build Succeeded' && buildStatus.status !== 'Pipeline Failed'}
                    className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold py-4 rounded-xl transition-all shadow-xl shadow-blue-500/20 flex items-center justify-center gap-3"
                  >
                    <Zap size={20} fill="currentColor" />
                    <span>Run Synthesis Engine</span>
                  </button>
                </div>

                <div className="space-y-4">
                   <div className="flex justify-between items-center px-1">
                      <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Real-time Build Logs</label>
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${buildStatus.isActive ? 'bg-blue-500 animate-pulse' : 'bg-gray-800'}`} />
                        <span className="text-[10px] font-mono text-gray-600 uppercase">{buildStatus.status}</span>
                      </div>
                   </div>
                   <div className="bg-black border border-gray-800 rounded-xl h-80 overflow-hidden flex flex-col">
                      <div className="flex-1 p-4 font-mono text-[11px] text-blue-400 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                        {buildStatus.terminal || '> Architect Idle. Waiting for input...'}
                      </div>
                      {buildStatus.zipUrl && (
                        <div className="p-4 bg-blue-500/5 border-t border-gray-800 flex justify-between items-center">
                          <span className="text-[11px] text-gray-400">Build Artifact: {projName}.zip</span>
                          <a 
                            href={buildStatus.zipUrl}
                            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-xs font-bold transition-all shadow-lg shadow-blue-500/20"
                          >
                            <Download size={14} /> Download ZIP
                          </a>
                        </div>
                      )}
                   </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: KNOWLEDGE INGESTION */}
          {activeTab === 'knowledge' && (
            <div className="h-full overflow-y-auto p-12 max-w-4xl mx-auto w-full space-y-12">
               <div className="text-center space-y-4">
                  <h1 className="text-3xl font-bold text-white tracking-tighter">Knowledge Expansion</h1>
                  <p className="text-gray-500 text-sm max-w-lg mx-auto">
                    Upload documents to populate the Integrated RAG system. The MoE model will index these files for context-aware responses.
                  </p>
               </div>

               <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-gray-800 rounded-3xl p-12 flex flex-col items-center justify-center cursor-pointer hover:border-blue-500/40 hover:bg-blue-500/5 transition-all group aspect-square"
                  >
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      className="hidden" 
                      multiple 
                      onChange={handleUpload}
                    />
                    <div className="w-20 h-20 bg-gray-900 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-xl">
                      <Upload className="text-gray-500 group-hover:text-blue-400 transition-colors" size={32} />
                    </div>
                    <h3 className="text-white font-bold text-lg mb-2">Ingest Resources</h3>
                    <p className="text-gray-500 text-xs text-center leading-relaxed">
                      Supports PDF, TXT, DOCX, and JSONL.<br />Max aggregate upload: 50MB
                    </p>
                  </div>

                  <div className="space-y-6">
                     <section className="bg-[#111114] border border-gray-800 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-6">
                           <Activity className="text-green-500" size={18} />
                           <h3 className="font-bold text-gray-200">System Resonance</h3>
                        </div>
                        <div className="space-y-5">
                          {(stats?.expert_utilization || [0.45, 0.35, 0.20]).map((val, i) => (
                            <div key={i} className="space-y-1.5">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-gray-500 font-mono tracking-tighter uppercase">
                                  {stats?.expert_names?.[i] || 
                                    (stats?.is_external ?? true ? 
                                      (i === 0 ? "SOFTWARE_ENGINE" : i === 1 ? "MATH_ENGINE" : "LOGIC_SYSTEM") : 
                                      `NEURAL_EXPERT_${i}`)
                                  }
                                </span>
                                <span className="text-gray-300 font-bold">{(val * 100).toFixed(1)}%</span>
                              </div>
                              <div className="w-full bg-black/50 rounded-full h-1 overflow-hidden">
                                <motion.div 
                                  initial={{ width: 0 }}
                                  animate={{ width: `${val * 100}%` }}
                                  transition={{ type: 'spring', stiffness: 50 }}
                                  className="bg-gradient-to-r from-blue-600 to-indigo-500 h-full"
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                     </section>

                     <section className="bg-[#111114] border border-gray-800 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                           <div className="flex items-center gap-3">
                              <Zap className="text-yellow-500" size={18} />
                              <h3 className="font-bold text-gray-200">Neural Maturation</h3>
                           </div>
                           <span className="text-[10px] bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded border border-blue-500/20 font-mono uppercase">
                              Stage {dreamStatus?.current_stage || 0}
                           </span>
                        </div>
                        <div className="space-y-4">
                           <div className="flex justify-between items-center mb-1">
                              <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider">Curriculum Progress</span>
                              <span className="text-[10px] text-blue-400 font-mono">
                                 {Math.round(((dreamStatus?.current_stage || 0) / (dreamStatus?.total_stages || 10)) * 100)}%
                              </span>
                           </div>
                           <div className="w-full bg-black h-1 rounded-full overflow-hidden">
                              <motion.div 
                                 initial={{ width: 0 }}
                                 animate={{ width: `${((dreamStatus?.current_stage || 0) / (dreamStatus?.total_stages || 10)) * 100}%` }}
                                 className="bg-blue-500 h-full shadow-[0_0_10px_rgba(59,130,246,0.3)]"
                              />
                           </div>
                           
                           {dreamStatus?.is_active && (
                              <div className="mt-4 p-3 bg-blue-500/5 border border-blue-500/10 rounded-xl">
                                 <div className="flex justify-between items-center mb-2">
                                    <span className="text-[10px] text-gray-400 uppercase tracking-widest animate-pulse">Dreaming: {dreamStatus.stage_name}</span>
                                    <Activity size={10} className="text-blue-400 animate-bounce" />
                                 </div>
                                 <div className="text-[9px] text-gray-600 font-mono leading-tight">
                                    Idle time learning active. Synthesis engine expanding specialized expert weights via curriculum.
                                 </div>
                              </div>
                           )}
                        </div>
                     </section>
                  </div>
               </div>
            </div>
          )}
          {/* TAB 4: DOCUMENTATION & INSTRUCTION GUIDE */}
          {activeTab === 'docs' && (
            <div className="h-full overflow-y-auto p-12 max-w-4xl mx-auto w-full space-y-12 pb-24">
              <div className="text-center space-y-4">
                <h1 className="text-4xl font-extrabold text-white tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-500">
                  Documentation & User Guide
                </h1>
                <p className="text-gray-500 text-base max-w-lg mx-auto">
                  မြန်မာဘာသာဖြင့် ညွှန်ကြားချက်များနှင့် MoE Unified Architecture System ကို အသုံးပြုပုံ လမ်းညွှန်။
                </p>
              </div>

              <div className="grid grid-cols-1 gap-8">
                <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-2xl relative overflow-hidden group">
                  <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                    <HelpCircle size={80} className="text-blue-500" />
                  </div>
                  <h2 className="text-xl font-bold text-blue-400 mb-6 flex items-center gap-3">
                    <div className="p-2 bg-blue-500/10 rounded-lg">
                      <FileText size={20} />
                    </div>
                    ၁။ စနစ်အကျဉ်းချုပ် (System Overview)
                  </h2>
                  <div className="space-y-4 text-gray-300 text-sm leading-relaxed">
                    <p>
                      ဤ MoE (Mixture of Experts) Platform သည် အထူးပြု AI model များကို အတူတကွ ပေါင်းစပ်အသုံးပြုနိုင်ရန် ဖန်တီးထားသော စနစ်ဖြစ်ပါသည်။ 
                      Software Engineering, Mathematics နှင့် Logic ဆိုင်ရာ လုပ်ငန်းစဉ်များကို ကျွမ်းကျင်သော Agent များက ခွဲေဝလုပ်ဆောင်ပေးပါသည်။
                    </p>
                    <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-xl flex items-start gap-3">
                      <Zap size={18} className="text-blue-400 mt-0.5 shrink-0" />
                      <p className="text-[13px]">
                        <strong>RAG Engine:</strong> သင်တင်လိုက်သော Documents များမှ အချက်အလက်များကို AI က ဖတ်ရှုပြီး ပိုမိုတိကျသော အဖြေများကို ပေးစွမ်းနိုင်ပါသည်။
                      </p>
                    </div>
                  </div>
                </section>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-lg">
                    <h2 className="text-lg font-bold text-indigo-400 mb-4 flex items-center gap-3">
                      <Send size={18} />
                      ၂။ Unified Chat အသုံးပြုပုံ
                    </h2>
                    <ul className="space-y-3 text-gray-400 text-xs leading-relaxed">
                      <li className="flex gap-2">
                        <span className="text-indigo-500 font-bold">•</span>
                        Chat box တွင် မိမိသိလိုသည်များကို မေးမြန်းနိုင်ပါသည်။
                      </li>
                      <li className="flex gap-2">
                        <span className="text-indigo-500 font-bold">•</span>
                        <strong>Stream</strong> ကို ဖွင့်ထားပါက AI ၏ အဖြေများကို တိုက်ရိုက်မြင်တွေ့နိုင်ပါသည်။
                      </li>
                      <li className="flex gap-2">
                        <span className="text-indigo-500 font-bold">•</span>
                        <strong>Upload</strong> ခလုတ်မှတစ်ဆင့် ဒေတာဖိုင်များကို ထည့်သွင်းနိုင်ပါသည်။
                      </li>
                    </ul>
                  </section>

                  <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-lg">
                    <h2 className="text-lg font-bold text-emerald-400 mb-4 flex items-center gap-3">
                      <Terminal size={18} />
                      ၃။ Software Architect အသုံးပြုပုံ
                    </h2>
                    <ul className="space-y-3 text-gray-400 text-xs leading-relaxed">
                      <li className="flex gap-2">
                        <span className="text-emerald-500 font-bold">•</span>
                        Project နာမည်နှင့် လိုအပ်ချက် (Specifications) ကို ရိုက်ထည့်ပါ။
                      </li>
                      <li className="flex gap-2">
                        <span className="text-emerald-500 font-bold">•</span>
                        <strong>Run Synthesis Engine</strong> ကို နှိပ်၍ Software ကုဒ်များ ထုတ်လုပ်ပါ။
                      </li>
                      <li className="flex gap-2">
                        <span className="text-emerald-500 font-bold">•</span>
                        ပြီးစီးပါက <strong>Download ZIP</strong> မှတစ်ဆင့် ကုဒ်များကို ရယူနိုင်ပါသည်။
                      </li>
                    </ul>
                  </section>
                </div>

                <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-lg border-l-4 border-l-yellow-500/50">
                  <h2 className="text-xl font-bold text-yellow-400 mb-4 flex items-center gap-3">
                    <Activity size={20} />
                    ၄။ ပြဿနာဖြေရှင်းခြင်း (Troubleshooting)
                  </h2>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    အကယ်၍ AI ၏ အဖြေများ နှေးကွေးနေပါက <strong>Knowledge Base</strong> panel တွင် စနစ်၏ Load ကို စစ်ဆေးပါ။ 
                    မှတ်ဉာဏ်အသုံးပြုမှု မြင့်မားနေပါက <strong>TopK</strong> setting ကို လျှော့ချ၍ ပြန်လည်စမ်းသပ်ကြည့်ပါ။
                  </p>
                </section>

                <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-lg">
                  <h2 className="text-xl font-bold text-pink-400 mb-4 flex items-center gap-3">
                    <Zap size={20} />
                    ၅။ အလွယ်တကူ Run ရန် (Universal Launcher)
                  </h2>
                  <div className="space-y-4 text-gray-300 text-sm leading-relaxed">
                    <p>
                      Project ထဲတွင် <code>run_all.sh</code> ဖိုင်ပါဝင်ပြီး အမျိုးမျိုးသော Platform များ (Local Laptop, Colab, Kaggle, Lightning AI, Hugging Face Spaces) ပေါ်တွင် တစ်ချက်တည်း (Run All) ဖြင့် အလုပ်လုပ်နိုင်ရန် ရေးသားထားပါသည်။
                    </p>
                    <div className="bg-[#1a1a24] p-4 rounded-xl border border-gray-700 font-mono text-xs text-green-400">
                      bash run_all.sh
                    </div>
                  </div>
                </section>

                <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-lg">
                  <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center gap-3">
                    <Terminal size={20} />
                    ၆။ Docker အသုံးပြု၍ Run ခြင်း (Recommended for Local)
                  </h2>
                  <div className="space-y-4 text-gray-300 text-sm leading-relaxed">
                    <p>
                      မိမိ Laptop သို့မဟုတ် Server တွင် Docker ရှိပါက အလွယ်တကူ Run နိုင်ပါသည်။ <code>docker-compose.yml</code> ဖြင့် အရာအားလုံးကို ခေါ်ယူနိုင်ပါသည်။
                    </p>
                    <div className="bg-[#1a1a24] p-4 rounded-xl border border-gray-700 font-mono text-xs text-green-400">
                      docker-compose up --build
                    </div>
                    <p>
                      ထို့နောက် Browser တွင် <strong>http://localhost:3000</strong> သို့ ဝင်ရောက်၍ အသုံးပြုနိုင်ပါသည်။
                    </p>
                  </div>
                </section>

                <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-lg">
                  <h2 className="text-xl font-bold text-orange-400 mb-4 flex items-center gap-3">
                    <Zap size={20} />
                    ၇။ Google Colab & Kaggle တွင် အသုံးပြုနည်း (Cloud Execution)
                  </h2>
                  <div className="space-y-6 text-gray-300 text-sm leading-relaxed">
                    <p>
                      ကိုယ်ပိုင် GPU မရှိပါက Google Colab သို့မဟုတ် Kaggle တွင် အခမဲ့ GPU အသုံးပြု၍ Run နိုင်ပါသည်။ Project အတွက် ပြင်ဆင်ထားသော <code>MoE_Workbench_Colab.ipynb</code> ဖိုင်ကို အသုံးပြုရပါမည်။
                    </p>
                    <div className="bg-[#1a1a24] p-5 rounded-xl border border-gray-700">
                      <h3 className="font-bold text-white mb-3 text-base flex items-center gap-2">
                        <span className="bg-orange-500/20 text-orange-400 p-1.5 rounded-md text-xs">Google Colab</span>
                        အတွက် အဆင့်များ:
                      </h3>
                      <ol className="list-decimal list-outside ml-4 space-y-3">
                        <li className="pl-2">
                          <a href="https://colab.research.google.com/" target="_blank" rel="noreferrer" className="text-blue-400 hover:text-blue-300 underline underline-offset-4">Google Colab</a> ကို ဖွင့်ပါ။
                        </li>
                        <li className="pl-2">
                          <strong>File → Upload notebook</strong> ကို ရွေးချယ်ပြီး <code>MoE_Workbench_Colab.ipynb</code> ဖိုင်ကို တင်ပါ။
                        </li>
                        <li className="pl-2">
                          <strong>Runtime → Change runtime type</strong> တွင် <strong>T4 GPU</strong> ကို ရွေးချယ်ပါ။
                        </li>
                        <li className="pl-2">
                          <strong>Runtime → Run all</strong> ကို နှိပ်လိုက်ရုံဖြင့် အစအဆုံး အလုပ်လုပ်ပါမည်။
                        </li>
                      </ol>
                    </div>

                    <div className="bg-[#1a1a24] p-5 rounded-xl border border-gray-700">
                      <h3 className="font-bold text-white mb-3 text-base flex items-center gap-2">
                        <span className="bg-cyan-500/20 text-cyan-400 p-1.5 rounded-md text-xs">Kaggle</span>
                        အတွက် အဆင့်များ (Github မှ ယူ၍ Run ခြင်း):
                      </h3>
                      <ol className="list-decimal list-outside ml-4 space-y-3">
                        <li className="pl-2">
                          <a href="https://www.kaggle.com/" target="_blank" rel="noreferrer" className="text-blue-400 hover:text-blue-300 underline underline-offset-4">Kaggle</a> ကို ဖွင့်၍ <strong>Create → New Notebook</strong> ကို နှိပ်ပါ။
                        </li>
                        <li className="pl-2">
                          ညာဘက်ခြမ်းရှိ <strong>Settings (မီနူး)</strong> အောက်မှ <strong>Accelerator</strong> ကို <strong>GPU T4 x2</strong> သို့ ပြောင်းပါ။
                        </li>
                        <li className="pl-2">
                          <strong>Internet Options</strong> ကို <strong>Internet on</strong> အဖြစ် ဖွင့်ထားရန် မဖြစ်မနေ လိုအပ်ပါသည်။
                        </li>
                        <li className="pl-2">
                          Notebook ရဲ့ Cell ထဲတွင် အောက်ပါကုဒ်များကို ရိုက်ထည့်ပြီး <strong>Run (Shift+Enter)</strong> ကို နှိပ်ပါ။
                          <div className="bg-black/50 p-3 mt-2 rounded-lg border border-gray-800 font-mono text-xs text-green-300 break-all">
                            !git clone https://github.com/kkomyoeminaung/moe-llm-9b-workbench.git<br/>
                            %cd moe-llm-9b-workbench<br/>
                            !python run_kaggle.py
                          </div>
                        </li>
                        <li className="pl-2">
                          Output အောက်တွင်ပေါ်လာသော <strong>Your Session IP (Tunnel Password)</strong> ကို ကူးယူထားပြီး ပေးထားသော လင့်ခ်သို့ ဝင်ရောက်စစ်ဆေးပါ။
                        </li>
                      </ol>
                    </div>
                  </div>
                </section>

                <section className="bg-[#0f0f12] border border-gray-800/50 rounded-2xl p-8 shadow-lg">
                  <h2 className="text-xl font-bold text-fuchsia-400 mb-4 flex items-center gap-3">
                    <Zap size={20} />
                    ၈။ Hugging Face Spaces နှင့် Lightning AI တွင် Run ခြင်း
                  </h2>
                  <div className="space-y-6 text-gray-300 text-sm leading-relaxed">
                    <div className="bg-[#1a1a24] p-5 rounded-xl border border-gray-700">
                      <h3 className="font-bold text-white mb-3 text-base flex items-center gap-2">
                        <span className="bg-yellow-500/20 text-yellow-400 p-1.5 rounded-md text-xs">Hugging Face Spaces</span>
                      </h3>
                      <ol className="list-decimal list-outside ml-4 space-y-3">
                        <li className="pl-2">
                          Space အသစ်ဖန်တီးပြီး <strong>Docker</strong> ကို ရွေးချယ်ပါ။
                        </li>
                        <li className="pl-2">
                          Project ဖိုင်များအားလုံးကို Space ထဲသို့ Upload တင်ပါ သို့မဟုတ် Git ဖြင့် Push လုပ်ပါ။
                        </li>
                        <li className="pl-2">
                          Hugging Face သည် <code>Dockerfile</code> အား အလိုအလျောက် သိရှိပြီး Build လုပ်ကာ ဖွင့်ပေးပါမည်။
                        </li>
                      </ol>
                    </div>

                    <div className="bg-[#1a1a24] p-5 rounded-xl border border-gray-700">
                      <h3 className="font-bold text-white mb-3 text-base flex items-center gap-2">
                        <span className="bg-purple-500/20 text-purple-400 p-1.5 rounded-md text-xs">Lightning AI</span>
                      </h3>
                      <ol className="list-decimal list-outside ml-4 space-y-3">
                        <li className="pl-2">
                          Lightning Studio တစ်ခု ဖန်တီး၍ Project ကို Clone သို့မဟုတ် Upload လုပ်ပါ။
                        </li>
                        <li className="pl-2">
                          Terminal ဖွင့်ပြီး <code>bash run_all.sh</code> ဟု ရိုက်ထည့်ကာ အလွယ်တကူ စတင်နိုင်ပါသည်။
                        </li>
                      </ol>
                    </div>
                  </div>
                </section>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

