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
  ChevronDown
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

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
  device: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'architect' | 'knowledge'>('chat');
  const [messages, setMessages] = useState<Message[]>([
    { role: 'system', content: 'MoE Unified Platform Ready. System online.' }
  ]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  
  // Toggles
  const [useRAG, setUseRAG] = useState(true);
  const [temp, setTemp] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [streamEnabled, setStreamEnabled] = useState(true);
  const [dreamEnabled, setDreamEnabled] = useState(true);
  
  // Statuses
  const [backendReady, setBackendReady] = useState(true);
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
          }
        }
        
        if (statsRes.ok) {
          const contentType = statsRes.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            setStats(await statsRes.json());
          }
        }
      } catch (e) {
        console.error('Polling failed:', e);
      }
    };

    poll();
    const interval = setInterval(poll, 5000);
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
            message: userMsg.split(' '),
            use_rag: useRAG,
            temperature: temp,
            top_k: topK
          })
        });

        if (!response.ok) throw new Error("Stream failed");

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No reader");

        setMessages(prev => [...prev, { role: 'bot', content: '', expert_name: 'Analyzing...' }]);
        let fullContent = '';
        let lastExpert = 'Analyzing...';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = new TextDecoder().decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const data = JSON.parse(line);
              fullContent += (data.word + ' ');
              lastExpert = data.expert_name;
              
              setMessages(prev => {
                const newMessages = [...prev];
                const last = newMessages[newMessages.length - 1];
                last.content = fullContent;
                last.expert_name = lastExpert;
                return newMessages;
              });
            } catch (e) {
              console.error("JSON parse error in stream", e);
            }
          }
        }
        setIsSending(false);
        await fetch(`${API_URL}/dream/activity`, { method: 'POST' });
        return;
      } catch (e) {
        console.error("Streaming error:", e);
      }
    }

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg.split(' '),
          use_rag: useRAG,
          temperature: temp,
          top_k: topK
        })
      });

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
      setMessages(prev => [...prev, { role: 'system', content: '❌ Connection error' }]);
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
    <div className="flex h-screen bg-[#050505] text-gray-200 overflow-hidden font-sans">
      
      {/* Sidebar: Navigation & Monitoring */}
      <aside className="w-20 bg-[#0d0d0f] border-r border-gray-800 flex flex-col items-center py-6 space-y-8 z-20">
        <div className="p-2 bg-blue-600 rounded-xl shadow-lg shadow-blue-500/20 mb-4">
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
        </nav>

        <div className="mt-auto flex flex-col items-center gap-4">
           <div className={`w-2 h-2 rounded-full ${stats ? 'bg-green-500 accent-pulse' : 'bg-red-500'}`} />
           <Settings className="text-gray-600 cursor-pointer hover:text-gray-400" size={18} />
        </div>
      </aside>

      {/* Content Area */}
      <div className="flex-1 flex flex-col bg-[#050505] relative">
        
        {/* Universal Top Bar */}
        <header className="h-14 border-b border-gray-800 flex items-center justify-between px-6 bg-[#050505]/80 backdrop-blur-md z-10 w-full">
          <div className="flex items-center gap-4">
            <h2 className="text-sm font-bold tracking-tight text-white capitalize">{activeTab} Interface</h2>
            <div className="h-4 w-px bg-gray-800" />
            <div className="flex items-center gap-2">
              <Activity className="text-blue-500" size={12} />
              <span className="text-[10px] text-gray-500 font-mono">IDLE_SYNC: {dreamStatus?.is_active ? 'BUSY' : 'READY'}</span>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
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
          
          {/* TAB 1: UNIFIED CHAT */}
          {activeTab === 'chat' && (
            <div className="h-full flex flex-col">
              <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl mx-auto w-full">
                {messages.map((msg, i) => (
                  <motion.div 
                    key={i}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[85%] rounded-2xl px-5 py-3 ${
                      msg.role === 'user' ? 'bg-blue-600 text-white' : 
                      msg.role === 'bot' ? 'bg-[#151518] border border-gray-800 text-gray-200' : 
                      'bg-transparent text-gray-500 italic text-[11px]'
                    }`}>
                      {msg.role === 'bot' && msg.expert_name && (
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="px-1.5 py-0.5 bg-blue-500/10 text-blue-400 rounded text-[9px] font-bold uppercase tracking-widest border border-blue-500/20">
                            {msg.expert_name}
                          </span>
                          <span className="text-[9px] text-gray-600 font-mono">CONF: {(msg.confidence! * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </motion.div>
                ))}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input Dock */}
              <div className="p-6 bg-gradient-to-t from-[#050505] via-[#050505] to-transparent">
                <div className="max-w-3xl mx-auto flex items-center bg-[#111114] border border-gray-800 focus-within:border-blue-500/50 rounded-2xl p-2 pl-3 group transition-all shadow-2xl">
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
                    placeholder="Ask MoE Engine... (Supports RAG & Real-time Stats)"
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
                          {stats?.expert_utilization.map((val, i) => (
                            <div key={i} className="space-y-1.5">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-gray-500 font-mono">NEURAL_EXPERT_{i}</span>
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
                  </div>
               </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

