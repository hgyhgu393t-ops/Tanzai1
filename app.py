import gradio as gr
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import math
import urllib.request
import os
import gdown  # <--- เพิ่ม gdown เพื่อโหลดไฟล์จาก Google Drive
from bs4 import BeautifulSoup
from googlesearch import search

# ==========================================
# ⚡ ระบบดาวน์โหลดไฟล์โมเดลอัตโนมัติจาก Google Drive
# ==========================================
MODEL_FILE = 'advanced_gemini.pt'

# 📌 คำแนะนำ: แทนแค่นำ File ID ของไฟล์ advanced_gemini.pt จาก Google Drive มาวางแทนที่ตรงนี้ครับ
# (ดู ID จากลิงก์แชร์ เช่น drive.google.com/file/d/FILE_ID_ตรงนี้/view)
GDRIVE_FILE_ID = 'ใส่_FILE_ID_จาก_GOOGLE_DRIVE_ตรงนี้' 

if not os.path.exists(MODEL_FILE):
    print("⏳ ไม่พบไฟล์โมเดลในเซิร์ฟเวอร์ กำลังดาวน์โหลดจาก Google Drive...")
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    gdown.download(url, MODEL_FILE, quiet=False)
    print("ดาวน์โหลดโมเดลเรียบร้อยแล้ว!")

# ==========================================
# 0. คลาสตัวตัดคำระดับอักขระดั้งเดิม
# ==========================================
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]
        
    def decode(self, l):
        return ''.join([self.itos[i] for i in l if i in self.itos])

import __main__
__main__.CharTokenizer = CharTokenizer

# ==========================================
# 1. โครงสร้างสถาปัตยกรรมโมเดลขั้นสูง
# ==========================================
n_embd = 768           
block_size = 256       
n_heads = 12           
n_kv_heads = 4         
n_layers = 8           
ffn_hidden_dim = 2048  

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * (1.0 + self.weight)

class GemmaSwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_in, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2) 
    sin = sin.unsqueeze(0).unsqueeze(2) 
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaGroupedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = n_embd // n_heads
        self.num_local_heads = n_heads
        self.num_local_kv_heads = n_kv_heads
        self.num_queries_per_kv = n_heads // n_kv_heads
        
        self.q_proj = nn.Linear(n_embd, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, n_embd, bias=False)
        
        self.rope = GemmaRotaryEmbedding(self.head_dim)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_local_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_local_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_local_kv_heads, self.head_dim)
        
        cos, sin = self.rope(q, T)
        q, k = apply_rope(q, k, cos, sin)
        
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 
        
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.tanh(scores / 10.0) * 10.0 
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_probs = F.softmax(scores, dim=-1)
        
        output = attention_probs @ v 
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(output)

class GemmaDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = GemmaGroupedAttention()
        self.ffn = GemmaSwiGLU(n_embd, ffn_hidden_dim)
        self.input_layernorm = GemmaRMSNorm(n_embd)
        self.post_attention_layernorm = GemmaRMSNorm(n_embd)
    def forward(self, x):
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.ffn(self.post_attention_layernorm(x))
        return x

class DeepTanzGemmaModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([GemmaDecoderBlock() for _ in range(n_layers)])
        self.norm = GemmaRMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, idx):
        x = self.embed(idx) * math.sqrt(n_embd)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

# โหลดระบบตัดคำศัพท์และไฟล์โมเดล
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = DeepTanzGemmaModel(tokenizer.vocab_size)
state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# ==========================================
# 2. ฟังก์ชันเสริมระบบ Google Live Search
# ==========================================
def fetch_google_knowledge(query):
    try:
        search_results = list(search(query, num_results=1, lang="th"))
        if not search_results:
            return ""
        
        target_url = search_results[0]
        req = urllib.request.Request(target_url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=5).read()
        
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = " ".join(chunk for chunk in chunks if chunk)
        
        return clean_text[:180]
    except Exception:
        return ""

# ==========================================
# 3. เอนจิ้นประมวลผลข้อความ
# ==========================================
def chat_engine_stream(user_input, history):
    full_prompt = f"Q: {user_input}\nA: "
    idx = torch.tensor([tokenizer.encode(full_prompt)], dtype=torch.long)
    max_new_tokens = 180
    generated_tokens = []
    
    with torch.no_grad():
        initial_logits = model(idx[:, -block_size:])[:, -1, :]
        probs = F.softmax(initial_logits, dim=-1)
        max_prob, _ = torch.max(probs, dim=-1)
        
        if max_prob.item() < 0.15:
            yield "⏳ ข้อมูลนี้ไม่อยู่ในหน่วยความจำเดิม... กำลังค้นหา Google เรียบลไทม์ให้ครับแทน..."
            live_info = fetch_google_knowledge(user_input)
            
            if live_info:
                full_prompt = f"ข้อมูลเพิ่มเติมจากกูเกิ้ล: {live_info}\nQ: {user_input}\nA: "
                idx = torch.tensor([tokenizer.encode(full_prompt)], dtype=torch.long)
            else:
                yield "ผมไม่สามารถตอบคำถามนี้ได้เนื่องจากไม่พบคลังข้อมูลบนระบบอินเทอร์เน็ตครับ"
                return

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
            
        v, ix = torch.topk(logits, k=3)  
        filtered_logits = torch.full_like(logits, -float('Inf'))
        filtered_logits.scatter_(1, ix, v)
        
        idx_next = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        generated_tokens.append(idx_next.item())
        generated_text = tokenizer.decode(generated_tokens)
        
        if "\n" in generated_text or "Q:" in generated_text or "A:" in generated_text:
            clean_output = generated_text.replace("\n", "").replace("Q:", "").replace("A:", "").strip()
            if not clean_output:
                yield "ผมไม่สามารถหาข้อสรุปจากเนื้อหาหน้าเว็บนี้ได้ครับ"
            else:
                yield clean_output
            break
            
        yield generated_text.strip()

# ==========================================
# 4. หน้ากากแอปพลิเคชัน Gradio
# ==========================================
custom_css = """
footer {visibility: hidden !important}
body, .gradio-container {
    background-color: #0d0e12 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    max-width: 950px !important;
    margin: 0 auto !important;
    color: #e3e3e3 !important;
}
.center-header {
    text-align: center;
    margin-top: 50px;
    margin-bottom: 30px;
}
.center-header h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #1ba2f6 0%, #a252ff 50%, #f6517a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px;
}
.center-header p { color: #9aa0a6 !important; font-size: 1.15rem !important; }
.chatbot { border: none !important; background-color: #0d0e12 !important; }
.chatbot .user { background-color: #1e1f24 !important; color: #ffffff !important; border-radius: 22px 22px 4px 22px !important; }
.chatbot .bot { background-color: transparent !important; color: #e3e3e3 !important; }
.gradio-container .buttons { display: none !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(
        """
        <div class="center-header">
            <h1>✦ Tanz Gemma Live Search Engine ⚡</h1>
            <p>สถาปัตยกรรมกลุ่มหัวเรือเชื่อมต่อโครงข่าย Google ค้นหาข้อมูลแบบพลวัตภายนอก</p>
        </div>
        """
    )
    gr.ChatInterface(fn=chat_engine_stream)

# 🌐 ปรับระบบรันให้รองรับ Render.com
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
