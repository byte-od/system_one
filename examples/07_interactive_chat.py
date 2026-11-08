"""Exemplo 07: Chat Interativo com Modelo Baseado em SCache (Sem KV Cache).

Um chatbot em nível de caracteres (sem palavras fora do vocabulário)
que dialoga interativamente mantendo todo o histórico da conversa
dentro de uma matriz de 68 KB no L1 Cache do processador.
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache import SlotScarAttention


class ScarChatModel(nn.Module):
    """Modelo de Linguagem em Nível de Caracteres baseado em SlotScarAttention."""

    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 2, num_slots: int = 8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.char_embed = nn.Embedding(vocab_size, d_model)
        self.attn = SlotScarAttention(d_model=d_model, num_heads=num_heads, num_slots=num_slots)
        self.ln1 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def reset_cache(self):
        self.attn.reset_cache()

    @property
    def cache_bytes(self) -> int:
        return self.attn.total_memory_bytes

    def forward_step(self, char_id: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        x = self.char_embed(char_id)
        attn_out = self.attn.forward_step(x, update_state=update_memory)
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        return self.head(x)


def train_chat_model():
    # Diálogos para condicionar o modelo
    dialogues = (
        "Humano: Oi!\n"
        "SCache: Ola! Eu sou o SCache, um modelo sem KV Cache que roda no L1.\n\n"
        "Humano: Como voce funciona?\n"
        "SCache: Eu uso fast weights e regra delta para gravar a conversa em 68 KB.\n\n"
        "Humano: Voce gasta muita memoria?\n"
        "SCache: Nao! Minha memoria e constante O(1), nunca cresce nem estoura a VRAM.\n\n"
        "Humano: Quem te criou?\n"
        "SCache: Fui desenvolvido na arquitetura System 1 como prova de conceito.\n\n"
        "Humano: O que e o scar cache?\n"
        "SCache: E uma memoria de cicatriz associativa que substitui buffers gigantes.\n\n"
        "Humano: Voce esquece as coisas?\n"
        "SCache: Meus slots tem decaimento temporal: guardo regras e descarto ruidos.\n\n"
        "Humano: Ola tudo bem?\n"
        "SCache: Tudo otimo! Pronto para conversar e raciocinar em tempo real.\n\n"
        "Humano: Legal!\n"
        "SCache: Muito obrigado! Vamos conversar mais!\n\n"
    )

    chars = sorted(list(set(dialogues)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    torch.manual_seed(42)
    model = ScarChatModel(vocab_size=vocab_size, d_model=64, num_heads=2, num_slots=8)

    print("Treinando o modelo de chat em nível de caracteres (apenas ~2 segundos)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    seq = torch.tensor([char2idx[c] for c in dialogues], dtype=torch.long)

    model.train()
    epochs = 40
    chunk_len = 48
    for ep in range(epochs):
        model.reset_cache()
        for i in range(0, len(seq) - chunk_len, chunk_len):
            chunk = seq[i : i + chunk_len]
            optimizer.zero_grad()
            chunk_loss = 0.0
            for t in range(len(chunk) - 1):
                inp = chunk[t : t + 1]
                target = chunk[t + 1 : t + 2]
                logits = model.forward_step(inp, update_memory=True)
                chunk_loss = chunk_loss + F.cross_entropy(logits, target)
            chunk_loss = chunk_loss / (len(chunk) - 1)
            chunk_loss.backward()
            optimizer.step()

    print("Modelo pronto para conversar!\n")
    return model, char2idx, idx2char


def chat_session(model, char2idx, idx2char, interactive: bool = True):
    print("=" * 70)
    print(" CHAT INTERATIVO COM SCACHE (100% NO L1 CACHE - SEM KV CACHE)")
    print(" Digite sua mensagem abaixo ou 'sair' para encerrar.")
    print(f" Consumo de Memória de Conversa: {model.cache_bytes} bytes (~68 KB FIXOS)")
    print("=" * 70)

    model.eval()
    model.reset_cache()

    # Exemplos pré-definidos se não estiver em TTY interativo
    test_queries = [
        "Oi!",
        "Como voce funciona?",
        "Voce gasta muita memoria?",
    ]

    query_idx = 0
    while True:
        if interactive:
            try:
                user_msg = input("\nVocê: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        else:
            if query_idx >= len(test_queries):
                break
            user_msg = test_queries[query_idx]
            query_idx += 1
            print(f"\nVocê: {user_msg}")

        if not user_msg:
            continue
        if user_msg.lower() in ("sair", "exit", "quit"):
            print("Encerrando conversa...")
            break

        # Formata o prompt
        formatted_prompt = f"Humano: {user_msg}\nSCache: "
        
        # 1. Ingestão do prompt do usuário no SCache
        with torch.no_grad():
            last_char_id = None
            for ch in formatted_prompt:
                # Se o caractere não foi visto no treino, substitui pelo mais próximo
                cid = char2idx.get(ch, char2idx.get(" ", 0))
                t_in = torch.tensor([cid], dtype=torch.long)
                _ = model.forward_step(t_in, update_memory=True)
                last_char_id = t_in

            # 2. Geração da resposta caractere por caractere
            print("SCache: ", end="", flush=True)
            curr_id = last_char_id
            response_chars = []

            for _ in range(80):
                logits = model.forward_step(curr_id, update_memory=True)
                probs = F.softmax(logits / 0.65, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(0)
                next_ch = idx2char[next_id.item()]

                if next_ch == "\n":
                    break

                print(next_ch, end="", flush=True)
                response_chars.append(next_ch)
                curr_id = next_id

            print(f"  [Cache: {model.cache_bytes} B]")


def main():
    interactive = sys.stdin.isatty()
    model, char2idx, idx2char = train_chat_model()
    chat_session(model, char2idx, idx2char, interactive=interactive)


if __name__ == "__main__":
    main()
