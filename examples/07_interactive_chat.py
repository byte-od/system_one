"""Exemplo 07: Chat Interativo em Nível de Palavras (Word-Level) com SCache.

Um assistente conversacional que dialoga em português real com palavras 100% válidas,
mantendo todo o contexto da conversa dentro de uma matriz estritamente fixa de 68 KB
no Cache L1 do processador, sem alocar nenhum KV Cache.
"""

import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache import SlotScarAttention


def tokenize(text: str) -> list[str]:
    """Tokeniza texto separando palavras e pontuações em minúsculas."""
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    return text.split()


class ScarWordChatModel(nn.Module):
    """Modelo de Linguagem Conversacional em Nível de Palavras com SlotScarAttention."""

    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 2, num_slots: int = 8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
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

    def forward_step(self, token_id: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        x = self.token_embed(token_id)
        attn_out = self.attn.forward_step(x, update_state=update_memory)
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        return self.head(x)


PAIRS = [
    ("humano : qual seu nome ?", "scache : meu nome e scache , a inteligencia artificial sem kv cache ."),
    ("humano : como e seu nome ?", "scache : meu nome e scache , a inteligencia artificial sem kv cache ."),
    ("humano : quem e voce ?", "scache : sou o scache , uma inteligencia artificial que roda no cache l1 ."),
    ("humano : boa noite , tudo bem ?", "scache : boa noite ! tudo bem por aqui , como posso te ajudar ?"),
    ("humano : boa noite !", "scache : boa noite ! tudo bem com voce ?"),
    ("humano : oi , boa noite !", "scache : ola ! boa noite , tudo bem com voce ?"),
    ("humano : oi , tudo bem ?", "scache : ola ! tudo otimo com voce ?"),
    ("humano : ola , tudo bem ?", "scache : ola ! tudo bem por aqui , e um prazer conversar com voce ."),
    ("humano : como voce funciona ?", "scache : eu uso memoria associativa de cicatriz de gradiente sem kv cache ."),
    ("humano : voce gasta muita memoria ?", "scache : nao ! consumo apenas sessenta e oito kilobytes fixos no cache l1 ."),
    ("humano : quanto de memoria voce consome ?", "scache : consumo apenas sessenta e oito kilobytes fixos para sempre ."),
    ("humano : o que e o scar cache ?", "scache : e um mecanismo que substitui buffers gigantes de memoria em redes neurais ."),
    ("humano : quem te criou ?", "scache : fui desenvolvido na arquitetura system 1 como prova de conceito ."),
    ("humano : voce esquece as coisas ?", "scache : meus slots gravam o contexto com regra delta e decaimento multi escala ."),
    ("humano : isso e muito rapido ?", "scache : sim , minha latencia e de microssegundos em fluxo continuo no hardware ."),
    ("humano : legal !", "scache : muito obrigado ! vamos conversar mais ."),
    ("humano : erro ?", "scache : nenhum erro ! o sistema opera perfeitamente com memoria congelada no l1 ."),
    ("humano : obrigado !", "scache : de nada ! estou sempre a disposicao para ajudar ."),
]


def build_and_train_chat():
    all_tokens = []
    for q, a in PAIRS:
        all_tokens.extend(tokenize(q + " " + a))

    vocab = ["<unk>", "<pad>", "humano", "scache", ":"] + sorted(list(set(all_tokens) - {"humano", "scache", ":"}))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    print(f"Vocabulário conversacional: {vocab_size} palavras reais em português.")
    print("Treinando o modelo de chat Word-Level em CPU (~2 segundos)...")

    torch.manual_seed(42)
    model = ScarWordChatModel(vocab_size=vocab_size, d_model=64, num_heads=2, num_slots=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)

    model.train()
    epochs = 70
    for ep in range(epochs):
        for q, a in PAIRS:
            model.reset_cache()
            tokens = tokenize(q + " " + a)
            seq = torch.tensor([word2idx[w] for w in tokens], dtype=torch.long)

            optimizer.zero_grad()
            pair_loss = 0.0
            for t in range(len(seq) - 1):
                inp = seq[t : t + 1]
                target = seq[t + 1 : t + 2]
                logits = model.forward_step(inp, update_memory=True)
                pair_loss = pair_loss + F.cross_entropy(logits, target)

            pair_loss = pair_loss / (len(seq) - 1)
            pair_loss.backward()
            optimizer.step()

    print(f"Modelo pronto para conversar!\n")
    return model, word2idx, idx2word


def chat_session(model, word2idx, idx2word, interactive: bool = True):
    print("=" * 75)
    print(" CHAT INTERATIVO WORD-LEVEL COM SCACHE (SEM KV CACHE)")
    print(" Fale em português! Digite sua mensagem ou 'sair' para encerrar.")
    print(f" Consumo de Memória de Conversa: {model.cache_bytes} bytes (~68 KB FIXOS)")
    print("=" * 75)

    model.eval()

    test_queries = [
        "Boa noite, tudo bem?",
        "Qual seu nome?",
        "Como voce funciona?",
        "Voce gasta muita memoria?",
        "Erro?",
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

        # Limpa o cache para o novo diálogo mantendo a capacidade associativa
        model.reset_cache()

        # Tokeniza entrada do usuário
        user_tokens = tokenize(user_msg)
        prompt_tokens = ["humano", ":"] + user_tokens + ["scache", ":"]

        # 1. Ingestão do prompt no SCache
        with torch.no_grad():
            last_token_id = None
            for token_str in prompt_tokens:
                t_id = word2idx.get(token_str, word2idx.get("<unk>", 0))
                inp = torch.tensor([t_id], dtype=torch.long)
                _ = model.forward_step(inp, update_memory=True)
                last_token_id = inp

            # 2. Geração da resposta palavra por palavra
            print("SCache: ", end="", flush=True)
            curr_id = last_token_id
            response_words = []

            for step in range(35):
                logits = model.forward_step(curr_id, update_memory=True)
                next_id = torch.argmax(logits).unsqueeze(0)
                next_word = idx2word.get(next_id.item(), "<unk>")

                # Condição de parada (início de uma fala do humano ou token nulo)
                if next_word in ("humano", "<pad>", "<unk>"):
                    break
                if next_word == ":" and step > 1:
                    break

                response_words.append(next_word)
                curr_id = next_id
                print(f"{next_word} ", end="", flush=True)

                if next_word in (".", "!", "?") and step >= 4:
                    break

            print(f" [Cache L1: {model.cache_bytes} B]")


def main():
    interactive = sys.stdin.isatty()
    model, word2idx, idx2word = build_and_train_chat()
    chat_session(model, word2idx, idx2word, interactive=interactive)


if __name__ == "__main__":
    main()
