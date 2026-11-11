"""
System One — Compass 1: 120D Host-Aware Forensic Frontier Scouts (Bússola 1)
============================================================================
Extrai determinísticamente 120 assinaturas criptográficas de tipografia Unicode,
geometria estrutural Markdown, explosividade (burstiness) rítmica de sentenças
e transições retóricas Ghost-Jump calibradas para o gerador oficial da competição.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUM_LIST_RE = re.compile(r"(?:^|\n)\s*\d+[\.\)]\s")
_BUL_LIST_RE = re.compile(r"(?:^|\n)\s*[-*•]\s")
_HDR_RE = re.compile(r"(?:^|\n)#{1,6}\s")
_ITALIC_SINGLE_RE = re.compile(r"(?<!\*)\*(?!\*|\s)[^*\n]{1,80}(?<!\s|\*)\*(?!\*)")
_UNSPACED_EMDASH_RE = re.compile(r"\w\u2014\w")
_SPACED_EMDASH_RE = re.compile(r"\s\u2014\s")
_UK_SPELLINGS = frozenset(
    "colour colours coloured behaviour behaviours centre centres metre metres "
    "kilometre kilometres centimetre centimetres millimetre millimetres litre litres "
    "fibre fibres fibreglass tyre tyres analyse analyses analysed organise organises "
    "organised recognise recognises recognised realise realises realised优化 "
    "optimise optimises optimisedグレー grey honour honours honoured favour favours "
    "favoured neighbour neighbours دفاع defence offence license practise travelled "
    "travelling cancelled cancelling labelled labelling whilst amongst towards".split()
)
_US_SPELLINGS = frozenset(
    "color colors colored behavior behaviors center centers meter meters "
    "kilometer kilometers centimeter centimeters millimeter millimeters liter liters "
    "fiber fibers fiberglass tire tires analyze analyzes analyzed organize organizes "
    "organized recognize recognizes recognized realize realizes realized "
    "optimize optimizes optimized gray honor honors honored favor favors "
    "favored neighbor neighbors defense offense traveled traveling canceled "
    "canceling labeled labeling toward".split()
)

_GHOST_PHRASES = [
    "however,",
    "moreover,",
    "furthermore,",
    "therefore,",
    "thus,",
    "instead,",
    "rather,",
    "in contrast,",
    "on the other hand",
    "for example,",
    "for instance,",
    "specifically,",
    "in particular,",
    "overall,",
    "in short,",
    "in summary,",
    "in conclusion",
    "ultimately,",
    "first,",
    "second,",
    "finally,",
    "imagine ",
    "think of ",
    "consider ",
    "picture ",
    "at its core",
    "in essence",
    "simply put",
    "in other words",
    "here is ",
    "here's ",
    "let's ",
    "to understand",
    "that means",
    "this means",
    "because of this",
    "as a result",
    "e.g.",
    "i.e.",
]

_OPENING_STARTERS = (
    "when ",
    "while ",
    "although ",
    "if ",
    "imagine ",
    "think ",
    "in ",
    "the ",
    "a ",
    "to ",
)


def extract_forensic_compass1_120d(texts: Iterable[str]) -> np.ndarray:
    """
    Retorna matriz (N, 120) float32 com assinaturas forenses livres de NaN/Inf.
    """
    rows: List[List[float]] = []
    for raw in texts:
        t = "" if raw is None else str(raw)
        n_chars = len(t)
        nc = float(max(n_chars, 1))

        words_raw = _WORD_RE.findall(t)
        nw_int = len(words_raw)
        nw = float(max(nw_int, 1))
        h100 = 100.0 / nw

        # Normaliza apóstrofos curvos para análise lexical sem perder contagem Unicode bruta
        words_low = [w.lower().replace("\u2019", "'") for w in words_raw]

        # ------------------------------------------------------------------
        # 1. Tipografia Criptográfica Unicode & Pontuação Fina (32 dims)
        # ------------------------------------------------------------------
        c_ldq = t.count("\u201c")
        c_rdq = t.count("\u201d")
        c_curly_dq = c_ldq + c_rdq
        c_str_dq = t.count('"')
        tot_dq = c_curly_dq + c_str_dq

        c_lsq = t.count("\u2018")
        c_rsq = t.count("\u2019")
        c_curly_sq = c_lsq + c_rsq
        c_str_sq = t.count("'")
        tot_sq = c_curly_sq + c_str_sq

        c_em = t.count("\u2014")
        c_em_unspaced = len(_UNSPACED_EMDASH_RE.findall(t))
        c_em_spaced = len(_SPACED_EMDASH_RE.findall(t))
        c_en = t.count("\u2013")
        c_hyp = t.count("-")
        c_dhyp = t.count("--")
        c_shyp = t.count(" - ")
        c_ell_uni = t.count("\u2026")
        c_ell_asc = t.count("...")
        c_semi = t.count(";")
        c_colon = t.count(":")
        c_comma = t.count(",")
        c_excl = t.count("!")
        c_quest = t.count("?")
        c_slash = t.count("/")
        c_amp = t.count("&")
        c_pct = t.count("%")
        c_lparen = t.count("(")
        c_lbrack = t.count("[")
        non_ascii = sum(1 for ch in t if ord(ch) > 127)

        f_typo = [
            1.0 if c_curly_dq > 0 else 0.0,
            c_curly_dq * h100,
            c_str_dq * h100,
            c_curly_dq / (tot_dq + 1e-6) if tot_dq > 0 else 0.0,
            1.0 if tot_dq > 0 else 0.0,
            1.0 if c_curly_sq > 0 else 0.0,
            c_curly_sq * h100,
            c_str_sq * h100,
            c_curly_sq / (tot_sq + 1e-6) if tot_sq > 0 else 0.0,
            1.0 if tot_sq > 0 else 0.0,
            1.0 if c_em > 0 else 0.0,
            c_em * h100,
            c_em_unspaced * h100,
            c_em_spaced * h100,
            c_em_unspaced / (c_em + 1e-6) if c_em > 0 else 0.0,
            1.0 if c_en > 0 else 0.0,
            c_en * h100,
            c_hyp * h100,
            c_dhyp * h100,
            c_shyp * h100,
            (c_ell_uni + c_ell_asc) * h100,
            1.0 if c_semi > 0 else 0.0,
            c_semi * h100,
            1.0 if c_colon > 0 else 0.0,
            c_colon * h100,
            c_comma * h100,
            c_excl * h100,
            c_quest * h100,
            c_slash * h100,
            (c_amp + c_pct) * h100,
            c_lparen * h100,
            non_ascii / nc,
        ]  # 32

        # ------------------------------------------------------------------
        # 2. Geometria Estrutural Markdown & Parágrafos (20 dims)
        # ------------------------------------------------------------------
        c_bold = t.count("**") // 2
        c_ital = len(_ITALIC_SINGLE_RE.findall(t))
        c_code_fence = t.count("```") // 2
        c_backtick = max(0, t.count("`") - 6 * c_code_fence)
        c_hdr = len(_HDR_RE.findall(t))
        c_nlist = len(_NUM_LIST_RE.findall(t))
        c_blist = len(_BUL_LIST_RE.findall(t))
        c_bquote = t.count("\n>")
        c_lbrace = c_lbrack + t.count("{")

        paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
        n_paras = max(len(paras), 1)
        para_lens = [max(len(_WORD_RE.findall(p)), 1) for p in paras] if paras else [nw_int]
        c_nl = t.count("\n")
        c_dnl = t.count("\n\n")

        f_struct = [
            1.0 if c_bold > 0 else 0.0,
            c_bold * h100,
            1.0 if c_ital > 0 else 0.0,
            c_ital * h100,
            c_backtick * h100,
            float(c_code_fence),
            float(c_hdr),
            1.0 if (c_nlist + c_blist) > 0 else 0.0,
            c_nlist * h100,
            c_blist * h100,
            float(c_bquote),
            c_lbrace * h100,
            float(n_paras),
            1.0 if n_paras == 1 else 0.0,
            float(np.mean(para_lens)) / 100.0,
            float(np.std(para_lens)) / 100.0 if n_paras > 1 else 0.0,
            para_lens[0] / nw,
            para_lens[-1] / nw,
            c_nl * h100,
            c_dnl * h100,
        ]  # 20 (cum 52)

        # ------------------------------------------------------------------
        # 3. Ritmo de Sentenças, Burstiness & Riqueza Lexical (24 dims)
        # ------------------------------------------------------------------
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]
        n_sents = max(len(sents), 1)
        sent_lens = [max(len(_WORD_RE.findall(s)), 1) for s in sents] if sents else [nw_int]
        s_arr = np.asarray(sent_lens, dtype=np.float32)
        s_mean = float(s_arr.mean())
        s_std = float(s_arr.std()) if len(s_arr) > 1 else 0.0
        s_med = float(np.median(s_arr))
        s_min = float(s_arr.min())
        s_max = float(s_arr.max())

        w_lens = np.asarray([len(w) for w in words_raw], dtype=np.float32) if nw_int > 0 else np.zeros(1, dtype=np.float32)
        counts = Counter(words_low)
        n_uniq = len(counts)
        hapax = sum(1 for v in counts.values() if v == 1)
        m2 = sum(v * v for v in counts.values())
        yule_k = 1e4 * (m2 - nw) / (nw * nw + 1e-6) if nw_int > 1 else 0.0

        upper_chars = sum(1 for ch in t if ch.isupper())
        digit_chars = sum(1 for ch in t if ch.isdigit())
        title_words = sum(1 for w in words_raw if w[0].isupper())
        allcaps_words = sum(1 for w in words_raw if len(w) >= 2 and w.isupper())
        num_tokens = sum(1 for w in words_raw if any(ch.isdigit() for ch in w))

        f_rhythm = [
            nw / 100.0,
            math.log1p(nw_int),
            nc / 500.0,
            float(w_lens.mean()) / 10.0,
            float(w_lens.std()) / 10.0,
            float(np.percentile(w_lens, 90)) / 10.0,
            float((w_lens >= 8).mean()),
            float((w_lens <= 3).mean()),
            n_uniq / nw,
            hapax / nw,
            min(yule_k / 200.0, 5.0),
            upper_chars / nc,
            digit_chars / nc,
            title_words / nw,
            allcaps_words * h100,
            num_tokens * h100,
            float(n_sents) / 10.0,
            s_mean / 25.0,
            s_std / 15.0,
            s_med / 25.0,
            s_min / 25.0,
            s_max / 40.0,
            s_std / (s_mean + 1e-6),
            c_comma / float(n_sents),
        ]  # 24 (cum 76)

        # ------------------------------------------------------------------
        # 4. Batedores Ghost-Jump: Contrações, UK/US & Discurso (44 dims)
        # ------------------------------------------------------------------
        t_low = t.lower().replace("\u2019", "'")
        n_nt = sum(1 for w in words_low if w.endswith("n't"))
        n_re_ve_ll = sum(1 for w in words_low if w.endswith(("'re", "'ve", "'ll", "'d")))
        n_its_contr = words_low.count("it's") + words_low.count("that's") + words_low.count("there's")
        n_it_is = t_low.count("it is ") + t_low.count("that is ") + t_low.count("do not ") + t_low.count("does not ") + t_low.count("cannot ")
        uk_hits = sum(1 for w in words_low if w in _UK_SPELLINGS)
        us_hits = sum(1 for w in words_low if w in _US_SPELLINGS)

        t_strip_low = t_low.lstrip()
        starts_question = 1.0 if (sents and sents[0].endswith("?")) else 0.0
        ends_exclamation = 1.0 if t.rstrip().endswith("!") else 0.0
        first_starter = 1.0 if t_strip_low.startswith(_OPENING_STARTERS[:5]) else 0.0

        f_ghost = [
            (n_nt + n_re_ve_ll + n_its_contr) * h100,
            n_nt * h100,
            n_it_is * h100,
            (n_nt + n_its_contr) / (n_nt + n_its_contr + n_it_is + 1e-6),
            uk_hits * h100,
            us_hits * h100,
            starts_question,
            ends_exclamation,
            first_starter,
        ]  # 9 -> remaining 35 from _GHOST_PHRASES[:35]

        for ph in _GHOST_PHRASES[:35]:
            f_ghost.append(t_low.count(ph) * h100)

        vec = f_typo + f_struct + f_rhythm + f_ghost
        assert len(vec) == 120, f"Expected 120 features, got {len(vec)}"
        rows.append(vec)

    out = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)
