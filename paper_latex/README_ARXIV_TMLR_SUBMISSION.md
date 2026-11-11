# LaTeX Manuscript Source (arXiv / TMLR / Double-Blind Workshops)

## Quick Upload to Overleaf
1. Go to https://www.overleaf.com/project -> **New Project** -> **Upload Project**.
2. Drop `MoS_SCache_LaTeX_arXiv_TMLR_v1.0.zip`. It compiles immediately with `pdfLaTeX`.

## Switching Between arXiv Preprint and Double-Blind Peer Review
Open `main.tex` line 10:
- **arXiv / Preprint (Author & Zenodo DOI shown):** `\anonymousfalse`
- **Double-Blind Peer Review (TMLR / NeurIPS / ICML / ICLR Workshop):** change line 10 to `\anonymoustrue`

## Submitting to arXiv (`cs.LG` primary, `cs.CL` secondary)
1. Upload `MoS_SCache_LaTeX_arXiv_TMLR_v1.0.zip` at https://arxiv.org/submit.
2. Set **Report-no / DOI** field to: `10.5281/zenodo.24143839`.
3. License: `CC BY 4.0`.
