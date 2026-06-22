# figure_prompts.md
"""
Prompts para gerar as figuras do README de Causal Trees
(Regressão · Binário · Multiclasse).

Para gerar TODAS as figuras técnicas de uma vez, execute:
    python generate_figures.py

Para a figura de arte conceitual (fig00) use um gerador de imagem
(DALL·E 3, Midjourney v6, Stable Diffusion XL) com o prompt abaixo.
"""

---

## Checklist de figuras

| Figura | Arquivo | Tipo | Seção do README |
|--------|---------|------|-----------------|
| fig00 | `figures/fig00.png` | Arte conceitual (banner) | Capa |
| fig01 | `figures/fig01.png` | Matplotlib — scatter + step-function | Seção 1 (Regressão) |
| fig02 | `figures/fig02.png` | Matplotlib — probabilidade binária | Seção 1.1 (Binário) |
| fig03 | `figures/fig03.png` | Matplotlib — scatter multiclasse + regiões | Seção 1.2 (Multiclasse) |
| fig04 | `figures/fig04.png` | Matplotlib — diagrama honesto vs. adaptativo | Seção 1.3 |
| fig05 | `figures/fig05.png` | Matplotlib — comparação dos 3 critérios | Seção 2.3 |
| fig06 | `figures/fig06.png` | Matplotlib — dot-plot por folha com CIs | Seção 2.2 |
| fig07 | `figures/fig07.png` | Matplotlib — heatmap CATE + política binária | Seção 2.6 |
| fig08 | `figures/fig08.png` | Matplotlib — heatmap tau_k multiclasse | Seção 2.7 |
| fig09 | `figures/fig09.png` | Matplotlib — walkthrough numérico (4 painéis) | Seção 2.8 |
| fig10 | `figures/fig10.png` | Matplotlib — distribuições CATE (reg + bin) | Seção 2.8 |
| fig11 | `figures/fig11.png` | Matplotlib — curva de sweep max_depth | Seção 2.8 |

---

## fig00 — Banner / Capa

**Tipo:** Arte conceitual (gerador de imagem)
**Arquivo:** `figures/fig00.png`

### Prompt para DALL·E 3 / Midjourney v6:
```
A futuristic scientific poster for a TinyML tutorial on Causal Trees.
Center: a glowing decision tree made of light, with purple internal
rectangular nodes labeled with split rules (e.g. "X0 <= 0"), and four
leaf nodes colored in blue (Regression), red (Binary), green (Multiclass),
and orange (Multiclass), each showing a small tau-hat value.
Left and right: scattered blue circles (treated units) and orange triangles
(control units) floating in the background.
Bottom right: a small Arduino/ESP32 microcontroller connected by data
streams to the tree, symbolising embedded deployment.
Background: deep navy blue (#0D1B2A) with a subtle grid of faint circuit-
board lines and small white dots.
Header text: "TinyML — Causal Trees" in white bold sans-serif, 22pt.
Subtitle: "Regression · Binary · Multiclass" in light gray (#AEB6BF), 12pt.
Style: modern scientific illustration, 4K detail, cinematic lighting.
Aspect ratio: 14:4 (wide banner format).
```

### Prompt alternativo (Claude — gerar como SVG):
```
Crie um banner SVG 1400×400px estilo dark-mode para um tutorial de
Causal Trees com as seguintes características:

Layout:
- Fundo: gradiente horizontal de #0D1B2A (esquerda) para #1A2744 (direita).
- Pontos dispersos aleatórios (2px, cor #1E3A5F, alpha 0.5) como fundo.

Elemento central — Árvore de decisão:
- 7 nós internos retangulares (FancyBboxPatch) cor #7B2D8B, texto branco.
- 4 folhas circulares coloridas:
    azul #3498DB (label "Reg"), vermelho #E74C3C (label "Bin"),
    verde #2ECC71 (label "MC"), laranja #F39C12 (label "MC").
- Setas cinzas conectando nós com pais.

Scatter (pontos de dados):
- Esquerda e direita da árvore: círculos azuis (#2E86AB) e triângulos
  laranjas (#E84855), tamanho 26px, alpha 0.7.

Texto:
- "TinyML — Causal Trees" — centro superior, branco, bold, 20pt,
  com text-shadow escuro.
- "Regression · Binary · Multiclass  |  Heterogeneous Treatment Effect
  Estimation" — abaixo, cor #AEB6BF, 10pt.
```

---

## fig01 — Regressão: HTE como função degrau

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig01.png`

```python
"""
Dois painéis lado a lado (figsize=(13, 5)):

Dados gerados:
    np.random.seed(42)
    n = 300
    X0 = np.random.uniform(-2, 2, n)
    W  = (np.random.rand(n) > 0.5).astype(int)
    tau_true = np.where(X0 > 0, 2.0, 0.5)
    Y  = tau_true * W + 0.4 * X0 + np.random.randn(n) * 0.3
    ATE = tau_true.mean()   # ≈ 1.25

Painel esquerdo — "Observed Outcomes by Treatment Assignment":
- Scatter: X0 (eixo x) vs Y (eixo y).
  Azuis círculos = tratados (W=1); laranjas triângulos = controles (W=0).
- Linha horizontal tracejada roxa: ATE = {ATE:.2f}.
- Legenda: "Treated (W=1)", "Control (W=0)", "ATE".
- Grid alpha=0.3.

Painel direito — "True Heterogeneous Treatment Effect tau*(x)":
- Linha sólida roxa lw=2.8: step function tau*(x) = 2.0 se x > 0 else 0.5.
- Banda sombreada roxa alpha=0.15: ±0.1 ao redor da linha.
- Linha horizontal tracejada cinza: ATE.
- Linha vertical pontilhada preta em X0=0 (boundary).
- Anotações:
    "tau*=2.0 (high responders)" acima, à direita de X0=0.
    "tau*=0.5 (low responders)" abaixo, à esquerda de X0=0.
- Eixo Y: "CATE tau*(x)". Ylim = (-0.3, 2.6).

Título geral: "Regression Task — Heterogeneous Treatment Effect".
Cores: azul=#2E86AB, laranja=#E84855, roxo=#7B2D8B.
Salvar: 'figures/fig01.png', dpi=150, bbox_inches='tight'.
"""
```

---

## fig02 — Binário: Risco-diferença como CATE

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig02.png`

```python
"""
Dois painéis lado a lado (figsize=(13, 5)):

Dados gerados:
    n = 300
    X0 = np.random.randn(n); X1 = np.random.randn(n)
    e  = 1 / (1 + np.exp(-0.5 * X0))   # propensity varia com X0
    W  = (np.random.rand(n) < e).astype(int)
    tau_true = np.where(X0 > 0.5, 0.30, 0.05)
    p_base   = 1 / (1 + np.exp(-0.4 * X1))
    p_obs    = W * np.clip(p_base + tau_true, 0, 1) + (1 - W) * p_base
    Yb       = (np.random.rand(n) < p_obs).astype(int)

Painel esquerdo — "Binary Outcome — Observed Data":
- Scatter: X0 vs Yb (com jitter vertical ±0.035 para legibilidade).
  Azuis círculos = tratados; laranjas triângulos = controles.
- Eixo Y: "Y (binary)". Grid alpha=0.3.

Painel direito — "True Risk Difference tau*(x)":
- Linha sólida laranja lw=2.8: step function 0.30 se X0>0.5 else 0.05.
- Área preenchida laranja alpha=0.18 entre 0 e a linha.
- Linha horizontal tracejada cinza: ATE ≈ 0.175.
- Linha vertical pontilhada preta em X0=0.5.
- Ylim = (-0.05, 0.38).
- Legenda: "True risk difference tau*(x)", "ATE", "True boundary X0=0.5".

Título geral: "Binary Task — Risk Difference Treatment Effect".
Salvar: 'figures/fig02.png', dpi=150.
"""
```

---

## fig03 — Multiclasse: Regiões de efeito de tratamento

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig03.png`

```python
"""
Dois painéis lado a lado (figsize=(13, 5)):

Dados: K=3 classes (Mild, Moderate, Severe), n=300.
- Baseline: p_base via softmax de [−0.5·X1, 0.3·X1, 0.4·X0].
- Shift (tratamento):
    X0 > 0:  +0.35 para classe 2 (Severe),  −0.20 para classe 0 (Mild).
    X0 <= 0: +0.20 para classe 1 (Moderate), −0.10 para classe 0 (Mild).

Painel esquerdo — "Observed Classes (circles=T, triangles=C)":
- Scatter: X0 vs X1.
- Cores por classe: tab10(0)=Mild, tab10(1)=Moderate, tab10(2)=Severe.
- Tratados = círculos; controles = triângulos.
- Linha pontilhada vertical preta em X0=0.
- Legenda com patches coloridos por classe.

Painel direito — "Treatment-Effect Regions":
- Fundo X0 <= 0: fundo verde-azulado (tab10(1) alpha=0.22) — "Shift to Moderate".
- Fundo X0 >  0: fundo vermelho-suave (tab10(2) alpha=0.22) — "Shift to Severe".
- Linha vertical preta lw=2.0 em X0=0: "Treatment boundary X0=0".
- Scatter: azuis (tratados) e laranjas (controles) sobrepostos.
- Xlim=(-3,3), Ylim=(-3,3).

Título geral: "Multiclass Task — Per-Class Treatment Effect Shifts (K=3)".
Salvar: 'figures/fig03.png', dpi=150.
"""
```

---

## fig04 — Honesto vs. Adaptativo

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig04.png`

```python
"""
Dois painéis (figsize=(13, 6)):

Cada painel tem xlim=(0,5), ylim=(0,5), axis='off'.

PAINEL ESQUERDO — "Adaptive (Dishonest) Tree":
- Dataset S: scatter de pontos azuis (círculos) e laranjas (triângulos)
  dentro de uma caixa arredondada cinza clara rotulada "S (full dataset)".
- Seta roxa de S → nó raiz da árvore: label "Find split".
- Seta laranja de S → folhas: label "Estimate tau".
- Árvore: 3 nós retangulares laranjas (#E84855) + 2 folhas circulares.
- Borda tracejada laranja ao redor de tudo: label "SAME DATA ⚠".
- Texto no rodapé: "Biased estimates — invalid CIs".

PAINEL DIREITO — "Honest Tree":
- Caixa cinza à esquerda: S_str (pontos cinzas, quadrados).
- Caixa azul à direita: S_est (pontos azuis/laranjas).
- Seta roxa de S_str → nó raiz: label "Find split".
- Seta verde de S_est → folhas: label "Estimate tau".
- Árvore: 3 nós retangulares verdes + 2 folhas circulares.
- Borda tracejada verde: label "SEPARATE DATA ✓".
- Texto no rodapé: "Unbiased estimates — valid CIs".

Título geral: "Honest vs. Adaptive Estimation".
Salvar: 'figures/fig04.png', dpi=150.
"""
```

---

## fig05 — Comparação dos três critérios de split

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig05.png`

```python
"""
Três painéis horizontais (figsize=(15, 5), sharey=True):

Dados compartilhados:
    n=250, X0 ∈ [-2,2], tau = 2.0 se X0>0 else 0.3
    Y = tau * W + 0.6 * X0 + noise(0, 0.5)

Para cada painel (Variance, MSE, Tau-Risk):
1. Scatter: X0 vs Y, azul=tratados, laranja=controles, alpha=0.5.
2. Linha vertical com o split encontrado por cada critério:
   - Variance: X0=0.0, cor roxa, sólida — "True boundary X0=0".
   - MSE:      X0=0.7, cor cinza, tracejada — "Misses causal boundary".
   - Tau-Risk: X0=0.05, cor teal, sólida — "Consistent CATE proxy".
3. Score curve no topo do painel (curva quadrática com pico no split,
   cor igual à do critério, alpha=0.65, lw=1.6).
4. Texto "Score curve" dentro do painel, fontsize=7.5.
5. Linha vertical preta pontilhada (alpha=0.35) em X0=0 (referência verdadeira).
6. Texto descritivo abaixo da linha do split com resultado do critério,
   num bbox arredondado.

Eixo Y no painel esquerdo: "Y".
Título geral: "Split Criterion Comparison — Regression (true boundary X0=0)".
Salvar: 'figures/fig05.png', dpi=150.
"""
```

---

## fig06 — Folhas com Intervalos de Confiança

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig06.png`

```python
"""
Dot-plot horizontal (figsize=(10, 7)):

8 folhas com (tau, CI_lo, CI_hi, n_samples):
  (-0.15, -0.35,  0.05, 12)
  ( 0.05, -0.10,  0.20, 28)
  ( 0.22,  0.05,  0.40, 45)
  ( 0.48,  0.30,  0.65, 38)
  ( 0.75,  0.55,  0.95, 31)
  ( 1.10,  0.85,  1.35, 22)
  ( 1.45,  1.10,  1.80, 18)
  ( 2.10,  1.70,  2.50, 14)

Para cada folha i:
- Barra horizontal: [CI_lo, CI_hi], cor = viridis(norm(n_samples)), lw=2.2.
- Ponto circular ms=9 no valor tau.

Eixo Y: "Leaf 1" a "Leaf 8", com asterisco (*) nas folhas 3–8
(CI exclui 0).
Linha vertical tracejada cinza em x=0.
Colorbar à direita: "Leaf sample size".
Nota: "* CI excludes 0" em canto superior direito.
Título: "Per-Leaf CATE with 95% Bootstrap CI  (Regression)".
Salvar: 'figures/fig06.png', dpi=150.
"""
```

---

## fig07 — Heatmap de risco-diferença + política binária

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig07.png`

```python
"""
Dois painéis lado a lado (figsize=(13, 5)):

Grid: xx, yy = meshgrid(linspace(-2.5,2.5,80), linspace(-2.5,2.5,80))
tau_grid = np.where(xx > 0.5, 0.30, 0.05)

Painel esquerdo — "Risk Difference Surface  (Binary)":
- contourf com cmap='RdBu_r', vmin=-0.05, vmax=0.35, levels=20, alpha=0.85.
- Colorbar: "tau_hat (risk difference)".
- Scatter: navy círculos (tratados) e firebrick triângulos (controles).

Painel direito — "Optimal Binary Policy (X0 vs X1)":
- contourf com cmap='RdYlGn', 2 níveis (tau_grid > 0.15 = verde = tratar).
- contour linha preta em 0.5 (boundary de política), lw=1.5.
- Scatter: navy círculos e firebrick triângulos.
- Legenda: patch verde "Treat (tau>0.15)" e patch vermelho "No treat".

Título geral: "Binary Task — CATE Heatmap and Optimal Policy".
Salvar: 'figures/fig07.png', dpi=150.
"""
```

---

## fig08 — Heatmap tau_k multiclasse por folha

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig08.png`

```python
"""
Dois painéis lado a lado (figsize=(13, 5)):

tau_mc = array de shape (6, 3):
  [[-0.12,  0.18, -0.06],
   [-0.20,  0.05,  0.15],
   [-0.08,  0.28, -0.20],
   [ 0.05, -0.10,  0.05],
   [-0.25, -0.05,  0.30],
   [-0.18,  0.22, -0.04]]

Painel esquerdo — "Per-Leaf Per-Class tau_k Heatmap":
- imshow com cmap='RdBu_r', vmin=-vmax, vmax=vmax.
- Eixo X: ["Mild", "Moderate", "Severe"].
- Eixo Y: ["Leaf 1", ..., "Leaf 6"].
- Texto em cada célula: valor com sinal (+/-), branco se |val|>0.15.
- Colorbar: "tau_k  [P(Y=k|T) - P(Y=k|C)]".

Painel direito — "Predicted Class per Leaf  (argmax tau_k)":
- barh: barra horizontal por folha.
  Comprimento = max(tau_mc[i]).
  Cor = tab10(argmax(tau_mc[i])).
- Texto à direita de cada barra: nome da classe (argmax).
- Legenda: patches tab10 para cada classe.
- Linha vertical tracejada em x=0.

Título geral: "Multiclass — Per-Leaf Per-Class Treatment Effect Estimates".
Salvar: 'figures/fig08.png', dpi=150.
"""
```

---

## fig09 — Walkthrough numérico (4 painéis)

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig09.png`

```python
"""
4 painéis em grade 2×2 (figsize=(13, 10)):

Dados:
  pts_T = [[0.8,3.9],[0.7,4.2],[0.9,4.5],[0.3,1.8]]   # tratados
  pts_C = [[0.2,1.1],[0.4,1.3],[0.1,0.9],[0.6,2.1]]   # controles

Painel (a) "Raw Data (n=8)":
  Scatter com rótulos i=1..8, círculos azuis (T) e triângulos laranjas (C).
  Xlim=(-0.05,1.1), Ylim=(0.5,5.1).

Painel (b) "Honest Split: Structure vs. Estimation":
  Mesmos pontos divididos em:
  - S_str: unidades ímpares (1,3,5,7) — quadrados cinzas (#566573).
  - S_est: unidades pares  (2,4,6,8) — tratados azuis, controles laranjas.
  Caixas FancyBboxPatch tracejadas ao redor de cada metade.
  Rótulos "S_str" (cinza) e "S_est" (azul).

Painel (c) "Root Split Found: X0 <= 0.55":
  Mesmos pontos originais + linha vertical roxa em X0=0.55.
  Fundo azul claro à esquerda, laranja claro à direita.
  Anotações: "Left leaf (X0 <= 0.55)" e "Right leaf (X0 > 0.55)".
  Bbox de texto: "Best split: X0 <= 0.55\n(score = 1.62)".

Painel (d) "Leaf CATE Estimates (Honest)":
  axis='off'. Dois retângulos FancyBboxPatch (axes fraction):
  - Esquerdo: azul claro (#DBEAFE), texto tau=2.10, detalhe estimação.
  - Direito: vermelho claro (#FECACA), texto tau=4.20, detalhe estimação.
  Anotação de seta dividindo para cada folha.
  Texto rodapé: "Honest: estimation half routes to leaves..."

Título geral: "Numerical Walkthrough — Honest Causal Tree (depth=1, n=8)".
Salvar: 'figures/fig09.png', dpi=150.
"""
```

---

## fig10 — Distribuições de CATE (regressão + binário)

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig10.png`

```python
"""
Dois painéis lado a lado (figsize=(13, 5)):

Painel esquerdo — "CATE Distribution — Regression":
- tau_hat_r: 250 pontos N(1.85, 0.04) + 250 pontos N(0.52, 0.03) [bimodal].
- hist: 35 bins, cor='mediumorchid', edgecolor='white', alpha=0.85.
- Linha tracejada roxa: Mean tau = {mean:.3f}.
- Linha sólida tomato: True ATE = 1.185.
- Legenda.

Painel direito — "Risk Difference Distribution — Binary":
- tau_hat_b: 250 pontos N(0.28, 0.008) + 250 pontos N(0.06, 0.005) [bimodal].
- hist: 35 bins, cor='#E74C3C', edgecolor='white', alpha=0.85.
- Linha tracejada #8E44AD: Mean tau.
- Linha sólida navy: True ATE = 0.175.
- Legenda.

Título geral: "Estimated CATE Distributions".
Salvar: 'figures/fig10.png', dpi=150.
"""
```

---

## fig11 — Curva de seleção max_depth

**Tipo:** Código Python / matplotlib
**Arquivo:** `figures/fig11.png`

```python
"""
Gráfico de linhas (figsize=(8, 4)):

depths = [1, 2, 3, 4, 5, 6, 7]
train_tau_risk = [4.2, 2.8, 1.9, 1.5, 1.52, 1.60, 1.75]
val_tau_risk   = [4.5, 3.1, 2.2, 1.8, 2.0,  2.3,  2.7 ]

- Linha roxa (#7B2D8B): train_tau_risk, marker='o', ms=6, lw=2.
- Linha laranja (#E84855) tracejada: val_tau_risk, marker='s', ms=6, lw=2.
- Linha vertical vermelha pontilhada: best_depth = argmin(val_tau_risk) = 4.
- Xlabel: "max_depth". Ylabel: "tau_risk".
- Título: "Hyperparameter Sweep — max_depth  (Regression)".
- Legenda: "Train tau_risk", "Val tau_risk", "Best depth = 4".
- Grid alpha=0.3.
Salvar: 'figures/fig11.png', dpi=150.
"""
```

---

## Como executar todos os prompts de código

Crie o arquivo `generate_figures.py` no diretório `36_CT/`:

```python
# generate_figures.py
"""Script que gera todas as figuras do README de Causal Trees."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
np.random.seed(42)

# Cole aqui o conteúdo de cada bloco de código acima,
# na ordem fig01 a fig11.
# Cada bloco chama plt.savefig('figures/figXX.png', ...) e plt.close().
```

Execute:
```bash
cd 36_CT
python generate_figures.py
```

Confirme que os 12 arquivos foram criados:
```bash
ls -lh figures/
```

Para `fig00` (banner), use o gerador de imagem com o prompt de arte conceitual
fornecido na seção correspondente acima.
