"""
Genera las figuras del informe de los 12 experimentos (OE2, protocolo B) y de la evidencia que sustenta sus conclusiones.
Todo sale de los CSV que ya estan en el repositorio; no entrena nada ni usa la API.

Uso (desde la raiz del repositorio):  python oe2_aumento_de_datos/informe_12_experimentos/generar_figuras.py
Salida: oe2_aumento_de_datos/informe_12_experimentos/figuras/*.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
OE2 = RAIZ / "oe2_aumento_de_datos"
CV = OE2 / "evaluacion/resultados/cv_interna"
DIAG = OE2 / "evaluacion/diagnostico_baseline/resultados"
REG = OE2 / "evaluacion/resultados/curva_regimen"
SAL = Path(__file__).resolve().parent / "figuras"
SAL.mkdir(exist_ok=True)

MOD = ["mbert", "labse", "xlmr"]; NOM = {"mbert": "mBERT", "labse": "LaBSE", "xlmr": "XLM-R"}
TEC = ["sin_aumento", "mixup", "retrotraduccion", "generate_then_refine"]
TNOM = {"sin_aumento": "Sin aumento", "mixup": "Mixup", "retrotraduccion": "Retrotraducción", "generate_then_refine": "Generate-then-Refine"}
COL = {"sin_aumento": "#212121", "mixup": "#1976d2", "retrotraduccion": "#f57c00", "generate_then_refine": "#2e7d32"}
CLASES = ["AFI", "DES", "EMO", "NEG", "PRG", "REQUEST", "SAL"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(SAL / nombre, bbox_inches="tight")
    plt.close(fig)
    print("figura:", nombre)


R = pd.read_csv(CV / "validacion_cruzada_resumen_sin_puntuacion_cv_interna_c120.csv")
D = pd.read_csv(CV / "comparacion_pareada_sin_puntuacion_cv_interna_c120.csv")
P = pd.read_csv(CV / "validacion_cruzada_predicciones_sin_puntuacion_cv_interna_c120.csv")

# ---------------------------------------------------------------- 1. los 12 experimentos
fig, ax = plt.subplots(figsize=(9, 4.6))
for j, t in enumerate(TEC):
    x = np.arange(3) + (j - 1.5) * 0.18
    y = [R[(R.modelo == m) & (R.tecnica == t)].f1_macro_agrupado.iloc[0] for m in MOD]
    lo = [y[i] - R[(R.modelo == m) & (R.tecnica == t)].ic95_bajo.iloc[0] for i, m in enumerate(MOD)]
    hi = [R[(R.modelo == m) & (R.tecnica == t)].ic95_alto.iloc[0] - y[i] for i, m in enumerate(MOD)]
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=COL[t], capsize=3, label=TNOM[t], markersize=6)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.043, f"{yi:.3f}", ha="center", fontsize=7.5, color=COL[t])
ax.set_xticks(range(3)); ax.set_xticklabels([NOM[m] for m in MOD]); ax.set_ylabel("F1 macro agrupado (IC 95%)")
ax.set_title("Los 12 experimentos (protocolo B, 7 intenciones, texto normalizado)"); ax.set_ylim(0.48, 0.77); ax.legend(ncol=4, fontsize=8, loc="upper right")
guardar(fig, "fig1_f1_12_experimentos.png")

# ---------------------------------------------------------------- 2. diferencias pareadas
fig, ax = plt.subplots(figsize=(8.5, 4.6))
filas = []
for m in MOD:
    for t in TEC[1:]:
        r = D[D.comparacion == f"{m}: {t} - sin_aumento"].iloc[0]
        filas.append((f"{NOM[m]} · {TNOM[t]}", r.diferencia_f1, r.ic95_bajo, r.ic95_alto, r.distinguible_de_cero, t))
for i, (n, d, lo, hi, sig, t) in enumerate(filas[::-1]):
    ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o" if sig else "s", color=COL[t], mfc=COL[t] if sig else "white", capsize=3, markersize=7)
ax.axvline(0, color="k", lw=1)
ax.set_yticks(range(len(filas))); ax.set_yticklabels([f[0] for f in filas[::-1]]); ax.set_xlabel("Diferencia de F1 frente a «sin aumento» (IC 95% pareado)")
ax.set_title("¿Alguna técnica supera al baseline? (relleno = distinguible de cero)")
guardar(fig, "fig2_diferencias_pareadas.png")

# ---------------------------------------------------------------- 3. por que gana mBERT: referencia de caracteres y pooling
ng = pd.read_csv(DIAG / "ngramas_caracteres.csv").iloc[0]
fp = pd.read_csv(RAIZ / "oe3_caracterizacion_embeddings/resultados/f1_por_pooling.csv")
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
ax = ejes[0]
vals = [("Clasificador de\nn-gramas de letras", ng.f1_macro_agrupado, "#6a1b9a")] + [(NOM[m], R[(R.modelo == m) & (R.tecnica == "sin_aumento")].f1_macro_agrupado.iloc[0], "#212121") for m in MOD]
ax.bar([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals])
for i, v in enumerate(vals):
    ax.text(i, v[1] + 0.008, f"{v[1]:.3f}", ha="center")
ax.set_ylim(0.4, 0.82); ax.set_ylabel("F1 macro"); ax.set_title("Una referencia que solo ve las letras\nsupera a los tres embeddings")
ax = ejes[1]
est = ["cls", "mean_pooling", "max_pooling", "combinacion_capas"]; enom = ["CLS", "mean", "max", "combinación\nde capas"]
for j, m in enumerate(MOD):
    y = [fp[(fp.modelo == m) & (fp.estrategia == e) & (fp.corpus == "sin aumento")].f1_macro.iloc[0] for e in est]
    ax.bar(np.arange(4) + (j - 1) * 0.26, y, 0.26, label=NOM[m])
ax.set_xticks(range(4)); ax.set_xticklabels(enom); ax.set_ylim(0.45, 0.72); ax.set_ylabel("F1 macro (sin aumento)"); ax.legend()
ax.set_title("El pooling mueve el F1; mBERT gana con las cuatro estrategias")
guardar(fig, "fig3_por_que_gana_mbert.png")

# ---------------------------------------------------------------- 4. curva de aprendizaje con datos reales
A = pd.read_csv(DIAG / "A_curva_aprendizaje.csv")
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for m in MOD:
    a = A[A.modelo == m]
    ax.plot(a.fraccion_de_reales * 100, a.f1_macro, marker="o", label=NOM[m])
    ax.annotate(f"+{a.f1_macro.iloc[-1] - a.f1_macro.iloc[-2]:.3f}", (100, a.f1_macro.iloc[-1]), textcoords="offset points", xytext=(6, -3), fontsize=8)
ax.set_xlabel("% de las oraciones reales de entrenamiento (~560 = 100%)"); ax.set_ylabel("F1 macro del baseline"); ax.set_xlim(20, 112)
ax.set_title("mBERT todavía mejora con más datos reales;\nLaBSE y XLM-R se aplanan"); ax.legend()
guardar(fig, "fig4_curva_aprendizaje_real.png")

# ---------------------------------------------------------------- 5. curva por regimen
RR = pd.read_csv(REG / "resumen_por_regimen.csv"); RD = pd.read_csv(REG / "comparacion_pareada.csv")
fig, ejes = plt.subplots(2, 3, figsize=(13, 7))
POS = {10: 0, 25: 1, 50: 2, 80: 3}
for j, m in enumerate(MOD):
    ax = ejes[0, j]
    for t in TEC:
        g = RR[(RR.modelo == m) & (RR.tecnica == t)].sort_values("regimen")
        ax.plot([POS[r] for r in g.regimen], g.f1_macro, marker="o", color=COL[t], label=TNOM[t], lw=2 if t in ("sin_aumento", "generate_then_refine") else 1.2)
    ax.set_title(NOM[m]); ax.set_xticks(range(4)); ax.set_xticklabels(["10", "25", "50", "80"]); ax.set_xlabel("ejemplos reales por clase")
    if j == 0:
        ax.set_ylabel("F1 macro"); ax.legend(fontsize=8)
    ax = ejes[1, j]
    for k, t in enumerate(TEC[1:]):
        g = RD[(RD.modelo == m) & (RD.tecnica == t)].sort_values("regimen")
        xs = np.arange(4) + (k - 1) * 0.25
        ax.bar(xs, g.dif_vs_sin_aumento, 0.25, color=COL[t], yerr=[g.dif_vs_sin_aumento - g.ic95_bajo, g.ic95_alto - g.dif_vs_sin_aumento], capsize=2, label=TNOM[t])
        for xi, d, s in zip(xs, g.dif_vs_sin_aumento, g.distinguible_de_cero):
            if s:
                ax.text(xi, d + (0.012 if d > 0 else -0.018), "*", ha="center", fontsize=13, color=COL[t])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(["10", "25", "50", "80"]); ax.set_xlabel("ejemplos reales por clase")
    if j == 0:
        ax.set_ylabel("Δ F1 frente a sin aumento")
fig.suptitle("El aumento con GtR ayuda con pocos datos reales y deja de ayudar con más (* = IC 95% no incluye 0)", y=1.0)
guardar(fig, "fig5_curva_por_regimen.png")

# ---------------------------------------------------------------- 6. utilidad de lo sintetico
B = pd.read_csv(DIAG / "B_utilidad_sintetico.csv")
acc = {m: float((P[(P.modelo == m) & (P.tecnica == "sin_aumento")].real == P[(P.modelo == m) & (P.tecnica == "sin_aumento")].prediccion).mean()) for m in MOD}
fig, ejes = plt.subplots(1, 3, figsize=(13, 4))
tt = ["mixup", "retrotraduccion", "generate_then_refine"]
for ax, col, tit in zip(ejes, ["B1_f1_solo_sinteticos", "B2_concordancia_etiqueta", "B4_auc_real_vs_sintetico"],
                        ["(a) F1 entrenando SOLO con lo sintético", "(b) Acierto de su etiqueta por un clasificador real", "(c) Separabilidad real vs sintético (AUC)"]):
    for j, t in enumerate(tt):
        y = [B[(B.tecnica == t) & (B.modelo == m)][col].iloc[0] for m in MOD]
        ax.bar(np.arange(3) + (j - 1) * 0.26, y, 0.26, color=COL[t], label=TNOM[t])
    ax.set_xticks(range(3)); ax.set_xticklabels([NOM[m] for m in MOD]); ax.set_title(tit, fontsize=10)
    if col == "B1_f1_solo_sinteticos":
        for i, m in enumerate(MOD):
            ax.hlines(R[(R.modelo == m) & (R.tecnica == "sin_aumento")].f1_macro_agrupado.iloc[0], i - 0.4, i + 0.4, color="k", ls="--", lw=1)
        ax.text(2.45, 0.67, "línea: F1 con las reales", fontsize=7, ha="right")
    if col == "B2_concordancia_etiqueta":
        for i, m in enumerate(MOD):
            ax.hlines(acc[m], i - 0.4, i + 0.4, color="k", ls="--", lw=1)
        ax.text(2.45, 0.02, "línea: acierto en reales nuevas", fontsize=7, ha="right")
    if col == "B4_auc_real_vs_sintetico":
        ax.axhline(0.5, color="k", ls=":", lw=1)
ejes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Lo sintético sirve menos que lo real (diagnóstico con C fijo)", y=1.02)
guardar(fig, "fig6_utilidad_sintetico.png")

# ---------------------------------------------------------------- 7. composicion del sintetico y dosis
import glob  # noqa: E402
CACHE = OE2 / "tecnicas_aumento/salidas/en_linea_marcadores_fuentes/generate_then_refine"
import sys  # noqa: E402
sys.path.insert(0, str(RAIZ))
from shiwilu.clasificacion import _normalizar_estricto  # noqa: E402
cuentas = []
for f in range(5):
    g = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(CACHE / f"fold{f}_pool*.csv")))])
    ap = g[g.estado_filtro == "aprobado"].copy(); ap["k"] = ap.shiwilu.map(_normalizar_estricto)
    cuentas.append(ap.drop_duplicates(["intencion", "k"]).intencion.value_counts())
comp = pd.concat(cuentas, axis=1).mean(axis=1).reindex(CLASES)
Dd = pd.read_csv(DIAG / "C_dosis.csv"); PCat = pd.read_csv(DIAG / "C_por_categoria.csv")
fig, ejes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1.2]})
ax = ejes[0]
ax.bar(np.arange(7) - 0.2, [80] * 7, 0.4, color="#9e9e9e", label="reales por clase (~80)")
ax.bar(np.arange(7) + 0.2, comp.values, 0.4, color=COL["generate_then_refine"], label="sintéticas aprobadas de GtR 120")
ax.set_xticks(range(7)); ax.set_xticklabels(CLASES, rotation=30); ax.legend(fontsize=8); ax.set_title("(a) El sintético de GtR queda desbalanceado")
ax = ejes[1]
orden = ["0", "25", "50", "75", "100"]
for m in MOD:
    d = Dd[Dd.modelo == m].copy(); d["porcentaje_de_lo_sintetico"] = d["porcentaje_de_lo_sintetico"].astype(str)
    y = [d[d.porcentaje_de_lo_sintetico == o].f1_macro.iloc[0] for o in orden]
    ax.plot([0, 25, 50, 75, 100], y, marker="o", label=NOM[m])
    ax.scatter([100], [d[d.porcentaje_de_lo_sintetico == "balanceado"].f1_macro.iloc[0]], marker="*", s=130, color=ax.lines[-1].get_color(), zorder=5)
ax.set_xlabel("% de lo sintético agregado (★ = balanceado)"); ax.set_ylabel("F1 macro (C fijo)"); ax.legend(fontsize=8); ax.set_title("(b) Dosis: más sintético no mejora a mBERT")
ax = ejes[2]
orden_c = PCat.assign(d=PCat.f1_macro_solo_sintetico_de_esta - PCat.f1_macro_baseline).sort_values("d")
ax.barh(orden_c.categoria, orden_c.d, color=["#c62828" if v < -0.005 else "#9e9e9e" for v in orden_c.d])
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Δ F1 macro de mBERT al agregar solo el sintético de esa clase", fontsize=8.5); ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.set_title("(c) El sintético de DES, PRG y SAL es el que más perjudica")
guardar(fig, "fig7_composicion_y_dosis.png")

# ---------------------------------------------------------------- 8. F1 por clase
def f1c(m, t):
    from sklearn.metrics import f1_score
    g = P[(P.modelo == m) & (P.tecnica == t)]
    return f1_score(g.real, g.prediccion, average=None, labels=CLASES, zero_division=0)
base = pd.DataFrame({NOM[m]: f1c(m, "sin_aumento") for m in MOD}, index=CLASES)
fig, ejes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
im = ejes[0].imshow(base.values, cmap="viridis", vmin=0.3, vmax=0.9, aspect="auto")
ejes[0].set_xticks(range(3)); ejes[0].set_xticklabels(base.columns); ejes[0].set_yticks(range(7)); ejes[0].set_yticklabels(CLASES)
for i in range(7):
    for j in range(3):
        ejes[0].text(j, i, f"{base.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
ejes[0].set_title("F1 por clase, sin aumento"); ejes[0].grid(False)
for ax, t in zip(ejes[1:], TEC[1:]):
    dd = pd.DataFrame({NOM[m]: f1c(m, t) - f1c(m, "sin_aumento") for m in MOD}, index=CLASES)
    ax.imshow(dd.values, cmap="RdYlGn", vmin=-0.12, vmax=0.12, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels(dd.columns); ax.set_yticks(range(7)); ax.set_yticklabels(CLASES)
    for i in range(7):
        for j in range(3):
            ax.text(j, i, f"{dd.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Δ F1 por clase: {TNOM[t]}")
guardar(fig, "fig8_f1_por_clase.png")

# ---------------------------------------------------------------- 9. por que el F1 se estanca con mas sintetico
EC = pd.read_csv(DIAG / "E_curva_por_lotes.csv"); EE = pd.read_csv(DIAG / "E_estadisticas_por_lote.csv")
fig, ejes = plt.subplots(2, 3, figsize=(14, 7.5))
VER = {"bruto": ("#c62828", "-", "todo lo aprobado"), "sin_ruido_etiq": ("#2e7d32", "-", "sin ruido de etiqueta"),
       "sin_casi_dup": ("#1976d2", "--", "sin casi-duplicados"), "ambos": ("#6a1b9a", ":", "sin ruido ni duplicados")}
for ax, m in zip(ejes[0], MOD):
    base = EC[(EC.modelo == m) & (EC.version == "sin_aumento")].f1_macro.iloc[0]
    ax.axhline(base, color="k", lw=1.2, label="sin aumento")
    for v, (c, ls, lab) in VER.items():
        g = EC[(EC.modelo == m) & (EC.version == v)].sort_values("lotes")
        ax.plot(g.lotes * 20, g.f1_macro, color=c, ls=ls, marker="o", ms=4, label=lab)
    ax.set_title(NOM[m]); ax.set_xlabel("sintético por categoría (lotes de 20)"); ax.set_xticks([20, 40, 60, 80, 100, 120])
    if m == "mbert":
        ax.set_ylabel("F1 macro (C fijo)"); ax.legend(fontsize=7.5)
ax = ejes[1, 0]
e = EE.groupby("lote").mean(numeric_only=True)
ax.bar(e.index * 20 + 20, e.n_nuevas_unicas, 14, color="#9e9e9e")
ax.set_ylabel("oraciones nuevas únicas por lote", color="#555"); ax.set_xlabel("lote (sintético acumulado por categoría)"); ax.set_xticks([20, 40, 60, 80, 100, 120])
ax2 = ax.twinx(); ax2.plot(e.index * 20 + 20, e.novedad_media, color="#f57c00", marker="o"); ax2.set_ylabel("novedad (1 − similitud con lotes previos)", color="#f57c00"); ax2.set_ylim(0, 0.3); ax2.grid(False)
ax.set_title("Cada lote aporta oraciones nuevas, pero menos novedosas", fontsize=9.5)
ax = ejes[1, 1]
for m in MOD:
    ax.plot(e.index * 20 + 20, e[f"acuerdo_{m}"], marker="o", label=f"acuerdo de etiqueta ({NOM[m]})")
ax.plot(e.index * 20 + 20, e.pct_casi_duplicadas, color="k", ls="--", marker="s", label="% casi-duplicadas")
ax.set_ylim(0, 1); ax.set_xlabel("lote (sintético acumulado por categoría)"); ax.set_xticks([20, 40, 60, 80, 100, 120]); ax.legend(fontsize=7.5)
ax.set_title("El ruido de etiqueta es constante por lote: se acumula", fontsize=9.5)
ax = ejes[1, 2]
ax.bar(e.index * 20 + 20, e.ejemplos_reales_vistos_por_clase, 14, color=COL["generate_then_refine"], label="ejemplos reales distintos que ve Claude")
ax.axhline(80, color="#9e9e9e", ls="--"); ax.text(24, 82, "~80 reales disponibles por clase", fontsize=8, color="#555")
ax.set_ylim(0, 95); ax.set_xlabel("lote (sintético acumulado por categoría)"); ax.set_xticks([20, 40, 60, 80, 100, 120]); ax.legend(fontsize=8, loc="center right")
ax.set_title("Con 6 lotes Claude ve 36 de los ~80 ejemplos reales", fontsize=9.5)
fig.suptitle("Por qué el F1 mejora con poco sintético y luego se estanca", y=1.0)
guardar(fig, "fig9_por_que_se_estanca.png")
# ---------------------------------------------------------------- 10. variantes de seleccion del sintetico de GtR (mBERT, protocolo B)
VR = OE2 / "evaluacion/resultados/variantes_gtr"
if (VR / "resumen_variantes.csv").exists():
    RV = pd.read_csv(VR / "resumen_variantes.csv"); RV = RV[RV.variante != "ref_sin_aumento"].reset_index(drop=True)
    PCv = pd.read_csv(VR / "f1_por_clase.csv").set_index("variante")[CLASES]
    dpc = PCv.sub(PCv.loc["ref_sin_aumento"]).drop(index="ref_sin_aumento")
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = ejes[0]
    for i, r in RV[::-1].reset_index(drop=True).iterrows():
        sig = bool(r.distinguible_de_cero)
        ref = r.variante.startswith("ref_")
        ax.errorbar(r.dif_vs_sin_aumento, i, xerr=[[r.dif_vs_sin_aumento - r.ic95_bajo], [r.ic95_alto - r.dif_vs_sin_aumento]], fmt="o" if sig else "s",
                    color="#9e9e9e" if ref else "#c62828", mfc=("#9e9e9e" if ref else "#c62828") if sig else "white", capsize=3, markersize=7)
    ax.axvline(0, color="k", lw=1); ax.set_yticks(range(len(RV))); ax.set_yticklabels(RV.variante[::-1], fontsize=8)
    ax.set_xlabel("Diferencia de F1 frente a sin aumento (mBERT, IC 95% pareado)"); ax.set_title("Ninguna variante de selección supera al baseline")
    ax = ejes[1]
    ax.imshow(dpc.values, cmap="RdYlGn", vmin=-0.07, vmax=0.07, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(7)); ax.set_xticklabels(CLASES); ax.set_yticks(range(len(dpc))); ax.set_yticklabels(dpc.index, fontsize=8)
    for i in range(len(dpc)):
        for j in range(7):
            ax.text(j, i, f"{dpc.values[i, j]:+.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_title("Cambio del F1 por clase: DES baja en todas, incluso sin su propio sintético")
    guardar(fig, "fig10_variantes_seleccion_gtr.png")
print("listo")
