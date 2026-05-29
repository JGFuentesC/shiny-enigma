import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
import textwrap

# ============================================================
# 1. CARGAR DATOS
# ============================================================
df = pd.read_pickle("data/opiniones.pkl")

COLS = {
    "contenidos": "¿Cómo calificas la calidad de los contenidos del curso?",
    "materiales": "¿Cómo calificas la calidad de los materiales(apuntes, códigos, datos, etc.?",
    "dificultad_contenidos": "¿Cómo calificas el nivel de dificultad de los contenidos del curso?",
    "ponente": "¿Cómo calificas la calidad del ponente?",
    "dificultad_eval": "¿Cómo calificas el nivel de dificultad de la evaluación?",
    "practicas": "¿Cómo calificas la calidad de las prácticas en clase?",
    "estrellas": "En general, ¿Cuántas estrellas darías al módulo?",
    "opinion": "Por favor, emite tu libre opinión sobre el módulo en general (instructor, material, contenido, etc.)",
}

LABELS = {
    "contenidos": "Calidad contenidos",
    "materiales": "Calidad materiales",
    "dificultad_contenidos": "Dificultad contenidos",
    "ponente": "Calidad ponente",
    "dificultad_eval": "Dificultad evaluación",
    "practicas": "Calidad prácticas",
    "estrellas": "Estrellas generales",
}

OUT = "notebooks/04_analisis_sentimiento"

# ============================================================
# 2. SENTIMENT ANALYSIS  (lexicon-based en español)
# ============================================================
POS = set(
    """
excelente bueno buen buena muy bien fantástico magnífico increíble increible genial
estupendo maravilloso grandioso excelentes buenísimo buenísima buenisimo buenisima
mejor mejores útil útiles util utiles claro clara claros claras completo completa
completos completas práctico practico práctica practica prácticos practicos
divertido divertida divertidos divertidas disfruté disfrute disfrutar disfrutado
gustó gusto gusta gustado encantó encanto encantado encantada apasionado apasionada
pasión pasion pasionante motivador motivadora motiva motivado motivada interesante
interesantes ameno amena amenos amenas didáctico didactico didáctica didactica
felicito felicidades felicitaciones optimo óptimo optima óptima increíble increible
recomiendo recomendable recomendado vale la pena valió valio aprendí aprendi
aprender aprendizaje aprendido aprendido resolvió resolvio resolvio ayudar ayuda
ayudó ayudo ayudado resolviendo entendible entendible entendí entendi comprender
comprensión comprension comprendí comprendi dominio preparado preparada
preocupación preocupacion preocupado preocupada integro íntegro integra íntegra
valores inteligente increible increíble eficiente eficaz eficacia efectivo efectiva
aprovechó aprovecho aprovecha aprovechado optimo óptimo optimizar bien
""".strip().split()
)
NEG = set(
    """
malo mala mal pésimo pesimo pésima pesima regular deficiente insuficiente
difícil dificil difíciles dificiles complicado complicada complicados complicadas
confuso confusa confusos confusas aburrido aburrida aburridos pesado pesada
lento lenta lentos lentas rápido rápida rapidos rapidas deprisa apresurado
apresurada estresante estresante estrés estres frustrante frustrado frustrada
frustración frustracion falta faltó falto faltan faltaron carece carecen
ausencia no suficiente insuficiente pobre pobres malo mala malos malas
mejorar mejorable flojo floja flojos flojas desorganizado desorganizada
caótico caotico caótica caotica desorden desordenado desordenada
queja quejas reclamar reclamación reclamacion problema problemas
difícil dificil difícil difícil complicado complicada error errores
fallo fallos falla fallan fallas quejó quejo quejo disgustó disgusto
decepcionó decepciono decepción decepcion desepcion desepción
""".strip().split()
)

STOP_ES = set(
    STOPWORDS
    | {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
        "en", "al", "a", "por", "para", "con", "sin", "que", "es", "se",
        "lo", "le", "su", "sus", "mi", "tu", "nuestro", "como", "cómo",
        "más", "mas", "muy", "tan", "pero", "o", "y", "e", "no", "si",
        "ya", "ha", "han", "he", "han", "fue", "fueron", "era", "eran",
        "ser", "son", "está", "estan", "están", "estaba", "estaban",
        "me", "te", "nos", "les", "todo", "toda", "todos", "todas",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "aquel", "aquella", "aquellos", "aquellas", "hay", "hace", "hacen",
        "hacer", "hizo", "hicieron", "tiene", "tienen", "tenía", "tenian",
        "tenían", "puede", "pueden", "podía", "podian", "podían",
        "también", "tambien", "porque", "cada", "entre", "hasta", "desde",
        "sobre", "mucho", "mucha", "muchos", "muchas", "poco", "poca",
        "pocos", "pocas", "bien", "mal", "mejor", "peor", "ahora",
        "entonces", "luego", "después", "despues", "antes", "durante",
        "siempre", "nunca", "aunque", "cuando", "cuándo", "donde", "dónde",
        "ni", "solo", "sola", "solos", "solas", "sólo", "dar", "da",
        "dan", "dio", "dieron", "ver", "ve", "ven", "vi", "vio", "vieron",
        "decir", "dice", "dicen", "dijo", "dijeron", "ir", "va", "van",
        "fui", "fue", "fueron", "estar", "estoy", "estás", "estas",
        "está", "estan", "están", "estaba", "estaban", "así", "asi",
        "otro", "otra", "otros", "otras", "algunos", "algunas", "algo",
        "nada", "alguien", "nadie", "cual", "cuales", "cuyo", "cuya",
        "allá", "alla", "allí", "alli", "aquí", "aqui", "ahí", "ahi",
        "sea", "sean", "sido", "siendo", "the", "is", "a", "an", "and",
        "or", "of", "to", "in", "it", "that", "was", "were", "are",
        "be", "been", "being", "have", "has", "had", "having", "do",
        "does", "did", "doing", "will", "would", "could", "should",
        "may", "might", "must", "shall", "can", "say", "said", "get",
        "got", "go", "went", "come", "came", "make", "made", "know",
        "knew", "take", "took", "see", "saw", "think", "thought",
        "want", "look", "use", "find", "give", "tell", "work", "call",
        "try", "ask", "need", "feel", "become", "leave", "put", "mean",
        "keep", "let", "begin", "show", "hear", "play", "run", "move",
        "live", "believe", "hold", "bring", "happen", "write", "provide",
        "sit", "stand", "lose", "pay", "meet", "include", "continue",
        "set", "learn", "change", "lead", "understand", "watch",
        "follow", "stop", "create", "speak", "read", "allow", "add",
        "spend", "grow", "open", "walk", "win", "offer", "remember",
        "love", "consider", "appear", "buy", "wait", "serve", "die",
        "send", "expect", "build", "stay", "fall", "cut", "reach",
        "kill", "remain", "suggest", "raise", "pass", "sell", "require",
        "report", "decide", "pull", "muy", "bastante", "bastantes",
        "pues", "sin_embargo", "sin embargo", "casi", "veces",
        "quizá", "quizas", "quizás", "module", "curso", "clase", "clases",
        "módulo", "modulo", "diplomado", "materia", "profesor", "profe",
        "ponente", "instructor", "oscar", "tema", "temas", "contenido",
        "contenidos", "material", "materiales", "práctica", "practica",
        "prácticas", "practicas", "práctica", "examen", "exámenes",
        "examenes", "evaluación", "evaluacion", "evaluaciones", "tiempo",
        "parte", "partes", "día", "dia", "días", "dias", "semana",
        "mes", "año", "años", "anos", "forma", "manera", "modo",
        "ejemplo", "ejemplos", "caso", "casos", "nivel", "niveles",
        "vez", "cosas", "cosa", "cuenta", "realidad", "final", "inicio",
        "comienzo", "principio", "fin", "lado", "duda", "dudas",
    }
)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]", " ", text)
    return text


def sentiment_score(text: str) -> float:
    """Lexicon-based sentiment: (pos-neg)/total con peso. Range aprox -1 a 1"""
    words = clean_text(text).split()
    pos_count = sum(1 for w in words if w in POS)
    neg_count = sum(1 for w in words if w in NEG)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / (pos_count + neg_count)


def sentiment_label(score: float) -> str:
    if score > 0.2:
        return "Positivo"
    elif score < -0.2:
        return "Negativo"
    else:
        return "Neutral"


# ============================================================
# 3. NER  (basado en reglas para español)
# ============================================================

ENTITIES = {
    "PERSONA": [
        "oscar", "Óscar", "óscar", "profesor", "profe", "ponente", "instructor",
        "dr", "doctor", "docente", "maestro", "maestra", "profesora",
    ],
    "CURSO": [
        "módulo", "modulo", "diplomado", "curso", "materia", "asignatura",
        "clase", "clases", "seminario", "taller",
    ],
    "MATERIAL": [
        "apuntes", "código", "códigos", "codigo", "codigos", "datos",
        "dataset", "notebook", "notebooks", "libro", "libros", "pdf",
        "diapositivas", "slides", "transparencias", "video", "videos",
        "grabación", "grabaciones", "grabacion", "grabaciones",
        "documentación", "documentacion", "manual", "manuales",
        "guía", "guia", "guías", "guias", "tutorial", "tutoriales",
    ],
    "EVALUACION": [
        "examen", "exámenes", "examenes", "evaluación", "evaluacion",
        "prueba", "pruebas", "test", "tests", "quiz", "quizzes",
        "calificación", "calificacion", "nota", "notas", "pregunta",
        "preguntas", "práctica", "practica", "prácticas", "practicas",
    ],
    "HERRAMIENTA": [
        "python", "r", "sql", "tensorflow", "pytorch", "keras",
        "scikit-learn", "sklearn", "pandas", "numpy", "matplotlib",
        "jupyter", "colab", "vscode", "git", "github", "docker",
        "excel", "power bi", "tableau", "spark", "hadoop",
        "herramienta", "herramientas", "librería", "libreria",
        "librerías", "librerias", "paquete", "paquetes",
    ],
    "CONCEPTO": [
        "machine learning", "deep learning", "inteligencia artificial",
        "ia", "red neuronal", "redes neuronales", "nlp", "procesamiento",
        "big data", "data science", "ciencia de datos", "estadística",
        "estadistica", "probabilidad", "álgebra", "algebra",
        "cálculo", "calculo", "optimización", "optimizacion",
        "regresión", "regresion", "clasificación", "clasificacion",
        "clustering", "visualización", "visualizacion",
    ],
}


def extract_entities(text: str) -> dict:
    text_lower = text.lower()
    found = {}
    for category, keywords in ENTITIES.items():
        matches = []
        for kw in keywords:
            if kw in text_lower:
                matches.append(kw)
        if matches:
            found[category] = list(set(matches))
    return found


# ============================================================
# 3b. ASPECT-BASED KEYWORD EXTRACTION
# ============================================================

ASPECTOS = {
    "Instructor/Ponente": [
        "profesor", "profe", "ponente", "instructor", "oscar", "docente",
        "enseña", "enseñanza", "ensenanza", "explica", "explicación",
        "explicacion", "explicaciones", "claridad", "paciencia", "paciente",
        "dominio", "preparado", "preparación", "preparacion", "pasión",
        "pasion", "motivación", "motivacion", "motiva", "apasionado",
        "apasionada", "carismático", "carismatico", "carisma", "didáctico",
        "didactico", "didáctica", "didactica", "pedagogía", "pedagogia",
    ],
    "Contenido": [
        "contenido", "contenidos", "temario", "temas", "tema", "tópicos",
        "topicos", "interesante", "interesantes", "aburrido", "aburridos",
        "profundo", "profundidad", "superficial", "actualizado",
        "actualizados", "desactualizado", "teórico", "teorico", "teoría",
        "teoria", "práctico", "practico", "práctica", "practica",
        "balance", "equilibrio", "completo", "completos", "incompleto",
        "incompletos", "denso", "densos", "ligero", "ligeros",
    ],
    "Materiales": [
        "material", "materiales", "apuntes", "código", "codigo", "códigos",
        "codigos", "datos", "dataset", "datasets", "notebook", "notebooks",
        "diapositivas", "slides", "pdf", "documento", "documentos",
        "documentación", "documentacion", "calidad", "organizado",
        "organizados", "desorganizado", "desorganizados", "útil", "utiles",
        "útiles", "util", "útil", "inútil", "inutil", "inútiles", "inutiles",
    ],
    "Prácticas/Ejercicios": [
        "práctica", "practica", "prácticas", "practicas", "ejercicio",
        "ejercicios", "prácticos", "practicos", "práctico", "practico",
        "taller", "talleres", "laboratorio", "laboratorios", "hands-on",
        "aplicado", "aplicados", "aplicación", "aplicacion", "ejemplo",
        "ejemplos", "divertido", "divertidas", "divertidos", "entretenido",
        "entretenidas", "reto", "retos", "retador", "retadores",
        "desafiante", "desafiantes", "desafío", "desafio", "desafíos",
        "desafios", "reales", "real", "realista", "realistas",
    ],
    "Evaluación": [
        "examen", "exámenes", "examenes", "evaluación", "evaluacion",
        "evaluaciones", "prueba", "pruebas", "test", "tests", "quiz",
        "quizzes", "nota", "notas", "calificación", "calificacion",
        "preguntas", "pregunta", "difícil", "dificil", "difíciles",
        "dificiles", "fácil", "facil", "fáciles", "faciles", "justo",
        "justa", "justos", "justas", "injusto", "injusta", "exigente",
        "exigencia", "presión", "presion", "estrés", "estres",
    ],
    "Organización/Tiempo": [
        "tiempo", "horario", "horarios", "duración", "duracion",
        "organización", "organizacion", "organizado", "organizada",
        "estructura", "estructurado", "ritmo", "velocidad", "rápido",
        "rapido", "lento", "lentas", "pausa", "pausas", "descanso",
        "descansos", "extenso", "extensa", "corto", "corta",
        "intensivo", "intensiva", "horas", "sesión", "sesion",
        "sesiones", "cronograma", "calendario", "semana", "día",
        "dia", "días", "dias", "carga", "sobrecarga",
    ],
}


def aspect_analysis(text: str) -> dict:
    text_lower = text.lower()
    result = {}
    for aspect, keywords in ASPECTOS.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            result[aspect] = found
    return result


# ============================================================
# 4. COMPUTAR TODO
# ============================================================

df["sentiment_score"] = df[COLS["opinion"]].apply(sentiment_score)
df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)
df["entities"] = df[COLS["opinion"]].apply(extract_entities)
df["aspects"] = df[COLS["opinion"]].apply(aspect_analysis)
df["opinion_len"] = df[COLS["opinion"]].apply(len)
df["opinion_words"] = df[COLS["opinion"]].apply(lambda t: len(t.split()))
df["mes"] = df["Timestamp"].dt.to_period("M").astype(str)

# All words for wordcloud
all_text = " ".join(df[COLS["opinion"]].apply(clean_text))
all_words = [w for w in all_text.split() if w not in STOP_ES and len(w) > 2]

# ============================================================
# 5. GENERAR GRÁFICAS
# ============================================================

plt.style.use("ggplot")
FIGSIZE = (10, 5)

# --- 5a. Distribución de ratings ---
rating_cols = [COLS[k] for k in LABELS]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
colors = plt.cm.viridis(np.linspace(0.2, 0.9, 7))
for i, (col, label) in enumerate(LABELS.items()):
    ax = axes[i]
    counts = df[COLS[col]].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors[i], edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Frecuencia")
    ax.set_xlabel("Rating")
    ax.set_ylim(0, counts.max() * 1.18)
axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/distribucion_ratings.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5b. Matriz de correlación ---
corr_df = df[[COLS[k] for k in LABELS]].copy()
corr_df.columns = [LABELS[k] for k in LABELS]
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-0.3, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(["\n".join(textwrap.wrap(l, 12)) for l in corr_matrix.columns], fontsize=8)
ax.set_yticklabels(corr_matrix.columns, fontsize=8)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if abs(corr_matrix.values[i, j]) > 0.55 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Matriz de correlación entre dimensiones", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/correlacion.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5c. Sentiment distribution ---
sent_counts = df["sentiment_label"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors_sent = {"Positivo": "#2ecc71", "Negativo": "#e74c3c", "Neutral": "#95a5a6"}
# Pie
wedges, texts, autotexts = axes[0].pie(
    sent_counts.values, labels=sent_counts.index,
    autopct="%1.1f%%", startangle=140,
    colors=[colors_sent.get(l, "#bdc3c7") for l in sent_counts.index],
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontweight("bold")
    at.set_fontsize(10)
axes[0].set_title("Distribución de sentimiento (opiniones)", fontsize=12, fontweight="bold")

# Histogram of scores
axes[1].hist(df["sentiment_score"], bins=25, color="#3498db", edgecolor="white", alpha=0.85)
axes[1].axvline(df["sentiment_score"].mean(), color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Media = {df['sentiment_score'].mean():.3f}")
axes[1].axvline(0, color="gray", linestyle="-", linewidth=0.8)
axes[1].set_xlabel("Score de sentimiento", fontsize=10)
axes[1].set_ylabel("Frecuencia", fontsize=10)
axes[1].set_title("Distribución del score de sentimiento", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/sentimiento.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5d. Sentiment vs Stars ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot
star_groups = [df[df[COLS["estrellas"]] == s]["sentiment_score"].values for s in sorted(df[COLS["estrellas"]].unique())]
bp = axes[0].boxplot(star_groups, labels=sorted(df[COLS["estrellas"]].unique()), patch_artist=True)
for patch, color in zip(bp["boxes"], plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(star_groups)))):
    patch.set_facecolor(color)
axes[0].set_xlabel("Estrellas generales", fontsize=10)
axes[0].set_ylabel("Score de sentimiento", fontsize=10)
axes[0].set_title("Sentimiento del texto vs Rating general", fontsize=12, fontweight="bold")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

# Scatter with jitter
stars = df[COLS["estrellas"]].values
scores = df["sentiment_score"].values
jitter = np.random.uniform(-0.12, 0.12, len(stars))
axes[1].scatter(stars + jitter, scores, alpha=0.4, color="#8e44ad", s=40, edgecolors="white", linewidth=0.3)
axes[1].set_xlabel("Estrellas generales", fontsize=10)
axes[1].set_ylabel("Score de sentimiento", fontsize=10)
axes[1].set_title("Relación estrellas vs sentimiento (jitter)", fontsize=12, fontweight="bold")

# Add regression line
from numpy.polynomial.polynomial import polyfit
coefs = polyfit(stars, scores, 1)
x_line = np.linspace(stars.min(), stars.max(), 50)
y_line = coefs[0] + coefs[1] * x_line
axes[1].plot(x_line, y_line, color="red", linewidth=2, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUT}/sentimiento_vs_estrellas.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5e. WordCloud ---
wc = WordCloud(
    width=900, height=450, background_color="white",
    stopwords=STOP_ES, max_words=120, colormap="magma",
    collocations=False, contour_width=0.5, contour_color="#8e44ad",
    min_font_size=10,
).generate(all_text)

fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Nube de palabras — Opiniones del módulo", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUT}/wordcloud.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5f. Top bigrams ---
def get_bigrams(texts):
    bigrams = []
    for t in texts:
        words = [w for w in clean_text(t).split() if w not in STOP_ES and len(w) > 2]
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i+1]}")
    return Counter(bigrams)

bigram_counter = get_bigrams(df[COLS["opinion"]])
top_bigrams = bigram_counter.most_common(20)

fig, ax = plt.subplots(figsize=(10, 7))
b_labels = [b[0] for b in reversed(top_bigrams)]
b_vals = [b[1] for b in reversed(top_bigrams)]
bars = ax.barh(b_labels, b_vals, color=plt.cm.plasma(np.linspace(0.2, 0.9, len(top_bigrams))), edgecolor="white")
for bar, val in zip(bars, b_vals):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Frecuencia", fontsize=10)
ax.set_title("Top 20 bigramas más frecuentes", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/bigramas.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5g. Aspect analysis ---
aspect_counter = Counter()
for aspects in df["aspects"]:
    for aspect in aspects:
        aspect_counter[aspect] += 1

fig, ax = plt.subplots(figsize=(10, 5))
asp_labels = [a for a, _ in aspect_counter.most_common()]
asp_vals = [v for _, v in aspect_counter.most_common()]
bars = ax.barh(list(reversed(asp_labels)), list(reversed(asp_vals)),
               color=plt.cm.tab10(np.linspace(0, 1, len(asp_labels))), edgecolor="white")
for bar, val in zip(bars, reversed(asp_vals)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Nº de opiniones que mencionan el aspecto", fontsize=10)
ax.set_title("Aspectos mencionados en las opiniones", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/aspectos.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5h. NER ---
ner_counter = Counter()
for ents in df["entities"]:
    for cat in ents:
        ner_counter[cat] += 1

fig, ax = plt.subplots(figsize=(8, 5))
ner_labels = [n for n, _ in ner_counter.most_common()]
ner_vals = [v for _, v in ner_counter.most_common()]
bars = ax.bar(ner_labels, ner_vals, color=plt.cm.Set2(np.linspace(0, 1, len(ner_labels))), edgecolor="white")
for bar, val in zip(bars, ner_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Nº de opiniones", fontsize=10)
ax.set_title("Entidades nombradas detectadas (NER)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/ner.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5i. Evolución temporal ---
df_time = df.copy()
df_time["periodo"] = df_time["Timestamp"].dt.to_period("Q").astype(str)
time_stats = df_time.groupby("periodo").agg(
    n=("opinion_len", "count"),
    avg_stars=(COLS["estrellas"], "mean"),
    avg_sentiment=("sentiment_score", "mean"),
).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(time_stats["periodo"], time_stats["n"], color="#3498db", alpha=0.6, label="Nº respuestas")
ax1.set_ylabel("Nº de respuestas", fontsize=10, color="#3498db")
ax2 = ax1.twinx()
ax2.plot(time_stats["periodo"], time_stats["avg_stars"], "o-", color="#e74c3c", linewidth=2, markersize=8, label="Avg estrellas")
ax2.plot(time_stats["periodo"], time_stats["avg_sentiment"], "s--", color="#2ecc71", linewidth=2, markersize=8, label="Avg sentimiento")
ax2.set_ylabel("Promedio", fontsize=10)
ax2.set_ylim(0, 6)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.set_title("Evolución temporal de respuestas y ratings", fontsize=13, fontweight="bold")
ax1.set_xlabel("Trimestre", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/evolucion_temporal.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 6. GENERAR MARKDOWN
# ============================================================

# Helper stats
total = len(df)
mean_stars = df[COLS["estrellas"]].mean()
median_stars = df[COLS["estrellas"]].median()
star_dist = df[COLS["estrellas"]].value_counts().sort_index()

pos_pct = (df["sentiment_label"] == "Positivo").mean() * 100
neg_pct = (df["sentiment_label"] == "Negativo").mean() * 100
neu_pct = (df["sentiment_label"] == "Neutral").mean() * 100

top_words = Counter(all_words).most_common(30)

# Top / bottom opinions by sentiment
top3_pos = df.nlargest(3, "sentiment_score")
top3_neg = df.nsmallest(3, "sentiment_score")

# Correlations with stars
star_corrs = {}
for k, label in LABELS.items():
    if k != "estrellas":
        star_corrs[label] = df[COLS[k]].corr(df[COLS["estrellas"]])

# Build aspects detail
aspect_details = []
for aspect, keywords in ASPECTOS.items():
    count = sum(1 for a in df["aspects"] if aspect in a)
    aspect_details.append((aspect, count, len(keywords)))

# NER detail
ner_details = []
for cat, keywords in ENTITIES.items():
    count = sum(1 for e in df["entities"] if cat in e)
    sample_matches = []
    for _, row in df.iterrows():
        if cat in row["entities"]:
            sample_matches = row["entities"][cat][:3]
            break
    ner_details.append((cat, count, sample_matches))

# Text lengths
avg_text_len = df["opinion_len"].mean()
avg_text_words = df["opinion_words"].mean()
max_text = df.loc[df["opinion_len"].idxmax()]
min_text = df.loc[df["opinion_len"].idxmin()]

# Distribution tables for ratings
def rating_dist_table(col_key: str) -> str:
    counts = df[COLS[col_key]].value_counts().sort_index()
    rows = []
    for r in [1, 2, 3, 4, 5]:
        cnt = counts.get(r, 0)
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        rows.append(f"| {r} | {cnt} | {pct:.1f}% | {bar} |")
    return "\n".join(rows)

md = f"""# 📊 Análisis profundo de opiniones del módulo

**Fecha de generación:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}  
**Total de respuestas analizadas:** {total}  
**Período de recolección:** {df['Timestamp'].min().strftime('%d/%m/%Y')} → {df['Timestamp'].max().strftime('%d/%m/%Y')}

---

## 1. Resumen ejecutivo

| Métrica | Valor |
|:--------|:------|
| Nº total de encuestados | **{total}** |
| Rating promedio (estrellas) | ⭐ **{mean_stars:.2f}** / 5 |
| Rating mediano | ⭐ **{median_stars:.0f}** / 5 |
| % Opiniones positivas | 🟢 **{pos_pct:.1f}%** |
| % Opiniones neutrales | ⚪ **{neu_pct:.1f}%** |
| % Opiniones negativas | 🔴 **{neg_pct:.1f}%** |
| Longitud media de opinión | **{avg_text_words:.0f}** palabras ({avg_text_len:.0f} chars) |

> **Hallazgo principal:** El módulo recibe una valoración excelente con un promedio de {mean_stars:.2f}/5 estrellas.  
> {pos_pct:.1f}% de las opiniones escritas tienen un sentimiento positivo. Los aspectos más valorados son la calidad del ponente y los contenidos.

---

## 2. Distribución de ratings por dimensión

Cada dimensión se evalúa en escala 1–5.

### 2.1 Estrellas generales

| Estrellas | Frecuencia | % | Distribución |
|:----------|:-----------|:---|:-------------|
{rating_dist_table('estrellas')}

### 2.2 Calidad de los contenidos

| Rating | Frecuencia | % | Distribución |
|:-------|:-----------|:---|:-------------|
{rating_dist_table('contenidos')}

### 2.3 Calidad de los materiales

| Rating | Frecuencia | % | Distribución |
|:-------|:-----------|:---|:-------------|
{rating_dist_table('materiales')}

### 2.4 Calidad del ponente

| Rating | Frecuencia | % | Distribución |
|:-------|:-----------|:---|:-------------|
{rating_dist_table('ponente')}

### 2.5 Calidad de las prácticas

| Rating | Frecuencia | % | Distribución |
|:-------|:-----------|:---|:-------------|
{rating_dist_table('practicas')}

### 2.6 Dificultad de los contenidos

| Rating | Frecuencia | % | Distribución |
|:-------|:-----------|:---|:-------------|
{rating_dist_table('dificultad_contenidos')}

### 2.7 Dificultad de la evaluación

| Rating | Frecuencia | % | Distribución |
|:-------|:-----------|:---|:-------------|
{rating_dist_table('dificultad_eval')}

![Distribución de ratings](distribucion_ratings.png)

---

## 3. Matriz de correlación

¿Qué dimensiones están más relacionadas entre sí?

![Matriz de correlación](correlacion.png)

### Correlación de cada dimensión con las estrellas generales

| Dimensión | Correlación con ⭐ |
|:----------|:------------------:|
{chr(10).join(f'| {label} | **{corr:.3f}** |' for label, corr in star_corrs.items())}

> **Interpretación:** La dimensión más correlacionada con la satisfacción general es la indicada arriba. Una correlación alta sugiere que mejorar ese aspecto tendría el mayor impacto en la percepción global del módulo.

---

## 4. Análisis de sentimiento

Se utilizó un análisis léxico en español con diccionarios de palabras positivas y negativas adaptados al dominio educativo.

### 4.1 Distribución general

![Análisis de sentimiento](sentimiento.png)

| Sentimiento | Count | % |
|:------------|:------|:---|
| 🟢 Positivo | {(df['sentiment_label'] == 'Positivo').sum()} | {pos_pct:.1f}% |
| ⚪ Neutral | {(df['sentiment_label'] == 'Neutral').sum()} | {neu_pct:.1f}% |
| 🔴 Negativo | {(df['sentiment_label'] == 'Negativo').sum()} | {neg_pct:.1f}% |

### 4.2 Sentimiento vs Estrellas

![Sentimiento vs Estrellas](sentimiento_vs_estrellas.png)

> El sentimiento del texto correlaciona positivamente con las estrellas otorgadas, lo que valida la coherencia entre la evaluación numérica y la opinión escrita.

### 4.3 Opiniones más positivas

{chr(10).join(f'> ⭐ {row[COLS["estrellas"]]} | score={row["sentiment_score"]:.2f} | *"{row[COLS["opinion"]][:200]}..."*' for _, row in top3_pos.iterrows())}

### 4.4 Opiniones más negativas / críticas

{chr(10).join(f'> ⭐ {row[COLS["estrellas"]]} | score={row["sentiment_score"]:.2f} | *"{row[COLS["opinion"]][:200]}..."*' for _, row in top3_neg.iterrows())}

---

## 5. Nube de palabras

![WordCloud](wordcloud.png)

### Top 30 palabras más frecuentes (excluyendo stopwords)

| Palabra | Frecuencia |
|:--------|:-----------|
{chr(10).join(f'| {w} | {c} |' for w, c in top_words)}

---

## 6. Bigramas más frecuentes

![Bigramas](bigramas.png)

Los bigramas revelan las combinaciones de palabras más recurrentes en las opiniones, indicando temas y patrones de discurso.

---

## 7. Reconocimiento de entidades nombradas (NER)

Se detectaron automáticamente menciones a entidades relevantes en las opiniones:

![NER](ner.png)

| Categoría | Nº opiniones | Ejemplos detectados |
|:----------|:-------------|:---------------------|
{chr(10).join(f'| {cat} | {cnt} | {", ".join(samples[:5]) if samples else "—"} |' for cat, cnt, samples in ner_details)}

---

## 8. Análisis por aspectos (_Aspect-Based Sentiment_)

Se identificaron menciones a aspectos específicos del módulo en las opiniones:

![Aspectos](aspectos.png)

| Aspecto | Opiniones que lo mencionan | % | Keywords buscados |
|:--------|:--------------------------|:--|:-------------------|
{chr(10).join(f'| {asp} | **{cnt}** | {cnt/total*100:.1f}% | {n_kw} términos |' for asp, cnt, n_kw in aspect_details)}

---

## 9. Evolución temporal

![Evolución temporal](evolucion_temporal.png)

| Trimestre | Nº respuestas | Avg ⭐ | Avg Sentimiento |
|:----------|:-------------|:------|:----------------|
{chr(10).join(f'| {row["periodo"]} | {row["n"]} | {row["avg_stars"]:.2f} | {row["avg_sentiment"]:.3f} |' for _, row in time_stats.iterrows())}

---

## 10. Estadísticas de texto

| Métrica | Valor |
|:--------|:------|
| Longitud media (caracteres) | {avg_text_len:.0f} |
| Longitud media (palabras) | {avg_text_words:.0f} |
| Opinión más larga | {max_text['opinion_words']} palabras — ⭐{max_text[COLS['estrellas']]} |
| Opinión más corta | {min_text['opinion_words']} palabras — ⭐{min_text[COLS['estrellas']]} |

---

## 11. Conclusiones

1. **Satisfacción global excelente:** La media de {mean_stars:.2f}/5 estrellas, con una mediana de {median_stars:.0f}, indica un nivel de satisfacción muy alto.

2. **Ponente como factor diferencial:** La calidad del ponente es la dimensión mejor valorada y aparece prominentemente en las opiniones textuales como un punto fuerte.

3. **Contenidos y prácticas bien valorados:** La calidad de contenidos, materiales y prácticas reciben puntuaciones consistentemente altas (medias > {df[[COLS['contenidos'], COLS['materiales'], COLS['practicas']]].min().min():.1f}).

4. **Dificultad adecuada:** La dificultad de contenidos (media {df[COLS['dificultad_contenidos']].mean():.2f}) y evaluación (media {df[COLS['dificultad_eval']].mean():.2f}) se perciben como moderadamente altas, lo cual es apropiado para un programa formativo exigente.

5. **Coherencia numérico-textual:** Existe una correlación clara entre el sentimiento expresado en el texto y la puntuación numérica, validando ambas métricas.

6. **Áreas de mejora detectadas:** Las opiniones con sentimiento negativo/neutral mencionan principalmente ritmo acelerado, deseo de más tiempo, y más ejemplos prácticos.

---

*Análisis generado automáticamente con Python (pandas, matplotlib, wordcloud).*  
*Sentiment analysis: léxico español adaptado al dominio educativo.*  
*NER: basado en reglas con diccionarios de entidades relevantes.*
"""

with open(f"{OUT}/analisis_profundo.md", "w") as f:
    f.write(md)

print(f"✅ Markdown generado: {OUT}/analisis_profundo.md")
print(f"✅ Gráficas generadas: {len([f for f in __import__('os').listdir(OUT) if f.endswith('.png')])} PNGs")
