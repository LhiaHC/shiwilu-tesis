"""
Anotacion automatica por reglas.

`anotar_intencion` implementa el sistema de reglas basado en patrones
morfosintacticos del espanol descrito en
`docs/metodologia_construccion_corpus.md`. Devuelve la intencion y un puntaje de
confianza usado despues para priorizar los pares de mayor calidad.

La Fase 2 importa esta funcion como baseline con el que contrastar los
clasificadores basados en embeddings, de modo que cualquier cambio en estas
reglas afecta tambien a los resultados del analisis.
"""

import re

from shiwilu.dominios import DOMINIOS


def anotar_intencion(texto: str) -> tuple[str, float]:
    if not isinstance(texto, str) or not texto.strip():
        return ("DES", 0.5)
    t = texto.strip().lower().lstrip("¡")

    if re.search(r"^(hola|buenos?\s+días?|buenas?\s+tardes?|buenas?\s+noches?|"
                 r"buenas?\.|adiós\.|chau[\.,!]|hasta\s+(luego|pronto)|"
                 r"bienvenid[oa]|feliz\s+(navidad|año|pascua)|nos\s+vemos|"
                 r"que\s+descanses|que\s+te\s+vaya|mucho\s+gusto)", t, re.I):
        return ("SAL", 0.95)

    if re.search(r"^¿(puedes?|pueden|podrías?|podríamos|"
                 r"me\s+ayudas?|nos\s+ayudas?|"
                 r"me\s+haces?|nos\s+haces?|"
                 r"me\s+traes?|nos\s+traes?|"
                 r"me\s+das?|nos\s+das?|"
                 r"me\s+prestas?|nos\s+prestas?|"
                 r"me\s+llevas?|nos\s+llevas?)", t, re.I):
        return ("REQUEST", 0.88)

    if t.lstrip().startswith("¿"):
        return ("PRG", 0.98)

    if re.search(r"^(¡?gracias[\.,!]?$|lo\s+siento|me\s+(gusta|encanta|alegra|"
                 r"da\s+pena|da\s+tristeza|da\s+gusto)|qué\s+(bonito|lástima|"
                 r"pena|tristeza|alegría|susto|rico|bueno|miedo|orgullo)|qué\b)", t, re.I):
        return ("EMO", 0.90)

    if re.search(r"^(sí[,\.]?\s*(claro|por\s+supuesto)?\.?$|de\s+acuerdo|"
                 r"está\s+(bien|perfecto|listo)|claro\s+que\s+sí|así\s+es|"
                 r"es\s+(verdad|correcto|cierto)|por\s+supuesto|"
                 r"claro[,\.]?\s|sí[,\.]?\s+(ya|hay|tengo|llegué|regresé|"
                 r"cocinamos|hice|traje|hervimos|aprendí|sé|puedo))", t, re.I):
        return ("AFI", 0.85)

    if re.search(r"^(no\s+(puedo|quiero|tengo|sé|entiendo|hay|voy|me\s+gusta|"
                 r"quiero\s+que|puedo\s+ir|hay\s+camino)|me\s+niego|"
                 r"nunca|imposible|ni\s+idea|nadie)", t, re.I):
        return ("NEG", 0.85)

    # negación 1ª persona: "No como.", "No pico.", "No sigo." (verbo -o = indicativo presente)
    if re.search(r"^no\s+[a-záéíóúñü]+o\b[\.\!\,]?\s*$", t, re.I):
        return ("NEG", 0.82)

    if re.search(r"^(ayúda(me|nos|lo)|dame|deme|danos|necesito|llámame|"
                 r"escríbeme|cuéntame|dime|explícame|muéstrame|"
                 r"tráe(me|nos|lo)|apóya(me|nos)|por\s+favor\s*$)", t, re.I):
        return ("REQUEST", 0.85)

    if re.search(r"^(ve\b|vete|vaya\b|corre\b|salta\b|espera\b|escucha\b|sal\b|entra\b|"
                 r"ven\b|toma\s|agarra\s|suelta\b|baja\b|sube\b|siéntate\b|levanta\b|"
                 r"párate\b|mira\s|abre\s|cierra\s|pon\s|quédate\b|regresa\b|"
                 r"hazlo\b|haz\s|sigue\s|continúa\b|trabaja\b|canta\b|descansa\b|"
                 r"estudia\b|lee\s|escribe\s|repite\b|habla\s|termina\b|empieza\b|"
                 r"trae\s|lleva\s|busca\s|guarda\s|limpia\s|responde\b|"
                 r"vengan\b|entren\b|salgan\b|esperen\b|escuchen\b|coman\b|beban\b|"
                 r"estate\s|especifica\b|especifiquen\b|"
                 r"tome\b|ponga\b|venga\b|siga\b|haga\b|ponte\b)", t, re.I):
        return ("REQUEST", 0.85)

    if re.search(r"^(cantemos|juguemos|nademos|trabajemos|caminemos|votemos|"
                 r"conversemos|charlemos|renunciemos|besémonos|comamos|"
                 r"bebamos|hablemos|vayamos|entremos|salgamos|busquemos|"
                 r"hagamos|digamos|pongamos|traigamos|llevemos|corramos|"
                 r"descansemos|aprendamos|intentemos|probemos|veamos)", t, re.I):
        return ("REQUEST", 0.85)

    if re.search(r"^(mantente|manténgase|mantén\s)", t, re.I):
        return ("REQUEST", 0.85)

    if re.search(r"^(déjame|déjalo|déjale|déjalos|déjenos|déjala)", t, re.I):
        return ("REQUEST", 0.85)

    if re.search(r"^(abrázame|abrázalo|levántalo|levántate|levántala|"
                 r"reemplázalo|reemplázala|detente|deténgase|sonría|sonríe|"
                 r"huelan\b|háblame|háblales|háblale|olvídalo|"
                 r"ignóralo|cuéntalo|préstame|préstalo|pásamelo|dámelo|"
                 r"tráelos|tráelas|llévalo|llévala|guárdalo|guárdala|"
                 r"bótalo|suéltalo|suéltala|ábrelo|ábrela|ciérralo|"
                 r"ciérrala|envíalo|envíala|muéstrame|acércate|acercate|acérquese|"
                 r"vístete|vistete|"
                 r"anota\s|anoten\s|obsérvalos|obsérvalas|obsérvame|"
                 r"vigílame|vigílalos|mírame|míralos|usa\s|usen\s|"
                 r"prueba\s|prueben\s|enciéndelo|enciéndela|apágalo|"
                 r"apágala|para\b|¡para\b|salta\b|¡salta\b|"
                 r"pregúntaselo|pregúntale|respira\s|contrólate|"
                 r"no\s+preguntes|vuelve\b|venid\s|atrápalo|"
                 r"atrápenlo|captúrenlo|captúralo|miren\s|"
                 r"pueden\s+irse|puedes\s+irte|pueden\s+salir|"
                 r"puedes\s+salir|pueden\s+pasar|puedes\s+pasar)", t, re.I):
        return ("REQUEST", 0.85)

    # imperativo negativo corto: "No fumes.", "¡No dispares!", "No copien."
    _m_neg = re.match(r"^no\s+([a-záéíóúñü]+)\b[\.\!\,]?\s*$", t, re.I)
    if _m_neg:
        _verb = _m_neg.group(1).lower()
        _excl = {"puedo","puede","podemos","pueden","puedes",
                 "quiero","quiere","queremos","quieren","quieres",
                 "tengo","tiene","tenemos","tienen","tienes",
                 "sé","sabe","sabemos","saben","sabes",
                 "entiendo","entiende","entendemos","entienden","entiendes",
                 "hay","voy","va","vamos","van","vas",
                 "es","son","está","están","estoy","estás",
                 "gracias","nada","nadie","nunca","más","menos"}
        if _verb not in _excl:
            return ("REQUEST", 0.85)

    # "sé/se + infinitivo" = "sé correr", "sé nadar", "sé cocinar" → AFI
    if re.search(r"^s[eé]\s+[a-záéíóúñü]+[aei]r\b", t, re.I):
        return ("AFI", 0.82)

    # imperativo de "ser": "sé tolerante", "sé puntual", "sé razonable"
    if re.search(r"^sé\s+(?!que\b|cómo\b|cuándo\b|dónde\b|si\b|lo\b|la\b|[a-záéíóúñü]+[aei]r\b)", t, re.I):
        return ("REQUEST", 0.85)

    # imperativo + clítico "me/nos": "cocíname", "prepárame", "sírveme"
    if re.search(r"^(cocíname|cocínanos|sírveme|sírvenos|prepárame|prepáranos|"
                 r"cárgame|alcánzame|alcánzanos|arrímate|aléjate|"
                 r"cúbrete|abrígate|báñate|lávate|levántame|préstame)", t, re.I):
        return ("REQUEST", 0.85)

    return ("DES", 0.55)


def anotar_dominio(texto: str) -> str:
    if not isinstance(texto, str):
        return "D8_Social"
    t = texto.lower()
    scores = {d: sum(1 for s in info["semillas"] if s in t)
              for d, info in DOMINIOS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "D8_Social"
