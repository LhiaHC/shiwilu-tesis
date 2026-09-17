"""
Dominios semanticos del corpus shiwilu.

Cada dominio agrupa vocabulario en espanol usado para orientar la generacion de
oraciones (Etapa 2) y para clasificar cada par por campo semantico.

Las listas de `semillas` se ampliaron con el vocabulario extraido de los textos
narrativos en la Etapa 0. El resultado completo de esa extraccion vive en
`1_objetivo1_corpus/intermedios/vocabulario_dominios.json`.
"""

COLOR_DOM = {
    "D1_Naturaleza":  "E8F5E9", "D2_Cuerpo":      "FFF3E0",
    "D3_Familia":     "FCE4EC", "D4_Alimentos":   "FFFDE7",
    "D5_Lugar":       "E3F2FD", "D6_Tiempo":      "F3E5F5",
    "D7_Actividades": "E0F2F1", "D8_Social":      "ECEFF1",
}

DOMINIOS = {
    "D1_Naturaleza": {
        "desc": "Entorno natural amazónico: flora, fauna, agua, clima, astros",
        "semillas": [
            "yuca","maíz","leña","agua","fuego","olla de aluminio","olla de barro","clima","pueblos"
        ],
    },
    "D2_Cuerpo": {
        "desc": "Cuerpo humano, salud, estados físicos y enfermedad",
        "semillas": [
            "madre","padre","corazón","boca","dientes","cabeza","mano"
        ],
    },
    "D3_Familia": {
        "desc": "Vínculos familiares, generacionales y de parentesco",
        "semillas": [
            "madre","padre","abuelita","abuelos","abuelas","marido","hijos","hija mujer","hermana",
            "yerno","viuda","hermanas menores"
        ],
    },
    "D4_Alimentos": {
        "desc": "Alimentación, cocina, recolección y preparación de bebidas",
        "semillas": [
            "chicha punta","yuca","maíz","azúcar","comida","bebida","tostadora","batán","moledor",
            "afrecho"
        ],
    },
    "D5_Lugar": {
        "desc": "Espacio, ubicación, deixis, movimiento y poblados",
        "semillas": [
            "pueblo","ciudad grande","chacra","casa","olla","boca","ollas grandes","recipiente","lejano"
        ],
    },
    "D6_Tiempo": {
        "desc": "Temporalidad, ciclos naturales y referencias temporales",
        "semillas": [
            "amanece","anochecer","antes","ahora","después","primero","luego","mientras","cuando",
            "nueve meses"
        ],
    },
    "D7_Actividades": {
        "desc": "Actividades cotidianas, laborales, productivas y rituales",
        "semillas": [
            "engendró","parieron","crió","murió","viviendo","encontró","convivir","salí embarazada","parí",
            "crecieron","moler","tostar","hervir","mezclar","masticar","batir","traer","extraer",
            "pelar","lavar","guarda","hierve","enseñar","aprender","hablar"
        ],
    },
    "D8_Social": {
        "desc": "Interacción social, identidad, valores, emociones colectivas",
        "semillas": [
            "Jeberina","señora","bien","quería","enseñanza","lengua shiwilu","idioma","costumbres","valores",
            "pena","felizmente","difundir","honor","culpa"
        ],
    },
}
