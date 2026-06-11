# Intelligent Learning Path

Sistema de planificación de trayectoria profesional que combina búsqueda informada (A\*), metaheurísticas (Algoritmo Genético) y un modelo de lenguaje (LLM) para interpretar objetivos en lenguaje natural y evaluar la calidad de las trayectorias propuestas.

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com/) instalado y corriendo (para el componente LLM)

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd Intelligent-Learning-Path

# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
```

Editar `.env` si es necesario:

| Variable           | Descripción            | Default                    |
| ------------------ | ----------------------- | -------------------------- |
| `OLLAMA_HOST`    | URL del servidor Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL`   | Modelo a utilizar       | `qwen3:1.7b`             |
| `OLLAMA_TIMEOUT` | Timeout en segundos     | `120`                    |

## Ejecución

### Interfaz Web (Recomendado)

```bash
# Instalar Streamlit (si no está instalado)
.venv/bin/pip install streamlit>=1.30.0

# Ejecutar la interfaz web
.venv/bin/streamlit run app.py
```

Se abrirá una interfaz estilo ChatGPT en el navegador (`http://localhost:8501`).
Escribe tu objetivo profesional en lenguaje natural y recibe un plan personalizado.

Ejemplo de uso:

```
Quiero ser ML Engineer, ya sé Python, tengo 200 de presupuesto y 30 semanas.
```

### Línea de Comandos (CLI)

```bash
# Descargar el modelo (si no se tiene)
ollama pull qwen3:1.7b

# Ejecutar el sistema
.venv/bin/python main.py
```

El sistema pedirá describir un objetivo profesional en lenguaje natural. Ejemplo:

```
> Quiero ser ML Engineer, ya sé Python, tengo 200 de presupuesto y 30 semanas.
```

Salida esperada:

```
===== PARSED GOAL =====
{ "target_career": "ML Engineer", "initial_skills": ["Python"], ... }

===== A* SEARCH =====
1. Statistics
2. Linux
3. Linear Algebra
4. Docker
5. Machine Learning
6. Deep Learning
Total cost: 546 | Total time: 32 weeks

===== GENETIC ALGORITHM =====
1. Statistics
2. Linux
...
```

## Generar dataset

Para regenerar el dataset sintético (46 cursos, 10 carreras):

```bash
.venv/bin/python data/generator.py
```

## Ejecutar experimentos

```bash
# Comparativa A* vs GA (genera experiments/results.csv)
.venv/bin/python -m experiments.compare_algorithms --trials 5

# Generar gráficos (bar chart + tabla resumen)
.venv/bin/python -m experiments.plot_results
```

Los gráficos se guardan en `experiments/figures/`.

## Ejecutar tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Estructura del proyecto

```
├── data/
│   ├── courses.json          # Catálogo de cursos (generado)
│   ├── careers.json          # Carreras y habilidades requeridas
│   ├── prerequisites.json    # Aristas del grafo de prerequisitos
│   ├── skills_seed.json      # Semilla curada a mano
│   └── generator.py          # Generador sintético del dataset
├── planning/
│   ├── career_planner.py     # Búsqueda A* con heurística admisible
│   ├── metaheuristic.py      # Algoritmo Genético (OX crossover)
│   ├── constraints.py        # Validación de prerequisitos
│   └── state.py              # Estado de búsqueda para A*
├── llm/
│   ├── client.py             # Wrapper de Ollama
│   ├── parser.py             # Parsing de objetivo en lenguaje natural
│   └── evaluator.py          # Evaluación de trayectorias
├── simulation/
│   └── simulator.py          # Simulación estocástica
├── experiments/
│   ├── compare_algorithms.py # Harness de comparación A* vs GA
│   ├── plot_results.py       # Generación de figuras
│   ├── results.csv           # Resultados crudos
│   └── figures/              # Figuras generadas
├── tests/                    # Tests unitarios
├── app.py                    # Interfaz web (Streamlit)
├── main.py                   # Punto de entrada CLI
├── requirements.txt
└── .env.example
```

## Algoritmos

### A\* (búsqueda informada)

- **Estado**: conjunto de cursos completados, camino, costo acumulado, tiempo acumulado.
- **Heurística admisible**: suma de duraciones de los cursos objetivo faltantes (misma unidad que g).
- **Complejidad**: óptimo en tiempo cuando la heurística es consistente.

### Algoritmo Genético

- **Representación**: permutación de cursos candidatos.
- **Operadores**: Order Crossover (OX), mutación por swap, reparación topológica.
- **Fitness**: `w_t × tiempo + w_c × costo + w_m × cursos_faltantes`.
- **Selección**: torneo (k=3), elitismo.

## Rol del LLM

El LLM (Ollama) cumple dos funciones:

1. **Parser de objetivos** (`GoalParser`): convierte una frase en lenguaje natural a un JSON estructurado con carrera objetivo, habilidades iniciales y restricciones.
2. **Evaluador de trayectorias** (`PathEvaluator`): evalúa la calidad de una trayectoria en coherencia, progresión lógica, empleabilidad y utilidad práctica, devolviendo un score de 0 a 10.
