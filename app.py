import json
import logging
from pathlib import Path

import streamlit as st

from llm.evaluator import PathEvaluator
from llm.parser import GoalParser, GoalSchema
from planning.career_planner import CareerPlanner
from planning.greedy import GreedySolver, GreedyResult
from planning.metaheuristic import GeneticAlgorithmSolver, GAResult
from planning.state import State
from simulation.simulator import LearningSimulator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark-mode CSS
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
/* ---- base ---- */
.stApp {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
header[data-testid="stHeader"] {
    background-color: #1a1a2e;
}
/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: #16213e;
}
/* ---- chat bubbles ---- */
div[data-testid="stChatMessage"] {
    background-color: #16213e;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
/* ---- metrics ---- */
div[data-testid="stMetric"] {
    background-color: #0f3460;
    border-radius: 8px;
    padding: 0.8rem;
}
/* ---- expander ---- */
details {
    background-color: #16213e;
    border-radius: 8px;
    padding: 0.5rem;
}
/* ---- inputs ---- */
div[data-testid="stChatInput"] {
    background-color: #16213e;
    border-radius: 12px;
}
/* ---- buttons ---- */
button[kind="primary"] {
    background-color: #e94560;
}
</style>
"""

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data
def _load_careers() -> list[dict]:
    return json.loads(Path("data/careers.json").read_text())

@st.cache_data
def _load_courses() -> list[dict]:
    return json.loads(Path("data/courses.json").read_text())

def _career_names() -> list[str]:
    return [c["career"] for c in _load_careers()]

def _all_course_names() -> list[str]:
    return sorted(c["name"] for c in _load_courses())

def _target_skills(career: str) -> set[str]:
    for c in _load_careers():
        if c["career"] == career:
            return set(c["skills"])
    return set()

# ---------------------------------------------------------------------------
# Planning helpers (cached per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_planner() -> CareerPlanner:
    return CareerPlanner()

@st.cache_resource
def _get_ga() -> GeneticAlgorithmSolver:
    return GeneticAlgorithmSolver()

@st.cache_resource
def _get_greedy() -> GreedySolver:
    return GreedySolver()

def _run_astar(initial: list[str], career: str, budget: int | None, weeks: int | None) -> State | None:
    return _get_planner().plan(
        initial_skills=initial,
        target_career=career,
        max_budget=budget,
        max_weeks=weeks,
    )

def _run_ga(
    initial: list[str],
    career: str,
    budget: int | None,
    weeks: int | None,
) -> GAResult | None:
    return _get_ga().solve(
        initial_skills=initial,
        target_career=career,
        max_budget=budget,
        max_weeks=weeks,
    )

def _run_greedy(initial: list[str], career: str, budget: int | None, weeks: int | None) -> GreedyResult | None:
    return _get_greedy().solve(
        initial_skills=initial,
        target_career=career,
        max_budget=budget,
        max_weeks=weeks,
    )

def _run_simulation(path: list[str]) -> dict:
    return LearningSimulator().simulate(path)

def _run_llm_evaluation(career: str, path: list[str]):
    try:
        return PathEvaluator().evaluate(career=career, path=path)
    except Exception as exc:
        logger.warning("LLM evaluation failed: %s", exc)
        return None

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _show_goal(goal: GoalSchema) -> None:
    with st.chat_message("assistant"):
        st.markdown("### Objetivo detectado")
        cols = st.columns(4)
        cols[0].metric("Carrera", goal.target_career)
        cols[1].metric("Skills iniciales", ", ".join(goal.initial_skills) if goal.initial_skills else "Ninguna")
        cols[2].metric("Presupuesto", f"${goal.max_budget}" if goal.max_budget else "Sin límite")
        cols[3].metric("Semanas", f"{goal.max_weeks}" if goal.max_weeks else "Sin límite")

def _show_astar(result: State | None) -> None:
    with st.chat_message("assistant"):
        if result is None:
            st.warning("No se encontró una ruta factible con A*.")
            return
        st.markdown("### Búsqueda A* (Óptima)")
        st.metric("Tiempo total", f"{result.total_time} semanas")
        st.metric("Costo total", f"${result.total_cost}")
        with st.expander(f"Ver ruta ({len(result.path)} cursos)"):
            for i, c in enumerate(result.path, 1):
                st.markdown(f"**{i}.** {c}")

def _show_ga(result: GAResult | None) -> None:
    with st.chat_message("assistant"):
        if result is None:
            st.warning("El algoritmo genético no encontró una solución factible.")
            return
        st.markdown("### Algoritmo Genético")
        cols = st.columns(3)
        cols[0].metric("Tiempo total", f"{result.total_time} semanas")
        cols[1].metric("Costo total", f"${result.total_cost}")
        cols[2].metric("Fitness", f"{result.fitness:.2f}")
        with st.expander(f"Ver ruta ({len(result.path)} cursos)"):
            for i, c in enumerate(result.path, 1):
                st.markdown(f"**{i}.** {c}")

def _show_greedy(result: GreedyResult | None) -> None:
    with st.chat_message("assistant"):
        if result is None:
            st.warning("El algoritmo greedy no encontró una solución factible.")
            return
        st.markdown("### Búsqueda Greedy (Baseline)")
        cols = st.columns(3)
        cols[0].metric("Tiempo total", f"{result.total_time} semanas")
        cols[1].metric("Costo total", f"${result.total_cost}")
        cols[2].metric("Cursos", result.courses_taken)
        with st.expander(f"Ver ruta ({result.courses_taken} cursos)"):
            for i, c in enumerate(result.path, 1):
                st.markdown(f"**{i}.** {c}")

def _show_simulation(sim: dict) -> None:
    with st.chat_message("assistant"):
        st.markdown("### Simulación Estocástica")
        cols = st.columns(2)
        cols[0].metric("Semanas estimadas", sim["total_weeks"])
        cols[1].metric("Prob. de abandono", f"{sim['abandonment_probability']:.0%}")

def _show_llm_evaluation(ev) -> None:
    with st.chat_message("assistant"):
        if ev is None:
            st.info("Evaluación LLM no disponible (¿Ollama ejecutándose?)")
            return
        st.markdown("### Evaluación LLM")
        st.metric("Score", f"{ev.score}/10")
        st.markdown(f"**Razón:** {ev.rationale}")
        if ev.weaknesses:
            st.markdown("**Debilidades:**")
            for w in ev.weaknesses:
                st.markdown(f"- {w}")
        if ev.suggestions:
            st.markdown("**Sugerencias:**")
            for s in ev.suggestions:
                st.markdown(f"- {s}")

# ---------------------------------------------------------------------------
# Process a user message
# ---------------------------------------------------------------------------
def _process_message(user_text: str, llm_eval: bool = True) -> None:
    # 1. Parse goal with LLM
    with st.spinner("Analizando tu objetivo con el LLM..."):
        try:
            goal = GoalParser().parse(user_text)
        except Exception as exc:
            with st.chat_message("assistant"):
                st.error(f"No pude entender tu objetivo: {exc}")
            return

    _show_goal(goal)

    # Validate career exists
    target = _target_skills(goal.target_career)
    if not target:
        with st.chat_message("assistant"):
            st.error(f"Carrera desconocida: '{goal.target_career}'. "
                     f"Carreras disponibles: {', '.join(_career_names())}")
        return

    # 2. A* search
    with st.spinner("Ejecutando búsqueda A*..."):
        astar = _run_astar(goal.initial_skills, goal.target_career, goal.max_budget, goal.max_weeks)
    _show_astar(astar)

    # 3. Genetic Algorithm
    with st.spinner("Ejecutando algoritmo genético..."):
        ga = _run_ga(goal.initial_skills, goal.target_career, goal.max_budget, goal.max_weeks)
    _show_ga(ga)

    # 4. Greedy search (baseline)
    with st.spinner("Ejecutando búsqueda greedy..."):
        greedy = _run_greedy(goal.initial_skills, goal.target_career, goal.max_budget, goal.max_weeks)
    _show_greedy(greedy)

    # 5. Simulation (on A* path if available)
    if astar is not None:
        with st.spinner("Simulando progreso..."):
            sim = _run_simulation(astar.path)
        _show_simulation(sim)

    # 6. LLM evaluation (on A* path if available)
    if astar is not None and llm_eval:
        with st.spinner("Evaluando ruta con LLM..."):
            ev = _run_llm_evaluation(goal.target_career, astar.path)
        _show_llm_evaluation(ev)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Intelligent Career Planner",
        page_icon="🎯",
        layout="wide",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## Opciones")
        st.markdown("---")
        if st.button("Limpiar historial", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        llm_eval = st.toggle("Evaluación LLM", value=True, help="Activar para evaluar la ruta con el modelo de lenguaje")
        st.markdown("---")
        st.markdown("#### Carreras disponibles")
        for name in _career_names():
            st.markdown(f"- {name}")
        st.markdown("---")
        st.markdown("#### Ejemplo de uso")
        st.code(
            'Quiero ser ML Engineer, sé Python y tengo 200 de presupuesto y 30 semanas',
            language=None,
        )

    # ---- Session state ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---- Header ----
    st.markdown("# Intelligent Career Planner")
    st.markdown("*Describe tu objetivo profesional en lenguaje natural y recibe un plan personalizado.*")

    # ---- Render历史 messages ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Chat input ----
    if user_text := st.chat_input("Describe tu objetivo profesional..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Process
        _process_message(user_text, llm_eval=llm_eval)

        # Store acknowledgement
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Plan generado. Puedes hacer otra consulta o ajustar los parámetros.",
        })


if __name__ == "__main__":
    main()
