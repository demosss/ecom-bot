"""
Демонстрация few-shot промпта для бота-программиста.

Скрипт:
1. Загружает примеры из examples.yaml
2. Показывает ответ модели БЕЗ примеров (zero-shot)
3. Показывает ответ модели С примерами (few-shot)
4. (Опционально) Использует SemanticSimilarityExampleSelector
"""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ============================================================
# 1. Настройка LLM
# ============================================================
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    model=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"),
    temperature=0.2,
)

# ============================================================
# 2. Загрузка примеров из YAML
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

def load_examples(path: str | None = None) -> list[dict]:
    if path is None:
        path = SCRIPT_DIR / "examples.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["examples"]

examples = load_examples()
print(f"Загружено примеров: {len(examples)}\n")

# ============================================================
# 3. Тестовый вопрос (как в задании)
# ============================================================
test_question = "Как отсортировать список объектов по полю в Python?"

# ============================================================
# 4. ZERO-SHOT: запрос без примеров
# ============================================================
print("=" * 60)
print("ZERO-SHOT (без примеров)")
print("=" * 60)

zero_shot_prompt = PromptTemplate.from_template(
    "Ты — эксперт по Python. Ответь на вопрос пользователя.\n\n"
    "Вопрос: {question}\n"
    "Ответ:"
)

zero_shot_response = llm.invoke(zero_shot_prompt.format(question=test_question))
print(zero_shot_response.content)

# ============================================================
# 5. FEW-SHOT: с примерами через FewShotPromptTemplate
# ============================================================
print("\n" + "=" * 60)
print("FEW-SHOT (с примерами из YAML)")
print("=" * 60)

# Шаблон одного примера
example_prompt = PromptTemplate.from_template(
    "Вопрос: {question}\n"
    "Ответ:\n{answer}\n"
)

# Сам few-shot промпт
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=(
        "Ты — эксперт по Python. "
        "Всегда отвечай в следующем формате: сначала краткое пояснение, "
        "затем пример кода в блоке ```python ... ```, затем короткое пояснение к коду.\n\n"
        "Ниже приведены примеры ответов. Следуй этому стилю."
    ),
    suffix="Вопрос: {question}\nОтвет:\n",
    input_variables=["question"],
)

formatted_prompt = few_shot_prompt.format(question=test_question)
few_shot_response = llm.invoke(formatted_prompt)
print(few_shot_response.content)

# ============================================================
# 6. (ОПЦИОНАЛЬНО) FEW-SHOT с ExampleSelector
# ============================================================
# ExampleSelector подбирает k наиболее релевантных примеров по семантике,
# что экономит контекстное окно и повышает точность.
print("\n" + "=" * 60)
print("FEW-SHOT с SemanticSimilarityExampleSelector (k=2)")
print("=" * 60)

# Используем локальные эмбеддинги, чтобы не тратить токены OpenAI
embeddings = HuggingFaceEmbeddings(
    model_name="./models/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=2,  # только 2 самых релевантных примера
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=(
        "Ты — эксперт по Python. "
        "Отвечай в формате: пояснение → код → пояснение к коду.\n\n"
        "Примеры для подражания:"
    ),
    suffix="Вопрос: {question}\nОтвет:\n",
    input_variables=["question"],
)

dynamic_response = llm.invoke(dynamic_prompt.format(question=test_question))
print(dynamic_response.content)

# ============================================================
# 7. Сравнение длины промптов
# ============================================================
print("\n" + "=" * 60)
print("Сравнение длины промптов (в символах)")
print("=" * 60)
print(f"Zero-shot:  {len(zero_shot_prompt.format(question=test_question)):>5} символов")
print(f"Few-shot:   {len(formatted_prompt):>5} символов")
print(f"Dynamic:    {len(dynamic_prompt.format(question=test_question)):>5} символов (только k=2 примера)")