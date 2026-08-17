import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
model = os.getenv("OPENAI_API_MODEL")

llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model=model,
        )

# Локальные эмбеддинги без OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")

# # эмбеддинги OpenAI (раскомментируйте если решили использовать их)
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings()

example_prompt = PromptTemplate.from_template("Вопрос: {question}\nОтвет: {answer}")
examples = [
    {"question": "Что делать, если опоздал на работу?",
     "answer": "Притворись, что это спецплан компании по тестированию терпения коллег."},
    {"question": "Как победить лень?",
     "answer": "Скажи лени, что завтра — её выходной, и действуй, пока она отдыхает."},
    {"question": "Что делать, если забыл день рождения друга?",
     "answer": "Сделай вид, что это сюрприз для него, и улыбайся, когда он удивлённо морщит лоб."}
]

# Создание ExampleSelector (динамический выбор k ближайших примеров)
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=2  # выбираем 2 ближайших примера
)

# Few-shot шаблон с динамическим ExampleSelector
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Отвечай на вопросы в шутливо-ироничном стиле, как в примерах:",
    suffix="Вопрос: {question}\nОтвет:",
    input_variables=["question"]
)

# Новый запрос — ExampleSelector выберет ближайший пример по смыслу
question = "Как перестать лениться?"
formatted_prompt = prompt.format(question=question)
response = llm.invoke(formatted_prompt)

print(response.content)