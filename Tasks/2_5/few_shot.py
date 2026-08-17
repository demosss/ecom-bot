import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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

example_prompt = PromptTemplate.from_template(
    "Ввод: {input}\nВывод: {output}"
)

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"}
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Дайте антоним каждого слова:",
    suffix="Ввод: {input}\nВывод:",
    input_variables=["input"]
)

formatted_prompt = prompt.format(input="Высокий")
response = llm.invoke(formatted_prompt)

print(response.content) # Вывод: old