import sys
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class WeatherInfo(BaseModel):
    city: str = Field(description="Название города")
    temperature: float = Field(description="Текущая температура в градусах Цельсия")
    condition: str = Field(description="Короткое описание текущей погоды, e.g. 'ясно', 'облачно', 'дождь'")


def get_weather_info(city: str) -> str:
    """Return a JSON string with WeatherInfo for *city*, or {"error": "..."} on failure."""
    try:
        model = ChatOpenAI(
            model_name=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"),
            temperature=0.7,
        )

        parser = PydanticOutputParser(pydantic_object=WeatherInfo)

        prompt = PromptTemplate(
            template=(
                "Придумай правдоподобные данные о погоде для города: {city}\n"
                "Условие (condition) пиши на русском языке.\n\n"
                "{format_instructions}"
            ),
            input_variables=["city"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser

        result: WeatherInfo = chain.invoke({"city": city})
        return result.model_dump_json(indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)


def main() -> None:
    if len(sys.argv) > 1:
        city = " ".join(sys.argv[1:]).strip()
    else:
        city = input("Введите название города: ").strip()

    if not city:
        print(json.dumps({"error": "Название города не может быть пустым"}, ensure_ascii=False, indent=2))
        sys.exit(1)

    print(get_weather_info(city))


if __name__ == "__main__":
    main()
