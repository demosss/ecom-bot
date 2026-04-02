import os
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


def find_faq_answer(user_question, faq_data):
    """
    Находит ответ на типовой вопрос из FAQ
    """
    user_question_lower = user_question.lower().strip()

    for item in faq_data:
        if item["q"].lower().strip() == user_question_lower:
            return item["a"]

    # Поиск по частичному совпадению
    for item in faq_data:
        if user_question_lower in item["q"].lower() or item["q"].lower() in user_question_lower:
            return item["a"]

    return None


class CliBot():
    def __init__(self, brand_name, model_name, api_key=None, base_url=None,  system_prompt=None):
        # self.chat_model = ChatOpenAI(
        #     api_key=api_key,
        #     base_url=base_url,
        #     model_name=model_name,
        #     temperature=0.0,
        #     timeout=15
        # )

        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=0.0
        )

        self.store = {}
        self.orders_data = self.load_orders_data()
        self.faq_data = self.load_faq_data()

        if system_prompt is None:
            system_prompt = f"""Ты вежливый и краткий помощник магазина {brand_name}. Отвечай по делу, не фантазируй. 
                                Если вопрос касается информации из FAQ, используй точные ответы из FAQ. 
                                Если вопрос касается статуса заказа, используй команду /order ID.
                                Не выдумывай информацию, которой нет в FAQ или в данных о заказе."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        self.chain = self.prompt | self.chat_model
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        Path("logs").mkdir(exist_ok=True)

    def load_orders_data(self):
        try:
            with open('data/orders.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("Файл data/orders.json не найден")
            return {}
        except json.JSONDecodeError:
            logging.error("Ошибка при чтении файла data/orders.json")
            return {}

    def load_faq_data(self):
        try:
            with open('data/faq.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("Файл data/faq.json не найден")
            return []
        except json.JSONDecodeError:
            logging.error("Ошибка при чтении файла data/faq.json")
            return []

    def get_order_status(self, order_id):
        order_info = self.orders_data.get(str(order_id))

        if order_info is None:
            return f"Заказ с ID {order_id} не найден. Пожалуйста, проверьте номер заказа и попробуйте снова."

        status = order_info.get("status", "неизвестен")
        status_descriptions = {
            "in_transit": "в пути",
            "delivered": "доставлен",
            "processing": "обрабатывается"
        }

        readable_status = status_descriptions.get(status, status)
        response = f"Заказ #{order_id}: статус - {readable_status}"

        if "eta_days" in order_info:
            response += f", ожидаемая доставка через {order_info['eta_days']} дн."
        if "delivered_at" in order_info:
            response += f", дата доставки: {order_info['delivered_at']}"
        if "carrier" in order_info:
            response += f", перевозчик: {order_info['carrier']}"
        if "note" in order_info:
            response += f", примечание: {order_info['note']}"

        return response

    def log_session_entry(self, log_file_path, role, content, usage=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        }
        if usage:
            entry["usage"] = usage

        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def __call__(self, session_id):
        print(
            "Чат-бот запущен! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_path = Path(f"logs/session_{session_id}_{timestamp}.jsonl")
        self.log_session_entry(session_log_path, "system", "=== New session ===")

        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            self.log_session_entry(session_log_path, "user", user_text)

            # Получаем объект истории для текущей сессии
            history = self.get_session_history(session_id)

            # Обработка команды /order
            if user_text.startswith('/order '):
                order_id = user_text.split('/order ', 1)[1].strip()
                bot_reply = self.get_order_status(order_id)
                print(f"Бот: {bot_reply}")

                # Сохраняем сообщения в историю вручную
                history.add_user_message(user_text)
                history.add_ai_message(bot_reply)

                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                self.log_session_entry(session_log_path, "assistant", bot_reply, usage)
                continue

            # Обработка FAQ
            faq_answer = find_faq_answer(user_text, self.faq_data)
            if faq_answer:
                bot_reply = faq_answer
                print(f"Бот: {bot_reply}")

                # Сохраняем сообщения в историю вручную
                history.add_user_message(user_text)
                history.add_ai_message(bot_reply)

                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                self.log_session_entry(session_log_path, "assistant", bot_reply, usage)
                continue

            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                self.log_session_entry(session_log_path, "system", "Пользователь завершил сессию. Сессия окончена.")
                break
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                self.log_session_entry(session_log_path, "system", "Пользователь сбросил контекст.")
                continue

            # Обычный вопрос через LLM
            try:
                response = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}}
                )

                if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                    token_usage = response.response_metadata['token_usage']
                    input_tokens = token_usage.get("prompt_tokens", 0)
                    output_tokens = token_usage.get("completion_tokens", 0)
                    total_tokens = token_usage.get("total_tokens", 0)
                else:
                    input_tokens = output_tokens = total_tokens = 0

                usage = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens
                }

            except Exception as e:
                logging.error(f"[error] {e}")
                print(f"[Ошибка] {e}")
                error_msg = f"[Ошибка] {e}"
                error_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                self.log_session_entry(session_log_path, "assistant", error_msg, error_usage)
                continue

            bot_reply = response.content.strip()
            print(f"Бот: {bot_reply}")
            self.log_session_entry(session_log_path, "assistant", bot_reply, usage)


if __name__ == "__main__":
    # api_key = os.getenv("OPENAI_API_KEY")
    # base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("OPENAI_API_MODEL")
    brand_name = os.getenv("BRAND_NAME")
    # Версия для
    # bot = CliBot(
    #     api_key=api_key,
    #     base_url=base_url,
    #     model_name=model_name,
    #     brand_name=brand_name,
    # )

    bot = CliBot(
        model_name=model_name,
        brand_name=brand_name,
    )
    bot("user_123")