import os
import httpx
from typing import Dict, List, Tuple

from langchain.messages import AIMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

from configs import LLMConfig

from dotenv import load_dotenv
load_dotenv()


def update_base_url(request: httpx.Request, model: str):
    model_paths = {
        "gpt-4o": "/v1/openai/deployments/gpt-4o-2024-08-06/chat/completions",
        "gpt-4.1": "/v1/openai/deployments/gpt-4.1-2025-04-14/chat/completions",
        "gpt-3.5-turbo": "/v1/chat/gpt-35-turbo-1106",
        "gpt-4-turbo": "/v1/openai/gpt4-turbo/chat/completions",
        "gpt-4-8k": "/v1/chat/gpt4-8k",
        "gpt-4o-mini": "/v1/openai/deployments/gpt-4o-mini-2024-07-18/chat/completions",
        "gpt-4.1-mini": "/v1/openai/deployments/gpt-4.1-mini-2025-04-14/chat/completions",
        "gpt-5": "/v1/openai/responses",
        "gpt-5.1": "/v1/openai/responses",
    }

    if model not in model_paths:
        raise Exception(f"Model {model} is not currently supported.")
    
    request.url = request.url.copy_with(path=model_paths[model])


class BaseGenerator:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.is_responses_api = False

    def _format_messages_response_api(self, messages: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        formatted_messages = []
        for m in messages:
            if isinstance(m, tuple) and len(m) == 2:
                role, content = m
                formatted_messages.append({"role": role, "content": content})
            elif isinstance(m, dict) and "role" in m and "content" in m:
                formatted_messages.append(m)
            else:
                raise ValueError(
                    f"Invalid message format: {m}. Must be tuple(role, text) or dict with 'role' and 'content'."
                )
        return formatted_messages
    
    def _extract_text(self, response: AIMessage) -> str:
        """
        Extract displayable text from a Responses API response or LangChain AIMessage.
        """
        if hasattr(response, "output_text"):
            return response.output_text

        content = getattr(response, "content", None)
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts)

        if isinstance(content, str):
            return content

        return ""

    def generate(
        self,
        messages: List[Tuple[str, str]],
        return_content: bool = True,
    ) -> str | AIMessage:
        """
        messages: List of tuples [('user', 'Hello'), ('assistant', 'Hi')]
        """

        input_messages = (
            self._format_messages_response_api(messages)
            if self.is_responses_api
            else messages
        )

        response = self.client.invoke(input_messages)

        if not return_content:
            return response

        if self.is_responses_api:
            return self._extract_text(response)

        return response.content


class AzureAIAgent(BaseGenerator): 
    RESPONSE_API_MODEL_MAPPING = {
        "gpt-5": "gpt-5-2025-08-07",
        "gpt-5.1": "gpt-5.1-2025-11-13",
    }

    def __init__(
        self, 
        config: LLMConfig = LLMConfig.default(),
        base_url: str = "https://aalto-openai-apigw.azure-api.net",
    ):
        super().__init__(config)
        model = self.config.model
        full_model = self.RESPONSE_API_MODEL_MAPPING.get(model, model)
        self.is_responses_api = model in self.RESPONSE_API_MODEL_MAPPING

        self.client = ChatOpenAI(
            default_headers={
                "Ocp-Apim-Subscription-Key": os.getenv("AALTO_OPENAI_API_KEY")
            },
            base_url=base_url,
            api_key=os.getenv("AALTO_OPENAI_API_KEY"),
            http_client=httpx.Client(
            event_hooks={
                "request": [lambda request: update_base_url(request, model=model)],
            }),
            temperature=0.,
            model=full_model,
            use_responses_api=self.is_responses_api,
            # reasoning_effort="minimal"
        )


class OpenAIAgent(BaseGenerator):
    def __init__(self, config: LLMConfig = LLMConfig.default()):
        super().__init__(config)

        self.client = ChatOpenAI(
            model=self.config.model, 
            api_key=os.getenv("OPENAI_API_KEY"), 
            temperature=self.config.temperature,
        )


class OllamaAgent(BaseGenerator):
    def __init__(
        self, 
        config: LLMConfig = LLMConfig.default(), 
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(config)
        
        self.client = ChatOllama(
            base_url=base_url,
            model=self.config.model, 
            temperature=self.config.temperature,
        )