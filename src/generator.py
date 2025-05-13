import os
import httpx
from typing import List, Tuple

from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

from configs import LLMConfig

from dotenv import load_dotenv
load_dotenv()


def update_base_url(request: httpx.Request, model: str):
    if request.url.path == "/chat/completions":
        if model == "gpt-4o":
            request.url = request.url.copy_with(path="/v1/openai/deployments/gpt-4o-2024-08-06/chat/completions")
        elif model == "gpt-3.5-turbo":
            request.url = request.url.copy_with(path="/v1/chat/gpt-35-turbo-1106")
        elif model == "gpt-4-turbo":
            request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")
        elif model == "gpt-4-8k":
            request.url = request.url.copy_with(path="/v1/chat/gpt4-8k")
        else:
            raise Exception(f"Model {model} is not currently supported.")
        

class BaseGenerator:
    def __init__(self, config: LLMConfig = LLMConfig.default()):
        self.config = config
        self.client = None
    
    def generate(self, messages: List[Tuple[str]], return_content: bool = True):
        response = self.client.invoke(messages)
        return response.content if return_content else response


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


class AzureAIAgent(BaseGenerator):
    def __init__(
        self, 
        config: LLMConfig = LLMConfig.default(),
        base_url: str = "https://aalto-openai-apigw.azure-api.net",
    ):
        super().__init__(config)

        self.client = ChatOpenAI(
            default_headers={
                "Ocp-Apim-Subscription-Key": os.getenv("AALTO_OPENAI_API_KEY")
            },
            base_url=base_url,
            api_key=None,
            http_client=httpx.Client(
            event_hooks={
                "request": [lambda request: update_base_url(request, model=self.config.model)],
            }),
            temperature=self.config.temperature,
        )