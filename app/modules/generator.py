import os
import httpx
from typing import List

from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

from configs import LLMConfig

from dotenv import load_dotenv

load_dotenv()

# To-do: Add support for Poro LUMI 
# https://huggingface.co/LumiOpen/Poro-34B

class BaseGenerator:
    def __init__(self, config: LLMConfig = LLMConfig.default()):
        self.config = config
        self.client = None
    
    def generate(self, messages: List[str], stream: bool = False):
        response = self.client.stream(messages) if stream \
            else self.client.invoke(messages).content
        return response


class OpenAIGenerator(BaseGenerator):
    def __init__(self, config: LLMConfig = LLMConfig.default()):
        super().__init__(config)

        self.client = ChatOpenAI(
            model=self.config.model, 
            api_key=os.getenv("OPENAI_API_KEY"), 
            temperature=self.config.temperature,
        )


class OllamaGenerator(BaseGenerator):
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

class AzureAIGenerator(BaseGenerator):
    def __init__(
        self, 
        config: LLMConfig,
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
                "request": [self.update_base_url],
            }),
            temperature=self.config.temperature,
        )


    def update_base_url(self, request: httpx.Request):
        if request.url.path == "/chat/completions":
            if self.config.model == "gpt-4o":
                request.url = request.url.copy_with(path="/v1/openai/gpt4o/chat/completions")
            elif self.config.model == "gpt-35-turbo":
                request.url = request.url.copy_with(path="/v1/chat/")
            elif self.config.model == "gpt-4-turbo":
                request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")
            elif self.config.model == "gpt-4-8k":
                request.url = request.url.copy_with(path="/v1/chat/gpt4-8k")
            else:
                raise Exception(f"Model {self.config.model} is not currently supported.")
