# from openai import OpenAI

# def create_client(api_key: str):
#     return OpenAI(api_key = api_key)

# def ask_gpt(client, model, prompt, temperature):
#     completion = client.chat.completions.create(
#         model=model,
#         messages=prompt,
#         max_completion_tokens = 6000,
#         temperature=temperature
#     )
#     return completion.choices[0].message.content

from openai import AsyncOpenAI
import asyncio

def create_client(api_key: str):
    return AsyncOpenAI(api_key=api_key)

async def ask_gpt(client, model: str, prompt: list, temperature: float = 1, timeout: int = 30) -> str:
    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=prompt,
                max_completion_tokens=6000,
                temperature=temperature,
            ),
        timeout=timeout,
        )
        return completion.choices[0].message.content
    except asyncio.TimeoutError:
        return f"[TIME OUT] after {timeout} seconds"
