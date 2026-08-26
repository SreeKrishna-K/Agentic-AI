import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class ReflectionResult(BaseModel):
    approved: bool = Field(
        description="True if the tweet is ready to post with no further revision needed."
    )
    critique: str = Field(
        description="Detailed critique and recommendations, including length, virality, style, etc."
    )


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
            "Set approved to True only if the tweet is already excellent and should be posted as-is.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"))
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm.with_structured_output(ReflectionResult)