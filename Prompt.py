from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)

chain = prompt | llm | StrOutputParser()

chain.invoke({"topic": "programming"})
