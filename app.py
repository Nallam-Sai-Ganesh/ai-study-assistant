from  dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()
mistral=ChatMistralAI(model="mistral-small", temperature=0.9)

#templates
summ_temp=PromptTemplate(
    input_variables=["text"],
    template="""
    Summarize the following text into 5 bullet points.
    {text}
    """
)
quiz_temp=PromptTemplate(
    input_variables=["text"],
    template="""
    Create a quiz based on the following text.Have 5 questions with 4 options each and indicate the correct answer.
    {text}
    """
)
chat_temp=PromptTemplate(
    input_variables=["text"],
    template="""
    Create a chat based on the following text.
    Handle the conversation in a natural way and provide informative responses.
    Answer from the context of the text and provide accurate information.
    if the information is not available in the text, say you don't know.
    {text}
    """
)
print("\nWelcome to the AI Study Assistant!\n");
print("Enter the text :\n")
query_text=input()

while True:
    print("\n\n")
    print("Choose an option:\n");
    print("1. Summarize Text\n");
    print("2. Create Quiz\n");
    print("3. Chat with AI\n");
    print("4. Exit\n");
    option=input("Enter your choice: ")
    if option == "1":
        chain = summ_temp | mistral
        res = chain.invoke({"text": query_text})
        print("Summary:\n")
        print(res.content)

    elif option == "2":
        chain = quiz_temp | mistral
        res = chain.invoke({"text": query_text})
        print("Quiz:\n")
        print(res.content)

    elif option == "3":
        print("Start chatting with the AI! (type 'exit' to end the chat)\n")

        while True:
            user_input = input("You: ")

            if user_input.lower() == "exit":
                print("Ending chat. Goodbye!")
                break

            chain = chat_temp | mistral
            res = chain.invoke({
                "text": query_text,
                "question": user_input
            })
            print("AI:", res.content)

    elif option == "4":
        print("Exiting the program. Goodbye!")
        break

    else:
        print("Invalid option. Please try again.")
