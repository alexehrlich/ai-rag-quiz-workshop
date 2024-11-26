from chat_solution.embedding_db import EmbeddingDatabase
from chat_solution.llm import LargeLanguageModel

class LearningAssistant:
    _instance = None

    def get_instance():
        if not LearningAssistant._instance:
            LearningAssistant._instance = LearningAssistant()
        return LearningAssistant._instance
    
    def __init__(self):
        self.embedding_db = EmbeddingDatabase()  # Remove the embedding_model argument
        self.llm = LargeLanguageModel()
        self.conversation_history = []
        self.documents_retrieved = []
        self.instructions = "Just answer in old english shaksepear poems"

    # we give as examples a small chat history to help the LLM understand the task
        self.examples = """<startexample>
    Interaction 1
    New Context: LLMs are large language models that can generate responses to user queries. They are trained on massive datasets to learn patterns, structures, and relationships in text. They can generate responses by combining language generation with real-time data retrieval.
    User input: How do LLMs generate responses?
    Assistant:
    Question: How do LLMs generate responses?
    1. LLMs generate responses by searching the internet for relevant information. 
    2. LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets. (CORRECT)
    3. LLMs generate responses by combining language generation with real-time data retrieval.
    4. LMs generate responses by using a predefined set of rules and templates.
    Interaction 2
    User input: 3
    Assistant:Incorrect! LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets.
    Interaction 3
    User input: 2
    Assistant:Correct! LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets.
    </endexample>
    """

    def query(self, query: str) -> dict:
        documents = None
        # do not populate the context if the user input is a number as we are answering a previous question
        # answer can come as a string number like "2", treat it as a number
        if not query.isnumeric():
            documents = self.embedding_db.retrieve(query)
        self.documents_retrieved = documents

        prompt = self._get_prompt(documents, query)
        response = self.llm.call(prompt)
        self.conversation_history.append((query, response))

        return response

    def _get_prompt(self, documents: str, query: str) -> str:

        chat_history = ""
        i = 1
        for old_query, response in self.conversation_history:
            chat_history += f"Interaction {i}\nUser: {old_query}\nAssistant: {response}\n"
            i += 1
        
        new_context_str = f"\nNew Context: {documents}" if documents else ""

        self.complete_prompt = f"""{self.instructions}
{self.examples}
Now we start the conversation history:
{chat_history}

Just predict the next answer:
Interaction {i+1} {new_context_str}
User input: {query}
Assistant:"""
        return self.complete_prompt
