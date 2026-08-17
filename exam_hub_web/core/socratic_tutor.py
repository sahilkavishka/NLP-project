import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

class SocraticTutor:
    """
    NLP module that uses prompt engineering to act as a Socratic Tutor.
    Instead of giving direct answers, it guides the student with hints and questions.
    """
    def __init__(self, llm_model="google/flan-t5-small"):
        print(f"[SocraticTutor] Loading LLM: {llm_model}...")
        # Using text-generation to be compatible with the newer transformers version
        hf_pipe = pipeline("text-generation", model=llm_model, max_new_tokens=100)
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        print("[SocraticTutor] Model loaded.")

    def get_guidance(self, student_question):
        """
        Takes the student's question and returns a Socratic response (a hint or a guiding question).
        """
        print("[SocraticTutor] Generating Socratic response...")
        
        # This is where the magic happens: Strict Prompt Engineering
        prompt_template = """You are a helpful Socratic AI tutor. 
DO NOT give the exact answer. 
Instead, ask a guiding question or give a small hint to make the student think.

Student Question: {question}

Tutor Hint:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        response = chain.invoke({"question": student_question}).strip()
        return response