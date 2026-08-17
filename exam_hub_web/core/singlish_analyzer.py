import warnings
warnings.filterwarnings("ignore")
import re
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

class SinglishAnalyzer:
    """
    Advanced NLP module to handle Sri Lankan Code-Switched (Singlish) text.
    It maps Singlish terms to English and uses an LLM to generate the answer.
    """
    def __init__(self, llm_model="google/flan-t5-small"):
        print("[SinglishAnalyzer] Loading LLM for intent resolution...")
        hf_pipe = pipeline("text-generation", model=llm_model, max_new_tokens=150)
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        
        # Super Dictionary: Mapping common Singlish question terms to English
        self.singlish_dict = {
            "mokakda": "what is",
            "mokadda": "what is",
            "kohomada": "how does",
            "ai": "why",
            "karanne": "do",
            "kiyala": "explain",
            "dennako": "please give",
            "denna": "give",
            "pahadili": "explain",
            "therenne": "understand",
            "wada": "work",
            "wenasa": "difference",
            "monawada": "what are",
            "koy": "which",
            "eka": "one"
        }
        print("[SinglishAnalyzer] Singlish Dictionary Loaded.")

    def analyze_and_answer(self, singlish_text):
        """
        Translates Singlish to Pseudo-English, keeps technical terms intact, 
        and generates an educational answer.
        """
        print(f"[SinglishAnalyzer] Original Input: {singlish_text}")
        
        # 1. Normalize and Translate (Singlish -> Pseudo-English)
        words = singlish_text.lower().split()
        translated_words = []
        
        for w in words:
            # Remove punctuation for matching
            clean_w = re.sub(r'[^a-z]', '', w)
            if clean_w in self.singlish_dict:
                translated_words.append(self.singlish_dict[clean_w])
            else:
                # Keep English technical words (like 'router', 'osi', 'database') as they are
                translated_words.append(w)
                
        pseudo_english = " ".join(translated_words)
        print(f"[SinglishAnalyzer] Mapped Translation: {pseudo_english}")
        
        # 2. LLM Prompting (Asking AI to fix the grammar and answer)
        prompt_template = """You are an AI teacher. A student asked a question in broken English mixed with technical terms. 
Try to understand the core technical question and give a short, clear educational answer.

Student's broken question: {question}

Clear Answer:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        # 3. Generate Answer
        answer = chain.invoke({"question": pseudo_english}).strip()
        
        return {
            "mapped_english": pseudo_english,
            "ai_answer": answer
        }