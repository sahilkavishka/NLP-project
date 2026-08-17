import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

class PredictivePaperGenerator:
    """
    Advanced module that uses Linear Regression to forecast exam trends, 
    and an LLM to generate the actual questions for the predicted topics.
    """
    def __init__(self, llm_model="google/flan-t5-small"):
        print("[PredictivePaper] Loading LLM for Question Generation...")
        # Using text-generation to generate new exam questions
        hf_pipe = pipeline("text-generation", model=llm_model, max_new_tokens=150)
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        print("[PredictivePaper] Models Loaded.")

    def analyze_and_generate(self, subject_name, target_year=2027):
        print(f"[PredictivePaper] Analyzing past papers for {subject_name}...")
        
        # Simulated Historical Data (Frequency of topics in past papers)
        # In a fully deployed app, this would be scraped from actual PDF past papers.
        data = {
            "Year": [2022, 2023, 2024, 2025, 2026],
            "Data Mining Algorithms": [2, 3, 4, 5, 7],         # Increasing trend
            "Cloud Data Warehousing": [0, 1, 3, 5, 8],         # Huge increase
            "Legacy SQL Systems": [8, 6, 5, 3, 2],             # Decreasing trend
            "Real-time Analytics": [1, 2, 4, 6, 9],            # Highly trending
            "Basic Network Topologies": [5, 5, 4, 4, 3]        # Slowly dropping
        }
        df = pd.DataFrame(data)
        X = df[['Year']]
        
        predictions = []
        
        # 1. Machine Learning: Predict the weight of each topic for the target year
        for topic in df.columns[1:]:
            y = df[topic]
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict([[target_year]])[0]
            predictions.append({"topic": topic, "expected_weight": round(pred, 1)})
        
        # 2. Sort to find the Top 3 most probable topics
        top_topics = sorted(predictions, key=lambda x: x['expected_weight'], reverse=True)[:3]
        
        # 3. Generative AI: Create brand new exam questions for these top topics
        print("[PredictivePaper] Generating new questions for the predicted topics...")
        mock_paper = []
        
        prompt_template = """You are an expert university professor setting a challenging final exam for {subject}.
Write a degree-level, analytical exam question about: {topic}.

Exam Question:"""
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        for i, item in enumerate(top_topics):
            topic_clean = item['topic'].replace('_', ' ')
            # Generate the question using LLM
            q_text = chain.invoke({"subject": subject_name, "topic": topic_clean}).strip()
            
            mock_paper.append({
                "number": i + 1,
                "topic": topic_clean,
                "probability": min(99.9, round((item['expected_weight'] / 10) * 100, 1)), # Convert weight to %
                "question": q_text
            })
            
        return mock_paper