from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

class SEOGenerator:
    """
    A professional module to generate SEO-optimized content and social media posts 
    for the 'Exam Hub' brand using LLMs.
    """
    def __init__(self, llm_model="google/flan-t5-small"):
        print(f"[SEOGenerator] Loading LLM: {llm_model}...")
        hf_pipe = pipeline("text2text-generation", model=llm_model, max_length=100)
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        print("[SEOGenerator] Model loaded.")

    def generate_all(self, text_content):
        """
        Takes the transcript and returns a dictionary with Title, Tags, and Facebook Post.
        """
        print("[SEOGenerator] Generating content pipeline...")
        
        # 1. Generate YouTube Title
        title_prompt = PromptTemplate.from_template("Write a short, professional video title for an educational channel based on this text: {text}\nTitle:")
        title_chain = title_prompt | self.llm
        title = title_chain.invoke({"text": text_content}).strip()
        
        # 2. Generate SEO Keywords (Tags)
        tags_prompt = PromptTemplate.from_template("Extract 3 technical keywords from this text. Separate them with commas: {text}\nKeywords:")
        tags_chain = tags_prompt | self.llm
        tags = tags_chain.invoke({"text": text_content}).strip()
        
        # 3. Generate Facebook Post
        fb_prompt = PromptTemplate.from_template("Write a one sentence Facebook post announcing a new lesson based on this text: {text}\nPost:")
        fb_chain = fb_prompt | self.llm
        fb_post = fb_chain.invoke({"text": text_content}).strip()
        
        print("[SEOGenerator] Generation complete.")
        
        return {
            "youtube_title": title,
            "seo_tags": tags,
            "facebook_post": fb_post
        }

# ==========================================
# Module Testing Block
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    sample_text = "Welcome to Exam Hub. Today we will focus on Data Warehousing and Computer Networks for the 2026 syllabus."
    
    seo = SEOGenerator()
    results = seo.generate_all(sample_text)
    
    print("\n--- SEO OUTPUTS ---")
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("-------------------\n")