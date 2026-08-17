import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer, util
import spacy

class ContentGapAnalyzer:
    """
    NLP module to find "Content Gaps" by comparing the official syllabus 
    against existing video titles using Semantic Similarity.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("[ContentGapAnalyzer] Loading NLP Models...")
        # Load Sentence Transformer for meaning comparison
        self.similarity_model = SentenceTransformer(model_name)
        # Load SpaCy to extract concepts/noun chunks
        self.nlp = spacy.load("en_core_web_sm")
        print("[ContentGapAnalyzer] Models loaded successfully.")

    def extract_concepts(self, text):
        """Extracts meaningful phrases (Noun Chunks) from a block of text."""
        doc = self.nlp(text)
        # Get chunks that have more than one word (e.g., "Data Warehousing", "Machine Learning")
        concepts = [chunk.text.strip().lower() for chunk in doc.noun_chunks if len(chunk.text.strip().split()) > 1]
        # Remove duplicates
        return list(set(concepts))

    def analyze_gap(self, syllabus_text, existing_content_text, threshold=0.60):
        """Compares required concepts vs existing content and finds the gaps."""
        print("[ContentGapAnalyzer] Analyzing Content Gaps...")
        
        syllabus_concepts = self.extract_concepts(syllabus_text)
        
        # If the existing content is empty, use a placeholder
        if not existing_content_text.strip():
            existing_content_text = "nothing"
            
        # We compare syllabus concepts directly against the whole existing content
        syl_emb = self.similarity_model.encode(syllabus_concepts, convert_to_tensor=True)
        # Split existing content by lines for better matching
        existing_lines = [line.strip() for line in existing_content_text.split('\n') if line.strip()]
        if not existing_lines:
            existing_lines = ["nothing"]
            
        ex_emb = self.similarity_model.encode(existing_lines, convert_to_tensor=True)
        
        # Calculate Cosine Similarity
        cosine_scores = util.cos_sim(syl_emb, ex_emb)
        
        gaps = []
        covered = []
        
        for i, syl_concept in enumerate(syllabus_concepts):
            max_score = cosine_scores[i].max().item()
            
            # If the best match score is below the threshold, it's a GAP!
            if max_score < threshold:
                gaps.append({
                    "topic": syl_concept.title(),
                    "opportunity_score": round((1 - max_score) * 100, 1) # Higher score = Bigger Gap
                })
            else:
                covered.append({
                    "topic": syl_concept.title(),
                    "match_score": round(max_score * 100, 1)
                })
        
        # Sort gaps so the biggest opportunity comes first
        gaps = sorted(gaps, key=lambda x: x['opportunity_score'], reverse=True)
        
        gap_percentage = round((len(gaps) / len(syllabus_concepts)) * 100, 1) if syllabus_concepts else 0
        
        return {
            "gaps": gaps,
            "covered": covered,
            "gap_percentage": gap_percentage,
            "total_concepts": len(syllabus_concepts)
        }