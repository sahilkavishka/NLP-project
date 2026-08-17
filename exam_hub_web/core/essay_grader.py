import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer, util

class EssayGrader:
    """
    Advanced NLP module for Exam Hub Teachers.
    Evaluates students' descriptive answers against the teacher's marking scheme based on meaning (Semantic Similarity).
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"[EssayGrader] Loading Semantic Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("[EssayGrader] Model loaded successfully.")

    def grade_answer(self, student_answer, marking_scheme_points, threshold=0.55):
        """
        Compares a student's text against a list of required points.
        Returns a dictionary with the score and detailed analysis.
        """
        print("[EssayGrader] Analyzing semantic similarities...")
        
        student_embedding = self.model.encode(student_answer, convert_to_tensor=True)
        scheme_embeddings = self.model.encode(marking_scheme_points, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(student_embedding, scheme_embeddings)[0]
        
        results = []
        total_score = 0
        
        for i, score in enumerate(cosine_scores):
            match_score = score.item()
            is_covered = match_score >= threshold
            
            if is_covered:
                total_score += 1
                
            results.append({
                "point": marking_scheme_points[i],
                "match_score": round(match_score * 100, 2),
                "is_covered": is_covered
            })
        
        final_percentage = (total_score / len(marking_scheme_points)) * 100 if marking_scheme_points else 0
        
        return {
            "final_score": round(final_percentage, 2),
            "detailed_analysis": results
        }