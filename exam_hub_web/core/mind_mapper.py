import spacy

class MindMapper:
    """
    NLP module to extract concepts and relationships from a text 
    and format them into Mermaid.js syntax for visual mind maps.
    """
    def __init__(self):
        print("[MindMapper] Loading SpaCy English Model...")
        self.nlp = spacy.load("en_core_web_sm")
        print("[MindMapper] Model loaded.")

    def generate_mermaid_syntax(self, text):
        """
        Parses text and generates a Mermaid Graph syntax string.
        """
        doc = self.nlp(text)
        
        # We will use Mermaid.js flowchart syntax
        mermaid_code = "graph TD;\n"
        
        # Keep track of added nodes to avoid duplicates
        added_edges = set()
        
        for sent in doc.sents:
            subject = None
            obj = None
            relation = None
            
            # Simple extraction: Find Subject, Verb (Relation), and Object in a sentence
            for token in sent:
                if "subj" in token.dep_:
                    subject = token.text.strip().replace(" ", "_")
                if "obj" in token.dep_:
                    obj = token.text.strip().replace(" ", "_")
                if token.pos_ == "VERB":
                    relation = token.lemma_
                    
            if subject and obj and relation:
                # Create a relationship line: Subject -->|relation| Object
                edge = f"    {subject} -->|{relation}| {obj};\n"
                if edge not in added_edges:
                    mermaid_code += edge
                    added_edges.add(edge)
                    
        # Fallback if the text was too simple or couldn't find subject-object relations
        if len(added_edges) == 0:
            mermaid_code += "    Text --> No_Clear_Relations_Found;\n"
            
        return mermaid_code