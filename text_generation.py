import google.generativeai as genai

class TextGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_response(self, query):
        response = self.model.generate_content(query, generation_config=genai.types.GenerationConfig(
            stop_sequences=["."],
            max_output_tokens=60,
            temperature=0.6,
        ))
        return response.text