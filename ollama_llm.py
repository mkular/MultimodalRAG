from ollama import Client

class llm():

    def __init__(self, multimodal_result, search_query):
       
        self.client = Client(host='http://localhost:11434')
        self.multimodal_result = multimodal_result
        self.query = search_query

    def format_results(self):
        """Format search results for LLM consumption."""
        formatted_results = []
        
        for modality in self.multimodal_result:
            formatted_results.append(f"\n{modality['name'].upper()} RESULTS:")
            for item in modality['hits']:
                formatted_results.append(
                    f"Score: {item.score:.3f}"
                    f"\n   Path: {item.payload['file_path']}"
                )
        
        
        return "\n".join(formatted_results)

    def generate_result(self, formatted_results):
        response = self.client.chat(model='llava:latest', messages=[
        {
            'role': 'system',
            'content' : 'You are a helpful assistant that provides information based on multimodal search results.',
            'role': 'user',
            'content': f"""Query: {self.query}

                        {formatted_results}
                        
                        Please provide a description of the images.
                        Response:""",
        },
        ])
        return response['message']['content']