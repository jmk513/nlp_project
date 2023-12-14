from openai import OpenAI
from dotenv import load_dotenv
import os

def generate_sentences_from_dict(data):
    # Load environment variables from .env file
    load_dotenv()

    # Create an OpenAI client using the API key from .env file
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    sentences = []

    # Iterate over each key-value pair in the dictionary
    key = list(data.keys())[0]
    values = list(data.values())[0]
    for value in values:
        # Form the prompt emphasizing variety and creativity
        system_prompt = "Craft a educative sentence using given words. The sentence should not be too long, and should not be significantly difficult to understand than words. Provide only one sentence, without any explanation."
        prompt = f"'{key}' and '{value}'"

        # Call the GPT-4 API
        response = client.chat.completions.create(
          model="gpt-4",  # Use the appropriate engine here
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
          ],
          max_tokens=120,
          temperature=0.7  # Adjust for more creative responses
        )

        # Extract and print the generated sentence
        generated_sentence = response.choices[0].message.content
        print(f"Key: {key}, Value: {value}, Sentence: {generated_sentence}")

        sentences.append(generated_sentence)

    return sentences
 


if __name__ == "__main__" :
    data = [{'강아지': ['멍멍', '고양이', '귀엽다', '사다', '신발']},
            {'신발': ['신다', '양말', '작다', '운동화', '공']},
            {'자전거': ['따르릉', '콰르릉', '넘어지다', '부릉부릉', '빠르다']},
            {'버스': ['기다리다', '내리다', '자전거', '넘어지다', '따르릉']},
            {'가방': ['가볍다', '무겁다', '열다', '넣다', '닫다']},
            {'나비': ['날다', '풀', '활짝', '꽃', '인사하다']}]
    
    for d in data:
        generate_sentences_from_dict(d)
