# skill-dynamics
Turkey labor Market Index
import transformers
import torch

# Load the pre-trained GPT-2 model
model = transformers.GPT2Model.from_pretrained('gpt2')

# Load the tokenizer for the GPT-2 model
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

# Preprocess the CV text
def preprocess(text):
    # Remove any formatting information
    text = text.strip()

    # Convert the text to lowercase
    text = text.lower()

    # Replace any non-alphanumeric characters with spaces
    text = re.sub(r'\W+', ' ', text)

    return text

# Tokenize the CV text
def tokenize(text):
    # Encode the text using the GPT-2 tokenizer
    encoded = tokenizer.encode(text)

    # Convert the encoded sequence to a PyTorch tensor
    input_ids = torch.tensor([encoded])

    return input_ids

# Extract the skills from the CV
def extract_skills(text):
    # Preprocess the text
    text = preprocess(text)

    # Tokenize the text
    input_ids = tokenize(text)

    # Generate the outputs from the GPT-2 model
    outputs = model.generate(input_ids)

    # Decode the outputs into text
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split the decoded text into sentences
    sentences = decoded.split('. ')

    # Find the sentences that mention skills
    skills = []
    for sentence in sentences:
        if 'skill' in sentence or 'expertise' in sentence:
            skills.append(sentence)

    return skills

# Rank the skills
def rank_skills(skills):
    # Define a list of important keywords
    keywords = ['programming', 'project management', 'data analysis', 'communication', 'leadership']

    # Rank the skills based on keyword frequency
    ranking = {skill: sum([1 for keyword in keywords if keyword in skill.lower()]) for skill in skills}
    ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    return ranking

# Example usage
cv_text = 'I have expertise in project management and communication. I am also skilled in programming languages such as Python and Java.'
skills = extract_skills(cv_text)
ranking = rank_skills(skills)

print('Extracted skills:')
for skill in skills:
    print('-', skill)

print('Ranked skills:')
for skill, score in ranking:
    print('-', skill, f'(score: {score})')
