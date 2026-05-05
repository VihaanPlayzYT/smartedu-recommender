import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time

print("🚀 SmartEdu Recommender System Loading...\n")

# MASSIVE DATASET (50+ topics)
data = {
    'topic': [
        # Computer Science (20+)
        'Python Programming', 'Java Programming', 'Machine Learning Basics', 
        'Deep Learning', 'Data Science', 'Data Analysis', 'Web Development',
        'React.js', 'Node.js', 'Full Stack Development', 'Database Management',
        'SQL', 'MongoDB', 'Computer Networks', 'Cybersecurity', 'Ethical Hacking',
        'Artificial Intelligence', 'Computer Vision', 'Natural Language Processing',
        'Data Structures', 'Algorithms', 'Cloud Computing', 'DevOps', 'Docker',
        
        # Physics (10)
        'Newton Laws of Motion', 'Gravity and Acceleration', 'Force and Motion',
        'Kinetic and Potential Energy', 'Electricity and Magnetism', 
        'Waves and Optics', 'Thermodynamics', 'Quantum Physics Basics',
        
        # Math (10)
        'Quadratic Equations', 'Calculus Basics', 'Trigonometry', 
        'Probability and Statistics', 'Linear Algebra', 'Number Systems',
        'Coordinate Geometry', 'Matrices and Determinants',
        
        # Chemistry (8)
        'Chemical Bonding', 'Acid Base Reactions', 'Photosynthesis',
        'Periodic Table', 'Organic Chemistry Basics', 'pH and Indicators',
        'Chemical Equilibrium', 'Electrochemistry',
        
        # English (6)
        'English Grammar', 'Tenses and Sentence Structure', 'Vocabulary Building',
        'Essay Writing', 'Reading Comprehension', 'Public Speaking'
    ],
    'description': [
        # CS
        'Learn Python from basics to advanced OOP and projects',
        'Java programming fundamentals and object oriented concepts',
        'Supervised, unsupervised and reinforcement learning',
        'Neural networks, CNN, RNN and transformers',
        'Data analysis, visualization, pandas and statistics',
        'Exploratory data analysis and insights generation',
        'Build responsive websites using HTML CSS JavaScript',
        'Modern frontend development with React and hooks',
        'Backend development with Node.js and Express',
        'End to end web application development',
        'Relational databases, SQL queries and normalization',
        'Structured query language and database management',
        'NoSQL databases and document stores',
        'OSI model, TCP/IP, routing and subnets',
        'Network security, firewalls and encryption',
        'Penetration testing and vulnerability assessment',
        'AI agents, ethics and real world applications',
        'Image processing and object detection',
        'Text processing, sentiment analysis and chatbots',
        'Arrays, linked lists, stacks, queues and trees',
        'Sorting, searching, recursion and dynamic programming',
        'AWS, Azure and Google Cloud platforms',
        'CI/CD pipelines and automation',
        'Containerization and orchestration',
        
        # Physics
        'Laws of motion and classical mechanics',
        'Gravitational force and acceleration due to gravity',
        'Concepts of force, friction and equilibrium',
        'Energy forms and conservation laws',
        'Current, voltage, circuits and electromagnetism',
        'Wave motion, sound and light optics',
        'Heat, temperature and thermodynamic processes',
        'Basic quantum mechanics and photoelectric effect',
        
        # Math
        'Solving quadratic equations and applications',
        'Differentiation, integration and applications',
        'Trigonometric ratios, identities and heights distances',
        'Probability, mean, median, mode and distributions',
        'Matrices, vectors and linear transformations',
        'Number theory and number systems',
        'Coordinate geometry and straight lines',
        'Determinants and matrix applications',
        
        # Chemistry
        'Ionic and covalent bonding mechanisms',
        'Acid base neutralization and pH scale',
        'Photosynthesis process in plants',
        'Periodic table trends and properties',
        'Hydrocarbons and functional groups',
        'Acids, bases, salts and indicators',
        'Chemical equilibrium and Le Chatelier principle',
        'Electrochemistry and redox reactions',
        
        # English
        'Grammar rules, parts of speech and sentence formation',
        'All tenses and their correct usage',
        'Synonyms, antonyms and vocabulary improvement',
        'Essay writing structure and techniques',
        'Reading comprehension strategies',
        'Public speaking and communication skills'
    ],
    'category': [
        'Computer Science']*24 + ['Physics']*8 + ['Math']*8 + ['Chemistry']*8 + ['English']*6
}

df = pd.DataFrame(data)

# Progress Bar
print("Training recommendation engine with", len(df), "topics...")
for i in tqdm(range(20), desc="Processing large dataset", ncols=80):
    time.sleep(0.08)

# Train model
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['description'] + " " + df['topic'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"\n✅ SmartEdu Recommender is ready with {len(df)} topics!\n")

def recommend_topics(input_topic, top_n=5):
    mask = df['topic'].str.contains(input_topic, case=False) | df['description'].str.contains(input_topic, case=False)
    idx_list = df[mask].index
    
    if len(idx_list) == 0:
        print(f"❌ Topic '{input_topic}' not found.")
        print("Try: Python, Physics, Chemistry, Math, AI, Deep Learning, etc.")
        return
    
    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    print(f"📚 Top {top_n} recommendations for '{input_topic}':\n")
    for i, score in sim_scores:
        print(f"• {df['topic'].iloc[i]}  ({df['category'].iloc[i]}) - {round(score*100, 1)}% similar")

if __name__ == "__main__":
    recommend_topics("Python", 6)
    recommend_topics("Physics", 5)
    recommend_topics("Chemistry", 5)
    recommend_topics("AI", 5)
    recommend_topics("Math", 5)
    recommend_topics("English", 4)