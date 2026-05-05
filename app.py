import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartEdu Recommender", page_icon="📚", layout="centered")

st.title("📚 SmartEdu Recommender")
st.markdown("**AI-Powered Study Material & Career Path Recommender**")

st.write("### Helping students discover the best learning paths")

# Dataset
data = {
    'topic': ['Python Programming', 'Machine Learning Basics', 'Deep Learning', 'Data Science', 'Web Development', 'React.js', 'Newton Laws of Motion', 'Gravity and Acceleration', 'Quadratic Equations', 'Chemical Bonding', 'Photosynthesis', 'English Grammar', 'Artificial Intelligence', 'Computer Networks', 'Cybersecurity'],
    'description': ['Learn Python from basics to advanced with real projects', 'Introduction to ML concepts and algorithms', 'Neural networks, CNN and advanced deep learning', 'Data analysis, visualization and statistics', 'Build modern responsive websites', 'Frontend development with React and hooks', 'Classical mechanics and laws of motion', 'Gravitational force and acceleration', 'Solving quadratic equations and word problems', 'Ionic, covalent bonding and reactions', 'Photosynthesis process in plants', 'Grammar rules, tenses and writing skills', 'AI concepts, agents and real-world applications', 'Networking fundamentals and protocols', 'Ethical hacking and network security'],
    'category': ['Computer Science', 'Computer Science', 'Computer Science', 'Computer Science', 'Computer Science', 'Computer Science', 'Physics', 'Physics', 'Math', 'Chemistry', 'Chemistry', 'English', 'Computer Science', 'Computer Science', 'Computer Science']
}

df = pd.DataFrame(data)

# Train model
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['description'] + " " + df['topic'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_topics(input_topic, top_n=5):
    mask = df['topic'].str.contains(input_topic, case=False) | df['description'].str.contains(input_topic, case=False)
    idx_list = df[mask].index
    if len(idx_list) == 0:
        return None
    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [(df['topic'].iloc[i], df['category'].iloc[i], round(score*100, 1)) for i, score in sim_scores]

topic = st.text_input("Enter a topic you want to study (e.g., Python, Physics, Chemistry, AI, Math)", 
                     placeholder="Python")

if st.button("Get Personalized Recommendations", type="primary"):
    if topic:
        results = recommend_topics(topic)
        if results:
            st.success(f"**Best recommendations for '{topic}'**")
            for rec in results:
                st.write(f"**{rec[0]}** — *{rec[1]}* — **{rec[2]}%** similar")
        else:
            st.error("Topic not found. Try: Python, Physics, AI, Chemistry, Math, etc.")
    else:
        st.warning("Please enter a topic")

st.markdown("---")
st.caption("**Project 3** | SmartEdu Recommender | Made by Vihaan | Aiming to help underprivileged students discover better learning resources")