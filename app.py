import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load your models and data once at the top
df = pd.read_csv('data/data.csv', delimiter='\t')
models = joblib.load('career_single_feature_ensemble.pkl')

feature_list = ['E_score', 'N_score', 'C_score', 'A_score', 'O_score']

# Define questions and their mapping to traits
questions = [
    # Extraversion
    ("E1", "I am the life of the party."),
    ("E2", "I don't talk a lot."),
    ("E3", "I feel comfortable around people."),
    ("E4", "I keep in the background."),
    ("E5", "I start conversations."),
    ("E6", "I have little to say."),
    ("E7", "I talk to a lot of different people at parties."),
    ("E8", "I don't like to draw attention to myself."),
    ("E9", "I don't mind being the center of attention."),
    ("E10", "I am quiet around strangers."),
    # Neuroticism
    ("N1", "I get stressed out easily."),
    ("N2", "I am relaxed most of the time."),
    ("N3", "I worry about things."),
    ("N4", "I seldom feel blue."),
    ("N5", "I am easily disturbed."),
    ("N6", "I get upset easily."),
    ("N7", "I change my mood a lot."),
    ("N8", "I have frequent mood swings."),
    ("N9", "I get irritated easily."),
    ("N10", "I often feel blue."),
    # Agreeableness
    ("A1", "I feel little concern for others."),
    ("A2", "I am interested in people."),
    ("A3", "I insult people."),
    ("A4", "I sympathize with others' feelings."),
    ("A5", "I am not interested in other people's problems."),
    ("A6", "I have a soft heart."),
    ("A7", "I am not really interested in others."),
    ("A8", "I take time out for others."),
    ("A9", "I feel others' emotions."),
    ("A10", "I make people feel at ease."),
    # Conscientiousness
    ("C1", "I am always prepared."),
    ("C2", "I leave my belongings around."),
    ("C3", "I pay attention to details."),
    ("C4", "I make a mess of things."),
    ("C5", "I get chores done right away."),
    ("C6", "I often forget to put things back in their proper place."),
    ("C7", "I like order."),
    ("C8", "I shirk my duties."),
    ("C9", "I follow a schedule."),
    ("C10", "I am exacting in my work."),
    # Openness
    ("O1", "I have a rich vocabulary."),
    ("O2", "I have difficulty understanding abstract ideas."),
    ("O3", "I have a vivid imagination."),
    ("O4", "I am not interested in abstract ideas."),
    ("O5", "I have excellent ideas."),
    ("O6", "I do not have a good imagination."),
    ("O7", "I am quick to understand things."),
    ("O8", "I use difficult words."),
    ("O9", "I spend time reflecting on things."),
    ("O10", "I am full of ideas."),
]

# Reverse-scored questions (where 1 = Agree, 5 = Disagree)
reverse_scored = {
    "E2", "E4", "E6", "E8", "E10",
    "N2", "N4",
    "A1", "A3", "A5", "A7",
    "C2", "C4", "C6", "C8",
    "O2", "O4", "O6"
}

reverse_scored = {
    "E2", "E4", "E6", "E8", "E10",
    "N2", "N4",
    "A1", "A3", "A5", "A7",
    "C2", "C4", "C6", "C8",
    "O2", "O4", "O6"
}

def rescale(score, old_min=10, old_max=50, new_min=2.67, new_max=9.45):
    return ((score - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

st.title("Career Prediction Based on Personality (Big Five)")
st.write("Please answer the following 50 questions. For each, select:")
st.write("**1 = Disagree, 3 = Neutral, 5 = Agree**")

# Collect responses with sliders, saved in session state to persist
responses = []
for code, text in questions:
    key = f"slider_{code}"
    default_val = 3
    if key not in st.session_state:
        st.session_state[key] = default_val
    value = st.slider(f"{code}: {text}", 1, 5, st.session_state[key], key=key)
    if code in reverse_scored:
        value = 6 - value
    responses.append((code, value))

def predict_career(responses):
    trait_scores = {
        'E_score': rescale(sum(v for (c, v) in responses if c.startswith('E'))),
        'N_score': rescale(sum(v for (c, v) in responses if c.startswith('N'))),
        'C_score': rescale(sum(v for (c, v) in responses if c.startswith('C'))),
        'A_score': rescale(sum(v for (c, v) in responses if c.startswith('A'))),
        'O_score': rescale(sum(v for (c, v) in responses if c.startswith('O'))),
    }
    sample_input = [[trait_scores[f] for f in feature_list]]
    probas = []
    for i, feature in enumerate(feature_list):
        proba = models[feature].predict_proba([[sample_input[0][i]]])
        probas.append(proba)
    combined_proba = np.mean(probas, axis=0)
    final_pred = np.argmax(combined_proba, axis=1)
    class_names = models[feature_list[0]].classes_
    predicted_label = class_names[final_pred[0]]
    return predicted_label, trait_scores

if st.button("Predict Career"):
    predicted_label, trait_scores = predict_career(responses)
    st.success(f"**Predicted Career:** {predicted_label}")
    
