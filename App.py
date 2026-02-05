import streamlit as st
import tweepy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from transformers import pipeline
import time
import torch

# Initialize NLP tools
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)

# Clean text function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word not in stop_words and len(word) > 2])

# Extract topics using LDA
def extract_topics(texts, num_topics=5):
    if not texts:
        return ["No topics found"]
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    dtm = vectorizer.fit_transform(texts)
    if dtm.shape[1] < 2:  # Not enough features
        return ["Insufficient data for topics"]
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
        topics.append(f"Topic {idx+1}: {' '.join(top_words)}")
    return topics

# Extract actionable insights and to-dos
def extract_insights(tweets_df):
    insights = []
    to_dos = []
    for _, row in tweets_df.iterrows():
        text = row['text']
        # Summarize if long
        if len(text) > 100:
            summary = summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        else:
            summary = text
        insights.append(f"From @{row['author']}: {summary}")
        # Detect to-dos (simple keyword-based)
        if any(word in text.lower() for word in ['do', 'action', 'reply', 'check', 'todo', 'task', 'reminder']):
            to_dos.append(f"Action: {text[:100]}...")
    return insights[:10], to_dos[:5]  # Limit to top ones

# Fetch timeline
@st.cache_data(ttl=60)  # Cache for 60s
def fetch_timeline(client, user_id, max_results=100):
    try:
        response = client.get_home_timeline(max_results=max_results, tweet_fields=['text', 'created_at', 'public_metrics'], user_fields=['username'], expansions=['author_id'])
        tweets = []
        users = {u['id']: u['username'] for u in response.includes.get('users', [])}
        for tweet in response.data:
            tweets.append({
                'text': tweet.text,
                'author': users.get(tweet.author_id, 'Unknown'),
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count'],
                'created_at': tweet.created_at
            })
        return pd.DataFrame(tweets)
    except Exception as e:
        st.error(f"Error fetching timeline: {str(e)}")
        return pd.DataFrame()

# Main app
st.title("Your X Feed Dashboard")
st.markdown("Distills your feed into key topics, critical info, insights, and actions. Updates every 60 seconds.")

# Input keys (securely in session)
if 'client' not in st.session_state:
    with st.sidebar:
        st.header("API Setup")
        consumer_key = st.text_input("Consumer Key")
        consumer_secret = st.text_input("Consumer Secret", type="password")
        access_token = st.text_input("Access Token")
        access_secret = st.text_input("Access Secret", type="password")
        user_id = st.text_input("Your User ID")
        if st.button("Authenticate"):
            try:
                client = tweepy.Client(
                    consumer_key=consumer_key,
                    consumer_secret=consumer_secret,
                    access_token=access_token,
                    access_token_secret=access_secret
                )
                st.session_state.client = client
                st.session_state.user_id = user_id
                st.success("Authenticated!")
            except Exception as e:
                st.error(f"Auth failed: {str(e)}")

if 'client' in st.session_state:
    client = st.session_state.client
    user_id = st.session_state.user_id

    # Auto-refresh
    refresh_interval = 60
    placeholder = st.empty()

    while True:
        with placeholder.container():
            df = fetch_timeline(client, user_id)
            if df.empty:
                st.warning("No tweets fetched. Check auth or feed.")
            else:
                cleaned_texts = df['text'].apply(clean_text).tolist()

                # Key Topics
                st.header("Key Topics")
                topics = extract_topics(cleaned_texts)
                for topic in topics:
                    st.write(f"- {topic}")

                # Critical Things (high engagement)
                st.header("Critical Things to Know")
                critical_df = df.sort_values(by=['likes', 'retweets'], ascending=False).head(5)
                for _, row in critical_df.iterrows():
                    st.write(f"**@{row['author']} ({row['likes']} likes, {row['retweets']} RTs):** {row['text'][:200]}...")

                # Actionable Insights and To-Dos
                insights, to_dos = extract_insights(df)
                st.header("Actionable Insights")
                for insight in insights:
                    st.write(f"- {insight}")

                st.header("Things to Do")
                if to_dos:
                    for todo in to_dos:
                        st.write(f"- {todo}")
                else:
                    st.write("No explicit actions detected.")

        time.sleep(refresh_interval)
        st.experimental_rerun()  # Rerun to update
else:
