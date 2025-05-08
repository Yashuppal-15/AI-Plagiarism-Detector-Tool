import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence
from nltk.probability import FreqDist
from collections import Counter
import plotly.express as px
from nltk.corpus import stopwords
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def calculate_perplexity(text):
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    perplexity = torch.exp(
        torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1)
        )
    )
    return perplexity.item()

def calculate_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score

def plot_top_repeated_words(text):
    # Tokenize the text and remove stopwords and punctuation
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # Count the occurrence of each word
    word_counts = Counter(tokens)

    # Get the top 10 most repeated words
    top_words = word_counts.most_common(10)

    if not top_words:
        st.info("No repeated words found to display.")
        return

    # Extract the words and their counts for plotting
    words, counts = zip(*top_words)

    # Plot the bar chart using Plotly
    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title='Top 10 Most Repeated Words')
    st.plotly_chart(fig, use_container_width=True)

# Streamlit page settings
st.set_page_config(layout="wide", page_title="GPT Shield", page_icon="🛡️")

# App Title
st.title("🛡️ GPT Shield: AI Plagiarism Detector")

# User Input
text_area = st.text_area("Enter text to analyze:", "", height=200)

if text_area.strip():
    if st.button("Analyze"):
        with st.spinner('Analyzing the text...'):
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.subheader("📄 Your Input Text")
                st.success(text_area)

            with col2:
                st.subheader("🔎 Detection Score")
                perplexity = calculate_perplexity(text_area)
                burstiness_score = calculate_burstiness(text_area)

                st.write(f"**Perplexity:** `{perplexity:.2f}`")
                st.write(f"**Burstiness Score:** `{burstiness_score:.2f}`")

                if perplexity > 30000 and burstiness_score < 0.2:
                    st.error("🚨 **Text Analysis Result: AI generated content detected!**")
                else:
                    st.success("✅ **Text Analysis Result: Likely human-written content**")

                st.caption("⚠️ *Disclaimer: AI detectors are useful, but they are not perfect. Always combine automatic checks with human judgment for important decisions.*")

            with col3:
                st.subheader("📊 Text Insights")
                plot_top_repeated_words(text_area)

else:
    st.info("💬 Please enter some text above to start the analysis.")
