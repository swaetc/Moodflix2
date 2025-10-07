import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import zipfile
from recommender_engine import MoodBasedRecommender
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MoodFlix - Your Mood-Based Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #E50914, #F5F5F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .mood-btn {
        padding: 1.5rem;
        margin: 0.5rem;
        border-radius: 15px;
        border: 2px solid #E50914;
        background: white;
        color: #E50914;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        text-align: center;
    }
    .mood-btn:hover {
        background: #E50914;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(229, 9, 20, 0.3);
    }
    .mood-btn.selected {
        background: #E50914;
        color: white;
        transform: scale(1.02);
    }
    .rec-card {
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #E50914;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
        color: #000000 !important;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .rec-card h3 {
        color: #E50914 !important;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class MoodFlixApp:
    def __init__(self):
        self.recommender = None
        self.initialize_recommender()

    def initialize_recommender(self):
        """Initialize the recommendation engine"""
        try:
            self.recommender = MoodBasedRecommender()
            
            # Try to load pre-trained model first
            if os.path.exists('trained_model.pkl'):
                if self.recommender.load_model('trained_model.pkl'):
                    st.success("✅ Pre-trained model loaded successfully!")
                else:
                    st.warning("⚠️ Could not load pre-trained model. Please train the model first.")
            else:
                st.info("ℹ️ No pre-trained model found. Please train the model with your Netflix data.")
                
        except Exception as e:
            st.error(f"❌ Error initializing recommender: {e}")

    def download_kaggle_dataset_windows(self):
        """Download Netflix dataset using Kaggle CLI"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            st.info("📥 Downloading Netflix dataset from Kaggle...")
            
            # Method 1: Using kaggle CLI
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', 'shivamb/netflix-shows'], 
                capture_output=True, 
                text=True, 
                shell=True
            )
            
            if result.returncode == 0:
                st.success("✅ Dataset downloaded successfully!")
                
                # Extract the zip file
                if os.path.exists('netflix-shows.zip'):
                    with zipfile.ZipFile('netflix-shows.zip', 'r') as zip_ref:
                        zip_ref.extractall('data/')
                    
                    # Find the CSV file
                    csv_file = None
                    for file in os.listdir('data/'):
                        if file.endswith('.csv'):
                            csv_file = file
                            break
                    
                    if csv_file:
                        dataset_path = f'data/{csv_file}'
                        
                        # Rename to standard name for consistency
                        standard_path = 'data/netflix_titles.csv'
                        if dataset_path != standard_path:
                            os.rename(dataset_path, standard_path)
                        
                        st.session_state.dataset_loaded = True
                        st.session_state.dataset_path = standard_path
                        st.success("✅ Netflix dataset is now ready for training!")
                        
                        # Clean up zip file
                        if os.path.exists('netflix-shows.zip'):
                            os.remove('netflix-shows.zip')
                        
                        return True
                    else:
                        st.error("❌ No CSV file found in downloaded dataset")
                        return False
                else:
                    st.error("❌ Downloaded zip file not found")
                    return False
            else:
                st.error(f"❌ Kaggle download failed: {result.stderr}")
                
                # Provide helpful error messages
                if "403" in result.stderr:
                    st.error("""
                    **Kaggle API Error 403 - Forbidden**
                    Possible reasons:
                    - You haven't accepted the dataset rules on Kaggle
                    - Your API key is invalid or expired
                    - You need to authenticate first
                    
                    **Solution:**
                    1. Go to: https://www.kaggle.com/shivamb/netflix-shows
                    2. Click on "⋮" (three dots) → "Copy API command"
                    3. Accept dataset rules if prompted
                    4. Ensure your kaggle.json credentials are correct
                    """)
                elif "404" in result.stderr:
                    st.error("❌ Dataset not found. The dataset might have been removed or renamed.")
                elif "command not found" in result.stderr.lower():
                    st.error("""
                    **Kaggle CLI not found**
                    Solution:
                    1. Install kaggle: `pip install kaggle`
                    2. Set up API credentials in ~/.kaggle/kaggle.json
                    3. For Streamlit Cloud, add KAGGLE_USERNAME and KAGGLE_KEY to secrets
                    """)
                
                return False
                
        except Exception as e:
            st.error(f"❌ Error downloading dataset: {e}")
            return False

    def load_preset_dataset(self):
        """Load preset dataset with Kaggle CLI fallback"""
        try:
            # For Streamlit Cloud, use environment variables
            if 'KAGGLE_USERNAME' in st.secrets and 'KAGGLE_KEY' in st.secrets:
                os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
                os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']
                st.success("✅ Using Kaggle credentials from Streamlit secrets")
            
            return self.download_kaggle_dataset_windows()
            
        except Exception as e:
            st.error(f"❌ Error in load_preset_dataset: {e}")
            return False

    def render_header(self):
        """Render the application header"""
        st.markdown('<div class="main-header">🎬 MoodFlix</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
            <h3>Your AI-powered mood-based movie and TV show recommender 🍿</h3>
            <p>Tell us how you're feeling, and we'll find the perfect content for your mood!</p>
        </div>
        """, unsafe_allow_html=True)

    def render_training_section(self):
        """Render model training section with preset dataset option"""
        with st.sidebar.expander("🔧 Model Training", expanded=not self.recommender.is_trained):
            st.write("Train the recommendation model with your Netflix data")
            
            # Preset dataset button
            st.markdown("---")
            st.subheader("📥 Quick Start")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎬 Download via Kaggle", 
                           help="Download Netflix dataset directly from Kaggle using Kaggle CLI",
                           use_container_width=True):
                    if self.load_preset_dataset():
                        st.rerun()
            
            with col2:
                if st.button("🔄 Train with Dataset", 
                           help="Train model using the downloaded dataset",
                           use_container_width=True,
                           disabled=not hasattr(st.session_state, 'dataset_loaded')):
                    try:
                        with st.spinner('Training model with Netflix dataset...'):
                            success = self.recommender.train(st.session_state.dataset_path)
                            if success:
                                self.recommender.save_model('trained_model.pkl')
                                st.success("✅ Model trained successfully with Netflix data!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")
            
            # Original file upload functionality
            st.markdown("---")
            st.subheader("📤 Custom Dataset")
            st.write("Or upload your own Netflix CSV file:")
            
            uploaded_file = st.file_uploader("Upload Netflix CSV", type=['csv'], key="file_uploader")
            
            if uploaded_file is not None:
                if st.button("🚀 Train with Uploaded File", use_container_width=True):
                    with st.spinner('Training AI model... This may take a few minutes.'):
                        try:
                            # Save uploaded file
                            os.makedirs('data', exist_ok=True)
                            with open('data/netflix_titles.csv', 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            
                            # Train model
                            success = self.recommender.train('data/netflix_titles.csv')
                            if success:
                                # Save trained model
                                self.recommender.save_model('trained_model.pkl')
                                st.success("🎉 Model trained and saved successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Model training failed.")
                                
                        except Exception as e:
                            st.error(f"Training error: {e}")
            
            # Help text
            st.markdown("---")
            st.info("""
            **Need Netflix data?** 
            - Use **Download via Kaggle** for automatic dataset download
            - Requires Kaggle CLI setup on local machine
            - For Streamlit Cloud: Add KAGGLE_USERNAME and KAGGLE_KEY to secrets
            - Or download manually from [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
            """)

    def render_mood_selection(self):
        """Render the mood selection interface"""
        st.header("🎯 Step 1: How are you feeling today?")
        
        if not self.recommender.is_trained:
            st.warning("⚠️ Please train the model first using the sidebar.")
            return None
        
        # Mood options with emojis
        mood_display_map = {
            'happy_upbeat': '😊 Happy & Upbeat',
            'sad_reflective': '😢 Reflective & Emotional', 
            'adventurous': '🚀 Adventurous & Excited',
            'relaxed': '😌 Relaxed & Chill',
            'thrilled': '🎯 Thrilled & On Edge',
            'intellectual': '🤔 Intellectual & Curious',
            'stressed': '😫 Stressed & Need Escape',
            'nostalgic': '📻 Nostalgic & Sentimental'
        }
        
        # Create mood buttons in a grid
        cols = st.columns(4)
        selected_mood = st.session_state.get('selected_mood', None)
        
        for i, (mood_key, mood_display) in enumerate(mood_display_map.items()):
            with cols[i % 4]:
                is_selected = (selected_mood == mood_key)
                button_label = f"**{mood_display}**"
                
                if st.button(
                    button_label,
                    key=f"mood_{mood_key}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    selected_mood = mood_key
                    st.session_state.selected_mood = selected_mood
                    st.session_state.selected_mood_name = mood_display
                    st.rerun()
                
                # Show description on hover (via tooltip)
                if mood_key in self.recommender.mood_genre_mapping:
                    with st.expander("ℹ️", expanded=False):
                        st.caption(self.recommender.mood_genre_mapping[mood_key]['description'])
        
        return selected_mood

    def render_preferences(self):
        """Render additional preference selectors"""
        st.header("🎛️ Step 2: Fine-tune your preferences")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            content_type = st.radio(
                "**Content Type**",
                ["Any", "Movies Only", "TV Shows Only"],
                help="Choose the type of content you prefer"
            )
            content_type_map = {
                "Any": None,
                "Movies Only": "Movie", 
                "TV Shows Only": "TV Show"
            }
        
        with col2:
            duration_preference = st.selectbox(
                "**Time Available**",
                ["Any", "Quick Watch (< 30 min)", "Short (30-90 min)", 
                 "Feature Length (90-180 min)", "Binge Session (> 180 min)"],
                help="How much time do you have to watch?"
            )
        
        with col3:
            era_preference = st.selectbox(
                "**Era Preference**",
                ["Any", "Recent (2010s+)", "Classic (Pre-2000s)", "2000s"],
                help="Preferred release era"
            )
            era_map = {
                "Any": None,
                "Recent (2010s+)": "Recent",
                "Classic (Pre-2000s)": "Classic",
                "2000s": "2000s"
            }
        
        return content_type_map[content_type], era_map[era_preference]

    def render_recommendations(self, recommendations, mood_name):
        """Render the recommendations in an attractive layout"""
        st.header(f"🎉 Step 3: Your Personalized Recommendations for {mood_name}")
        
        if not recommendations:
            st.warning("No recommendations found. Try adjusting your filters.")
            return
        
        if isinstance(recommendations, dict) and 'error' in recommendations:
            st.error(recommendations['error'])
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Recommendations", len(recommendations))
        with col2:
            avg_score = np.mean([r['similarity_score'] for r in recommendations])
            st.metric("Average Match", f"{avg_score:.1%}")
        with col3:
            movies_count = sum(1 for r in recommendations if r['type'] == 'Movie')
            st.metric("Movies", movies_count)
        with col4:
            shows_count = sum(1 for r in recommendations if r['type'] == 'TV Show')
            st.metric("TV Shows", shows_count)
        
        st.markdown("---")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                # Create a card-like container
                st.markdown(f"""
                <div class="rec-card">
                    <h3>#{i} 🎬 {rec['title']} ({rec['release_year']})</h3>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                        <div>
                            <strong>Type:</strong> {rec['type']} | 
                            <strong>Duration:</strong> {rec['duration']} | 
                            <strong>Match Score:</strong> ⭐ {rec['similarity_score']:.3f}
                        </div>
                    </div>
                    <p><strong>Genres:</strong> {rec['genres']}</p>
                    <p><strong>Why it matches your mood:</strong> {rec['mood_match_reason']}</p>
                    <p><em>{rec['description']}</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button(f"👍 Watchlist", key=f"watch_{i}"):
                        st.success(f"Added '{rec['title']}' to your watchlist!")
                with col2:
                    if st.button(f"🔄 Similar", key=f"similar_{i}"):
                        similar_content = self.recommender.get_similar_content(rec['title'], top_n=3)
                        if isinstance(similar_content, list):
                            st.info(f"Content similar to '{rec['title']}':")
                            for similar in similar_content:
                                st.write(f"- {similar['title']} (Score: {similar['similarity_score']:.3f}")
                        else:
                            st.error(similar_content.get('error', 'Error finding similar content'))
                with col3:
                    if st.button(f"📊 Why this recommendation?", key=f"why_{i}"):
                        st.write(f"**AI Explanation:** This was recommended because it has high thematic alignment with {mood_name.lower()} content, featuring genres and storytelling styles that typically resonate with viewers in your current emotional state.")
                
                if i < len(recommendations):  # Don't add separator after last item
                    st.markdown("---")

    def render_quick_recommendations(self):
        """Render quick recommendation buttons for common scenarios"""
        st.sidebar.header("⚡ Quick Picks")
        
        quick_scenarios = {
            "Friday Night Fun": ("happy_upbeat", "Movie", "Any"),
            "Relaxing Weekend": ("relaxed", "Any", "Any"),
            "Thrilling Night": ("thrilled", "Movie", "Recent"),
            "Nostalgic Evening": ("nostalgic", "Any", "Classic")
        }
        
        for scenario, (mood, content_type, era) in quick_scenarios.items():
            if st.sidebar.button(scenario, use_container_width=True):
                st.session_state.quick_scenario = scenario
                st.session_state.selected_mood = mood
                st.session_state.auto_generate = True

    def run(self):
        """Run the main application"""
        self.render_header()
        
        # Sidebar sections
        with st.sidebar:
            st.header("⚙️ Configuration")
            self.render_training_section()
            self.render_quick_recommendations()
            
            if self.recommender.is_trained:
                st.success("✅ Model Ready!")
                st.write(f"Catalog size: {len(self.recommender.df_clean)} titles")
                st.write(f"Available moods: {len(self.recommender.mood_genre_mapping)}")

        # Main content
        if not self.recommender.is_trained:
            st.info("""
            ## 🚀 Getting Started
            
            1. **Use 'Download via Kaggle'** in the sidebar for automatic dataset download
            2. **Train the model** - this will analyze all the content
            3. **Select your mood** and get personalized recommendations!
            
           
            """)
            return

        # Mood selection
        selected_mood = self.render_mood_selection()
        
        if selected_mood:
            st.success(f"🎯 Selected: {st.session_state.selected_mood_name}")
            
            # Preferences
            content_type, era_preference = self.render_preferences()
            
            # Generate recommendations
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🚀 Generate Personalized Recommendations", use_container_width=True):
                    with st.spinner('🔮 Analyzing thousands of titles to find your perfect matches...'):
                        recommendations = self.recommender.recommend_by_mood(
                            selected_mood, 
                            top_n=8,
                            content_type=content_type,
                            era_preference=era_preference
                        )
                        st.session_state.recommendations = recommendations
            
            with col2:
                if st.button("🎲 Surprise Me! (Random Mood)", use_container_width=True):
                    import random
                    moods = list(self.recommender.mood_vectors.keys())
                    random_mood = random.choice(moods)
                    st.session_state.selected_mood = random_mood
                    st.session_state.selected_mood_name = random_mood.replace('_', ' ').title()
                    st.rerun()
            
            # Display recommendations if available
            if 'recommendations' in st.session_state:
                self.render_recommendations(
                    st.session_state.recommendations, 
                    st.session_state.selected_mood_name
                )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ from 5 man Khataza team | explore your content-based Movie recommendation algorithm</p>
            <p>Data Source: Netflix Catalog | Model: Content-Based Filtering</p>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    app = MoodFlixApp()
    app.run()