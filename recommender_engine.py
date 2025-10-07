# recommender_engine.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
except:
    pass

class MoodBasedRecommender:
    """
    Core recommendation engine for mood-based movie recommendations
    """
    
    def __init__(self):
        self.df_clean = None
        self.feature_matrix = None
        self.mood_vectors = {}
        self.mood_genre_mapping = {}
        self.tfidf_vectorizer = None
        self.mlb = None
        self.similarity_matrix = None
        self.is_trained = False
        
        # Initialize text preprocessing
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        custom_stopwords = {'movie', 'film', 'series', 'show', 'story', 'stories', 'watch', 'see'}
        self.stop_words.update(custom_stopwords)
        
        print("✅ MoodBasedRecommender initialized!")

    def load_and_clean_data(self, file_path):
        """
        Load and clean the Netflix dataset
        """
        print("📥 Loading and cleaning data...")
        
        # Load dataset
        self.df_clean = pd.read_csv(file_path)
        print(f"✅ Loaded dataset with {len(self.df_clean)} records")
        
        # Fill missing values
        text_columns = ['description', 'director', 'cast', 'listed_in']
        for col in text_columns:
            if col in self.df_clean.columns:
                self.df_clean[col] = self.df_clean[col].fillna('Unknown')
        
        # Remove duplicates
        self.df_clean = self.df_clean.drop_duplicates(subset=['title'], keep='first')
        
        # Filter only movies and TV shows
        if 'type' in self.df_clean.columns:
            self.df_clean = self.df_clean[self.df_clean['type'].isin(['Movie', 'TV Show'])]
        
        print(f"✅ Cleaned dataset: {len(self.df_clean)} records remaining")
        return self.df_clean

    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        """
        if pd.isna(text) or text == 'Unknown':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def create_mood_mappings(self):
        """
        Define mood to genre/keyword mappings
        """
        self.mood_genre_mapping = {
            'happy_upbeat': {
                'genres': ['Comedies', 'Music & Musicals', 'Romantic Comedies', 'Stand-Up Comedy', 
                          'Children & Family Movies', 'Animation', 'Lighthearted'],
                'keywords': ['funny', 'hilarious', 'happy', 'joy', 'love', 'comedy', 'laugh', 'celebration'],
                'description': 'Fun, lighthearted content to lift your spirits'
            },
            'sad_reflective': {
                'genres': ['Dramas', 'Independent Movies', 'Romantic Movies', 'Classic Movies', 
                          'Emotional', 'Social Issue Dramas'],
                'keywords': ['emotional', 'heartbreaking', 'sad', 'loss', 'drama', 'reflection'],
                'description': 'Thoughtful, emotional stories for introspection'
            },
            'adventurous': {
                'genres': ['Action & Adventure', 'Sci-Fi & Fantasy', 'Anime Features', 'Thrillers',
                          'Westerns', 'Adventures'],
                'keywords': ['adventure', 'journey', 'quest', 'action', 'thrilling', 'epic'],
                'description': 'Exciting adventures and thrilling journeys'
            },
            'relaxed': {
                'genres': ['Documentaries', 'Nature & Ecology Documentaries', 'Travel & Adventure Documentaries',
                          'Kids TV', 'Educational', 'Cooking & Food'],
                'keywords': ['relaxing', 'calm', 'peaceful', 'nature', 'documentary', 'travel'],
                'description': 'Calm, educational content to unwind with'
            },
            'thrilled': {
                'genres': ['Horror Movies', 'Thrillers', 'Mysteries', 'Crime TV Shows', 
                          'Suspenseful', 'Psychological Thrillers'],
                'keywords': ['thriller', 'suspense', 'horror', 'scary', 'mystery', 'crime'],
                'description': 'Heart-pounding thrillers and mysteries'
            },
            'intellectual': {
                'genres': ['Documentaries', 'Biographical Documentaries', 'Historical Documentaries',
                          'Science & Nature TV', 'Political Documentaries', 'Docuseries'],
                'keywords': ['documentary', 'history', 'science', 'biography', 'theory', 'intellectual'],
                'description': 'Thought-provoking documentaries and educational content'
            },
            'stressed': {
                'genres': ['Comedies', 'Stand-Up Comedy', 'Children & Family Movies', 
                          'Animation', 'Romantic Comedies', 'Light TV Shows'],
                'keywords': ['funny', 'comedy', 'lighthearted', 'humor', 'sitcom', 'laughter'],
                'description': 'Light, funny content to relieve stress'
            },
            'nostalgic': {
                'genres': ['Classic Movies', 'Cult Movies', 'Kids TV', '1980s', '1990s',
                          'Family Movies', 'Animation'],
                'keywords': ['nostalgia', 'classic', 'retro', 'old', 'memory', 'childhood'],
                'description': 'Classic favorites and nostalgic throwbacks'
            }
        }
        print("✅ Mood mappings created!")

    def engineer_features(self):
        """
        Create feature matrix for content-based filtering
        """
        print("🔨 Engineering features...")
        
        # Preprocess descriptions
        self.df_clean['cleaned_description'] = self.df_clean['description'].apply(self.preprocess_text)
        
        # Process genres
        self.df_clean['genres_list'] = self.df_clean['listed_in'].apply(
            lambda x: [genre.strip() for genre in str(x).split(',')] 
            if pd.notna(x) and x != 'Unknown' else ['Unknown']
        )
        
        # Create TF-IDF matrix for descriptions
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df_clean['cleaned_description'])
        
        # Create genre matrix
        self.mlb = MultiLabelBinarizer()
        genre_matrix = self.mlb.fit_transform(self.df_clean['genres_list'])
        
        # Create additional features
        self.df_clean['release_era'] = pd.cut(
            self.df_clean['release_year'],
            bins=[1900, 1980, 1990, 2000, 2010, 2020, 2030],
            labels=['Classic', '80s', '90s', '2000s', '2010s', '2020s']
        )
        
        # Duration processing for movies
        def extract_duration(duration_str):
            if pd.isna(duration_str):
                return 0
            numbers = re.findall(r'\d+', str(duration_str))
            return int(numbers[0]) if numbers else 0
        
        self.df_clean['duration_min'] = self.df_clean['duration'].apply(extract_duration)
        self.df_clean['duration_category'] = pd.cut(
            self.df_clean['duration_min'],
            bins=[0, 30, 90, 180, 1000],
            labels=['Short', 'Medium', 'Feature', 'Epic']
        )
        
        # Create dummy variables for categorical features
        era_dummies = pd.get_dummies(self.df_clean['release_era'], prefix='era').astype(np.float32)
        duration_dummies = pd.get_dummies(self.df_clean['duration_category'], prefix='duration').astype(np.float32)
        type_dummies = pd.get_dummies(self.df_clean['type'], prefix='type').astype(np.float32)
        
        # Convert to sparse matrices
        era_matrix = csr_matrix(era_dummies.values)
        duration_matrix = csr_matrix(duration_dummies.values)
        type_matrix = csr_matrix(type_dummies.values)
        
        # Combine all features with weights
        self.feature_matrix = hstack([
            tfidf_matrix,           # Text content (40%)
            genre_matrix * 0.35,    # Genres (35%)
            era_matrix * 0.1,       # Era (10%)
            duration_matrix * 0.1,  # Duration (10%)
            type_matrix * 0.05      # Content type (5%)
        ])
        
        print(f"✅ Feature matrix created: {self.feature_matrix.shape}")
        
        # Create mood vectors
        self._create_mood_vectors(era_dummies, duration_dummies, type_dummies)

    def _create_mood_vectors(self, era_dummies, duration_dummies, type_dummies):
        """
        Create representative vectors for each mood category
        """
        print("🎭 Creating mood vectors...")
        
        for mood, config in self.mood_genre_mapping.items():
            # Create genre vector for this mood
            mood_genre_vector = np.zeros(len(self.mlb.classes_))
            for i, genre in enumerate(self.mlb.classes_):
                if any(target_genre in genre for target_genre in config['genres']):
                    mood_genre_vector[i] = 1
            
            # Create description vector using keywords
            mood_description = ' '.join(config['keywords'])
            mood_tfidf_vector = self.tfidf_vectorizer.transform([mood_description])
            
            # For era, duration, type - use neutral vectors
            mood_era_vector = np.ones(era_dummies.shape[1]) * 0.5
            mood_duration_vector = np.ones(duration_dummies.shape[1]) * 0.5
            mood_type_vector = np.ones(type_dummies.shape[1]) * 0.5
            
            # Combine all mood features
            mood_combined_vector = hstack([
                mood_tfidf_vector,
                mood_genre_vector.reshape(1, -1) * 0.35,
                mood_era_vector.reshape(1, -1) * 0.1,
                mood_duration_vector.reshape(1, -1) * 0.1,
                mood_type_vector.reshape(1, -1) * 0.05
            ])
            
            self.mood_vectors[mood] = mood_combined_vector
        
        print(f"✅ Created {len(self.mood_vectors)} mood vectors")

    def train(self, data_path):
        """
        Complete training pipeline
        """
        print("🚀 Starting training pipeline...")
        
        # Step 1: Load and clean data
        self.load_and_clean_data(data_path)
        
        # Step 2: Create mood mappings
        self.create_mood_mappings()
        
        # Step 3: Engineer features
        self.engineer_features()
        
        # Step 4: Build similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        self.is_trained = True
        print("🎉 Training completed successfully!")
        
        return True

    def recommend_by_mood(self, mood, top_n=10, content_type=None, era_preference=None):
        """
        Get recommendations based on mood with optional filters
        """
        if not self.is_trained:
            return {"error": "Model not trained. Please call train() first."}
        
        if mood not in self.mood_vectors:
            available_moods = list(self.mood_vectors.keys())
            return {"error": f"Mood '{mood}' not found. Available moods: {available_moods}"}
        
        print(f"🎯 Finding {top_n} recommendations for mood: {mood}")
        
        # Calculate similarity between mood vector and all content
        mood_vector = self.mood_vectors[mood]
        similarities = cosine_similarity(mood_vector, self.feature_matrix).flatten()
        
        # Get indices of top matches
        top_indices = similarities.argsort()[-top_n*3:][::-1]
        
        # Apply filters
        filtered_indices = []
        for idx in top_indices:
            if len(filtered_indices) >= top_n:
                break
                
            item = self.df_clean.iloc[idx]
            
            # Content type filter
            if content_type and item['type'] != content_type:
                continue
                
            # Era filter
            if era_preference and era_preference != 'Any':
                item_era = item['release_era']
                if era_preference == 'Recent' and item_era not in ['2010s', '2020s']:
                    continue
                elif era_preference == 'Classic' and item_era not in ['Classic', '80s', '90s']:
                    continue
                elif era_preference == '2000s' and item_era != '2000s':
                    continue
            
            filtered_indices.append(idx)
        
        # Prepare results
        recommendations = []
        for idx in filtered_indices[:top_n]:
            item = self.df_clean.iloc[idx]
            recommendations.append({
                'title': item['title'],
                'type': item['type'],
                'genres': item['listed_in'],
                'description': item['description'],
                'release_year': item['release_year'],
                'duration': item['duration'],
                'similarity_score': float(similarities[idx]),
                'mood_match_reason': self._explain_mood_match(mood, item)
            })
        
        return recommendations

    def _explain_mood_match(self, mood, item):
        """Generate explanation for why content matches the mood"""
        mood_config = self.mood_genre_mapping[mood]
        item_genres = item['genres_list']
        
        # Find matching genres
        matching_genres = []
        for genre in item_genres:
            for target_genre in mood_config['genres']:
                if target_genre.lower() in genre.lower():
                    matching_genres.append(genre)
                    break
        
        reasons = []
        if matching_genres:
            reasons.append(f"Genres: {', '.join(matching_genres[:2])}")
        
        return " | ".join(reasons) if reasons else "General content match"

    def get_similar_content(self, title, top_n=5):
        """Find similar content based on a liked title"""
        if not self.is_trained:
            return {"error": "Model not trained. Please call train() first."}
        
        if title not in self.df_clean['title'].values:
            return {"error": f"Title '{title}' not found in database"}
        
        title_idx = self.df_clean[self.df_clean['title'] == title].index[0]
        
        # Get similarities for this title
        similarities = self.similarity_matrix[title_idx]
        
        # Get top similar items (excluding itself)
        similar_indices = similarities.argsort()[-(top_n+1):][::-1]
        similar_indices = [idx for idx in similar_indices if idx != title_idx][:top_n]
        
        similar_content = []
        for idx in similar_indices:
            item = self.df_clean.iloc[idx]
            similar_content.append({
                'title': item['title'],
                'type': item['type'],
                'genres': item['listed_in'],
                'description': item['description'],
                'similarity_score': float(similarities[idx])
            })
        
        return similar_content

    def get_available_moods(self):
        """Get list of available moods with descriptions"""
        return {mood: config['description'] for mood, config in self.mood_genre_mapping.items()}

    def save_model(self, file_path):
        """Save the trained model"""
        if not self.is_trained:
            print("❌ Model not trained. Cannot save.")
            return False
        
        model_data = {
            'df_clean': self.df_clean,
            'feature_matrix': self.feature_matrix,
            'mood_vectors': self.mood_vectors,
            'mood_genre_mapping': self.mood_genre_mapping,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'mlb': self.mlb,
            'similarity_matrix': self.similarity_matrix
        }
        
        joblib.dump(model_data, file_path)
        print(f"✅ Model saved to {file_path}")
        return True

    def load_model(self, file_path):
        """Load a pre-trained model"""
        try:
            model_data = joblib.load(file_path)
            
            self.df_clean = model_data['df_clean']
            self.feature_matrix = model_data['feature_matrix']
            self.mood_vectors = model_data['mood_vectors']
            self.mood_genre_mapping = model_data['mood_genre_mapping']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.mlb = model_data['mlb']
            self.similarity_matrix = model_data['similarity_matrix']
            self.is_trained = True
            
            print(f"✅ Model loaded from {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

# for testing purposes
if __name__ == "__main__":
    # Test the recommender
    recommender = MoodBasedRecommender()
    
    # Train with your Netflix data
    # recommender.train('data/netflix_titles.csv')
    
    # Or load a pre-trained model
    # recommender.load_model('trained_model.pkl')
    
    print("🎬 MoodBasedRecommender is ready!")



    