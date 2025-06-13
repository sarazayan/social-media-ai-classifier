import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import re
import hashlib
import pickle
import os
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# API clients (only import if available)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class CacheManager:
    """Advanced caching system for API responses and computations"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Different cache files for different purposes
        self.api_cache_file = self.cache_dir / "api_responses.pkl"
        self.feature_cache_file = self.cache_dir / "features.pkl"
        self.batch_cache_file = self.cache_dir / "batch_results.pkl"
        
        # Cache statistics
        self.stats = {
            'api_hits': 0,
            'api_misses': 0,
            'feature_hits': 0,
            'feature_misses': 0,
            'cost_saved': 0.0
        }
        
        # Load existing caches
        self.api_cache = self._load_cache(self.api_cache_file)
        self.feature_cache = self._load_cache(self.feature_cache_file)
        self.batch_cache = self._load_cache(self.batch_cache_file)
        
        # Cache expiration times (in hours)
        self.expiration_times = {
            'api_responses': 24,  # API responses cached for 24 hours
            'features': 168,     # Feature extraction cached for 1 week
            'batch_results': 72  # Batch results cached for 3 days
        }

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from file"""
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def _save_cache(self, cache_data: Dict, cache_file: Path):
        """Save cache to file"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            st.error(f"Cache save error: {e}")

    def _generate_text_hash(self, text: str, model: str = "") -> str:
        """Generate hash for text + model combination"""
        combined = f"{text.lower().strip()}|{model}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _is_expired(self, timestamp: datetime, cache_type: str) -> bool:
        """Check if cache entry is expired"""
        expiration_hours = self.expiration_times.get(cache_type, 24)
        return datetime.now() - timestamp > timedelta(hours=expiration_hours)

    def get_api_response(self, text: str, model: str) -> Optional[Dict]:
        """Get cached API response"""
        cache_key = self._generate_text_hash(text, model)
        
        if cache_key in self.api_cache:
            entry = self.api_cache[cache_key]
            if not self._is_expired(entry['timestamp'], 'api_responses'):
                self.stats['api_hits'] += 1
                self.stats['cost_saved'] += entry.get('cost', 0.002)
                return entry['response']
            else:
                # Remove expired entry
                del self.api_cache[cache_key]
        
        self.stats['api_misses'] += 1
        return None

    def cache_api_response(self, text: str, model: str, response: Dict, cost: float = 0.002):
        """Cache API response"""
        cache_key = self._generate_text_hash(text, model)
        
        self.api_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now(),
            'cost': cost,
            'model': model
        }
        
        # Save to disk periodically (every 10 entries)
        if len(self.api_cache) % 10 == 0:
            self._save_cache(self.api_cache, self.api_cache_file)

    def get_features(self, text: str) -> Optional[Dict]:
        """Get cached feature extraction"""
        cache_key = self._generate_text_hash(text)
        
        if cache_key in self.feature_cache:
            entry = self.feature_cache[cache_key]
            if not self._is_expired(entry['timestamp'], 'features'):
                self.stats['feature_hits'] += 1
                return entry['features']
            else:
                del self.feature_cache[cache_key]
        
        self.stats['feature_misses'] += 1
        return None

    def cache_features(self, text: str, features: Dict):
        """Cache feature extraction results"""
        cache_key = self._generate_text_hash(text)
        
        self.feature_cache[cache_key] = {
            'features': features,
            'timestamp': datetime.now()
        }
        
        if len(self.feature_cache) % 20 == 0:
            self._save_cache(self.feature_cache, self.feature_cache_file)

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_api = self.stats['api_hits'] + self.stats['api_misses']
        total_feature = self.stats['feature_hits'] + self.stats['feature_misses']
        
        api_hit_rate = (self.stats['api_hits'] / max(total_api, 1)) * 100
        feature_hit_rate = (self.stats['feature_hits'] / max(total_feature, 1)) * 100
        
        return {
            'api_hit_rate': api_hit_rate,
            'feature_hit_rate': feature_hit_rate,
            'total_api_calls_saved': self.stats['api_hits'],
            'cost_saved': self.stats['cost_saved'],
            'cache_sizes': {
                'api_responses': len(self.api_cache),
                'features': len(self.feature_cache),
                'batch_results': len(self.batch_cache)
            }
        }

    def clear_cache(self, cache_type: str = "all"):
        """Clear specific or all caches"""
        if cache_type == "all" or cache_type == "api":
            self.api_cache.clear()
            if self.api_cache_file.exists():
                self.api_cache_file.unlink()
        
        if cache_type == "all" or cache_type == "features":
            self.feature_cache.clear()
            if self.feature_cache_file.exists():
                self.feature_cache_file.unlink()
        
        if cache_type == "all" or cache_type == "batch":
            self.batch_cache.clear()
            if self.batch_cache_file.exists():
                self.batch_cache_file.unlink()

    def cleanup_expired(self):
        """Remove expired entries from all caches"""
        # Cleanup API cache
        expired_keys = []
        for key, entry in self.api_cache.items():
            if self._is_expired(entry['timestamp'], 'api_responses'):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.api_cache[key]
        
        # Cleanup feature cache
        expired_keys = []
        for key, entry in self.feature_cache.items():
            if self._is_expired(entry['timestamp'], 'features'):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.feature_cache[key]
        
        # Save cleaned caches
        self._save_cache(self.api_cache, self.api_cache_file)
        self._save_cache(self.feature_cache, self.feature_cache_file)

class CachedSocialMediaClassifier:
    """Enhanced classifier with comprehensive caching"""
    
    def __init__(self):
        self.models = {
            'openai': None,
            'anthropic': None,
            'llama3': None
        }
        self.api_calls = {'openai': 0, 'anthropic': 0, 'llama3': 0}
        self.costs = {'openai': 0.0, 'anthropic': 0.0, 'llama3': 0.0}
        
        # Cost per 1K tokens
        self.cost_per_1k = {
            'openai': 0.002,
            'anthropic': 0.003,
            'llama3': 0.0005
        }
        
        # Initialize cache manager
        self.cache = CacheManager()
        
        # Feature engineering patterns (cached computation)
        self.age_vocabulary = {
            'teens': {
                'slang': ['omg', 'literally', 'like', 'so', 'totally', 'whatever', 'lol', 'tbh'],
                'topics': ['school', 'homework', 'teacher', 'class', 'grade', 'test', 'parents'],
                'emotions': ['stressed', 'excited', 'annoyed', 'confused', 'worried'],
                'social': ['friends', 'drama', 'party', 'crush', 'popular', 'weird']
            },
            'young_adults': {
                'career': ['job', 'work', 'career', 'interview', 'resume', 'salary', 'boss'],
                'education': ['college', 'university', 'degree', 'graduate', 'student'],
                'independence': ['apartment', 'rent', 'bills', 'adulting', 'moving'],
                'relationships': ['dating', 'relationship', 'partner', 'single', 'love']
            },
            'adults': {
                'family': ['kids', 'children', 'parenting', 'family', 'spouse', 'marriage'],
                'financial': ['mortgage', 'insurance', 'savings', 'budget', 'taxes'],
                'professional': ['management', 'team', 'project', 'business', 'leadership'],
                'balance': ['schedule', 'busy', 'juggling', 'balance', 'priorities']
            },
            'seniors': {
                'life_stage': ['retirement', 'retired', 'grandchildren', 'legacy', 'wisdom'],
                'health': ['health', 'doctor', 'medical', 'medication', 'wellness'],
                'reflection': ['years', 'experience', 'remember', 'back then', 'learned'],
                'values': ['respect', 'honor', 'tradition', 'values', 'principle']
            }
        }

    @st.cache_data(ttl=3600)  # Streamlit cache for 1 hour
    def load_dataset(_self, file_path: str) -> pd.DataFrame:
        """Cached dataset loading"""
        return pd.read_csv(file_path)

    def setup_apis(self, openai_key: str = None, anthropic_key: str = None, llama3_endpoint: str = None):
        """Setup API clients"""
        if openai_key and OPENAI_AVAILABLE:
            try:
                self.models['openai'] = openai.OpenAI(api_key=openai_key)
                st.success("✅ OpenAI API connected")
            except Exception as e:
                st.error(f"❌ OpenAI setup failed: {e}")
        
        if anthropic_key and ANTHROPIC_AVAILABLE:
            try:
                self.models['anthropic'] = anthropic.Anthropic(api_key=anthropic_key)
                st.success("✅ Anthropic (Claude) API connected")
            except Exception as e:
                st.error(f"❌ Anthropic setup failed: {e}")
        
        if llama3_endpoint:
            try:
                self.models['llama3'] = llama3_endpoint
                st.success("✅ Llama 3 endpoint configured")
            except Exception as e:
                st.error(f"❌ Llama 3 setup failed: {e}")

    def extract_features_cached(self, text: str) -> Dict:
        """Feature extraction with caching"""
        
        # Check cache first
        cached_features = self.cache.get_features(text)
        if cached_features:
            return cached_features
        
        # Compute features if not cached
        features = self._compute_features(text)
        
        # Cache the result
        self.cache.cache_features(text, features)
        
        return features

    def _compute_features(self, text: str) -> Dict:
        """Actual feature computation (expensive operation)"""
        if not text or not isinstance(text, str):
            return {}

        text = text.lower().strip()
        words = text.split()
        features = {}

        # Basic linguistic features
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        # Age vocabulary scoring
        for age_group, categories in self.age_vocabulary.items():
            total_score = 0
            for category, vocab_list in categories.items():
                count = sum(1 for word in vocab_list if word in text)
                features[f'{age_group}_{category}'] = count
                total_score += count
            features[f'{age_group}_total'] = total_score

        # Confidence patterns
        confidence_patterns = {
            'low': ['not good enough', 'terrible', 'awful', 'dont know', 'not sure', 'afraid', 'scared'],
            'medium': ['sometimes', 'usually', 'often', 'learning', 'improving', 'depends'],
            'high': ['confident', 'sure', 'certain', 'excellent', 'great', 'definitely']
        }
        
        for conf_level, patterns in confidence_patterns.items():
            count = sum(1 for pattern in patterns if pattern in text)
            features[f'conf_{conf_level}_total'] = count

        return features

    def create_classification_prompt(self, text: str, model_type: str) -> str:
        """Create optimized prompts for different models"""
        
        base_prompt = f"""Analyze this social media post and classify the author's demographics and psychological state.

CLASSIFICATION TASK:
Determine the author's AGE GROUP and CONFIDENCE LEVEL based on language patterns, topics, and communication style.

AGE GROUPS:
- teens (13-19): School focus, slang, peer dynamics, identity exploration
- young_adults (20-30): Career building, independence, relationships, "adulting"
- adults (31-55): Family responsibilities, career advancement, financial planning
- seniors (55+): Reflection, health awareness, grandchildren, retirement

CONFIDENCE LEVELS:
- low: Self-doubt, comparison to others, negative self-talk, uncertainty
- medium: Balanced perspective, some confidence mixed with uncertainty
- high: Self-assured, confident statements, positive self-image

POST TO ANALYZE:
"{text}"

INSTRUCTIONS:
1. Look for linguistic markers (vocabulary, slang, topics)
2. Analyze psychological patterns (confidence indicators, concerns)
3. Consider life stage markers (responsibilities, relationships, goals)
4. Provide your classification with reasoning

"""

        if model_type == 'openai':
            return base_prompt + """
RESPONSE FORMAT:
Respond with exactly this JSON format:
{
    "age_group": "teens|young_adults|adults|seniors",
    "confidence_level": "low|medium|high",
    "reasoning": "Brief explanation of key indicators",
    "confidence_score": 0.0-1.0
}"""

        elif model_type == 'anthropic':
            return base_prompt + """
Please respond with your analysis in this exact format:

Age Group: [teens/young_adults/adults/seniors]
Confidence Level: [low/medium/high]
Reasoning: [Brief explanation of key linguistic and psychological indicators]
Confidence Score: [0.0-1.0]"""

        elif model_type == 'llama3':
            return base_prompt + """
Respond in this format:
AGE: [teens/young_adults/adults/seniors]
CONFIDENCE: [low/medium/high]  
REASON: [Key indicators that led to this classification]
SCORE: [0.0-1.0]"""

        return base_prompt

    async def classify_with_caching(self, text: str, model: str) -> Dict:
        """Classify with intelligent caching"""
        
        # Check cache first
        cached_result = self.cache.get_api_response(text, model)
        if cached_result:
            return cached_result
        
        # If not cached, make API call
        if model == 'openai':
            result = await self._classify_with_openai(text)
        elif model == 'anthropic':
            result = await self._classify_with_anthropic(text)
        elif model == 'llama3':
            result = await self._classify_with_llama3(text)
        else:
            return {'error': f'Unknown model: {model}'}
        
        # Cache the result if successful
        if 'error' not in result:
            cost = self.cost_per_1k.get(model, 0.002)
            self.cache.cache_api_response(text, model, result, cost)
        
        return result

    async def _classify_with_openai(self, text: str) -> Dict:
        """OpenAI classification (internal method)"""
        if not self.models['openai']:
            return {'error': 'OpenAI not configured'}

        try:
            prompt = self.create_classification_prompt(text, 'openai')
            
            response = self.models['openai'].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            self.api_calls['openai'] += 1
            self.costs['openai'] += 0.002
            
            content = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(content)
                return {
                    'model': 'OpenAI GPT-3.5',
                    'age_group': result.get('age_group', 'unknown'),
                    'confidence_level': result.get('confidence_level', 'unknown'),
                    'reasoning': result.get('reasoning', 'No reasoning provided'),
                    'confidence_score': result.get('confidence_score', 0.5),
                    'raw_response': content
                }
            except json.JSONDecodeError:
                return self.parse_response(content, 'openai')
                
        except Exception as e:
            return {'error': f'OpenAI error: {str(e)}'}

    async def _classify_with_anthropic(self, text: str) -> Dict:
        """Anthropic classification (internal method)"""
        if not self.models['anthropic']:
            return {'error': 'Anthropic not configured'}

        try:
            prompt = self.create_classification_prompt(text, 'anthropic')
            
            response = self.models['anthropic'].messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.api_calls['anthropic'] += 1
            self.costs['anthropic'] += 0.003
            
            content = response.content[0].text.strip()
            return self.parse_response(content, 'anthropic')
                
        except Exception as e:
            return {'error': f'Anthropic error: {str(e)}'}

    async def _classify_with_llama3(self, text: str) -> Dict:
        """Llama 3 classification (internal method)"""
        if not self.models['llama3'] or not REQUESTS_AVAILABLE:
            return {'error': 'Llama 3 not configured'}

        try:
            prompt = self.create_classification_prompt(text, 'llama3')
            
            headers = {
                "Authorization": f"Bearer {st.secrets.get('TOGETHER_API_KEY', '')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3-8b-chat-hf",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                self.api_calls['llama3'] += 1
                self.costs['llama3'] += 0.0005
                
                content = response.json()['choices'][0]['message']['content'].strip()
                return self.parse_response(content, 'llama3')
            else:
                return {'error': f'Llama 3 API error: {response.status_code}'}
                
        except Exception as e:
            return {'error': f'Llama 3 error: {str(e)}'}

    def parse_response(self, content: str, model_type: str) -> Dict:
        """Parse model responses into standardized format"""
        
        result = {
            'model': f'{model_type.title()}',
            'age_group': 'unknown',
            'confidence_level': 'unknown',
            'reasoning': 'Could not parse response',
            'confidence_score': 0.5,
            'raw_response': content
        }
        
        content_lower = content.lower()
        
        # Extract age group
        age_groups = ['teens', 'young_adults', 'adults', 'seniors']
        for age in age_groups:
            if age in content_lower or age.replace('_', ' ') in content_lower:
                result['age_group'] = age
                break
        
        # Extract confidence level
        conf_levels = ['low', 'medium', 'high']
        for conf in conf_levels:
            if conf in content_lower:
                result['confidence_level'] = conf
                break
        
        # Extract reasoning
        reasoning_patterns = [
            r"reasoning:?\s*(.+?)(?:\n|$)",
            r"reason:?\s*(.+?)(?:\n|$)",
            r"explanation:?\s*(.+?)(?:\n|$)",
            r"because\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                result['reasoning'] = match.group(1).strip()[:200]
                break
        
        # Extract confidence score
        score_pattern = r"(?:score|confidence):?\s*([0-9]*\.?[0-9]+)"
        score_match = re.search(score_pattern, content, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0 <= score <= 1:
                    result['confidence_score'] = score
                elif score > 1:
                    result['confidence_score'] = score / 100
            except ValueError:
                pass
        
        return result

def display_cache_dashboard(cache_manager: CacheManager):
    """Display cache performance dashboard"""
    
    st.subheader("🚀 Cache Performance Dashboard")
    
    stats = cache_manager.get_cache_stats()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Hit Rate", f"{stats['api_hit_rate']:.1f}%", 
                 help="Percentage of API calls served from cache")
    
    with col2:
        st.metric("Calls Saved", stats['total_api_calls_saved'],
                 help="Total API calls avoided due to caching")
    
    with col3:
        st.metric("Cost Saved", f"${stats['cost_saved']:.3f}",
                 help="Estimated money saved through caching")
    
    with col4:
        feature_hit_rate = stats['feature_hit_rate']
        st.metric("Feature Hit Rate", f"{feature_hit_rate:.1f}%",
                 help="Percentage of feature extractions served from cache")
    
    # Cache sizes
    st.subheader("💾 Cache Storage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Responses", stats['cache_sizes']['api_responses'],
                 help="Number of cached API responses")
    
    with col2:
        st.metric("Feature Extractions", stats['cache_sizes']['features'],
                 help="Number of cached feature extractions")
    
    with col3:
        st.metric("Batch Results", stats['cache_sizes']['batch_results'],
                 help="Number of cached batch analysis results")
    
    # Cache management
    st.subheader("🧹 Cache Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Cache"):
            cache_manager.clear_cache("all")
            st.success("All caches cleared!")
            st.experimental_rerun()
    
    with col2:
        if st.button("🧹 Clean Expired"):
            cache_manager.cleanup_expired()
            st.success("Expired entries removed!")
            st.experimental_rerun()
    
    with col3:
        if st.button("💾 Force Save"):
            cache_manager._save_cache(cache_manager.api_cache, cache_manager.api_cache_file)
            cache_manager._save_cache(cache_manager.feature_cache, cache_manager.feature_cache_file)
            st.success("Caches saved to disk!")

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_data_cached(file_path: str = None, uploaded_file = None) -> pd.DataFrame:
    """Cached data loading function"""
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    elif file_path:
        return pd.read_csv(file_path)
    else:
        try:
            return pd.read_csv('data/social_media_analytics.csv')
        except FileNotFoundError:
            return pd.DataFrame()

def main():
    """Main Streamlit application with advanced caching"""
    
    st.set_page_config(
        page_title="Social Media AI Classifier (Cached)",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Social Media AI Classifier (Optimized)")
    st.markdown("**Advanced Caching + Multi-Model AI Analysis**")
    st.markdown("Intelligent caching strategies to minimize API costs while maximizing performance")
    
    # Initialize classifier with caching
    if 'classifier' not in st.session_state:
        st.session_state.classifier = CachedSocialMediaClassifier()
    
    classifier = st.session_state.classifier
    
    # Sidebar for API configuration
    st.sidebar.header("🔑 API Configuration")
    
    # API Keys
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password")
    llama3_endpoint = st.sidebar.text_input("Llama 3 API Key", type="password")
    
    if st.sidebar.button("🔌 Connect APIs"):
        classifier.setup_apis(openai_key, anthropic_key, llama3_endpoint)
    
    # Cache dashboard in sidebar
    st.sidebar.header("📊 Cache Status")
    stats = classifier.cache.get_cache_stats()
    st.sidebar.metric("API Hit Rate", f"{stats['api_hit_rate']:.1f}%")
    st.sidebar.metric("Cost Saved", f"${stats['cost_saved']:.3f}")
    st.sidebar.metric("Cached Responses", stats['cache_sizes']['api_responses'])
    
    # Model selection
    st.sidebar.header("🎯 Model Selection")
    use_openai = st.sidebar.checkbox("OpenAI GPT-3.5", value=True)
    use_anthropic = st.sidebar.checkbox("Anthropic Claude", value=False)
    use_llama3 = st.sidebar.checkbox("Llama 3", value=False)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🎯 Analysis", "📈 Results", "🚀 Cache Dashboard"])
    
    with tab1:
        st.header("📊 Dataset Loading")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload social_media_analytics.csv", type="csv")
        
        # Load data with caching
        df = load_data_cached(uploaded_file=uploaded_file)
        
        if not df.empty:
            st.success(f"✅ Loaded {len(df):,} social media posts (cached)")
            
            # Show dataset preview
            with st.expander("👀 Preview Dataset"):
                st.dataframe(df.head(10))
                
                # Basic stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Posts", f"{len(df):,}")
                with col2:
                    st.metric("Platforms", df['Platform'].nunique() if 'Platform' in df.columns else 0)
                with col3:
                    avg_likes = df['Likes'].mean() if 'Likes' in df.columns else 0
                    st.metric("Avg Likes", f"{avg_likes:.0f}")
                with col4:
                    avg_comments = df['Comments'].mean() if 'Comments' in df.columns else 0
                    st.metric("Avg Comments", f"{avg_comments:.0f}")
            
            # Store in session state
            st.session_state.df = df
        else:
            st.warning("📁 No dataset loaded. Please upload a CSV file or ensure social_media_analytics.csv exists.")
    
    with tab2:
        st.header("🎯 AI Classification Analysis")
        
        if 'df' not in st.session_state or st.session_state.df.empty:
            st.warning("Please load a dataset first in the Dataset tab.")
            return
        
        df = st.session_state.df
        
        # Analysis options
        analysis_tab1, analysis_tab2 = st.tabs(["Single Post", "Batch Analysis"])
        
        with analysis_tab1:
            st.subheader("Analyze Individual Post")
            
            # Select post from dataset
            if len(df) > 0:
                post_options = list(range(min(100, len(df))))
                selected_idx = st.selectbox(
                    "Select post:", 
                    post_options,
                    format_func=lambda x: f"Post {x+1}: {df.iloc[x]['Post Content'][:60]}..." if 'Post Content' in df.columns else f"Post {x+1}"
                )
                
                if 'Post Content' in df.columns:
                    selected_text = df.iloc[selected_idx]['Post Content']
                    
                    # Show post details
                    st.text_area("Selected Post:", selected_text, height=100)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'Platform' in df.columns:
                            st.info(f"**Platform:** {df.iloc[selected_idx]['Platform']}")
                        if 'Post Date' in df.columns:
                            st.info(f"**Date:** {df.iloc[selected_idx]['Post Date']}")
                    with col2:
                        if 'Likes' in df.columns:
                            st.info(f"**Likes:** {df.iloc[selected_idx]['Likes']:,}")
                        if 'Comments' in df.columns:
                            st.info(f"**Comments:** {df.iloc[selected_idx]['Comments']:,}")
                    
                    # Analysis button
                    if st.button("🔍 Analyze with Caching"):
                        st.subheader("Analysis Results")
                        
                        results = []
                        
                        with st.spinner("Analyzing (checking cache first)..."):
                            
                            if use_openai and classifier.models['openai']:
                                with st.status("Processing with OpenAI (cached)..."):
                                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'openai'))
                                    if 'error' not in result:
                                        results.append(result)
                                        st.success("✅ OpenAI complete")
                                    else:
                                        st.error(f"❌ OpenAI: {result['error']}")
                            
                            if use_anthropic and classifier.models['anthropic']:
                                with st.status("Processing with Anthropic (cached)..."):
                                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'anthropic'))
                                    if 'error' not in result:
                                        results.append(result)
                                        st.success("✅ Anthropic complete")
                                    else:
                                        st.error(f"❌ Anthropic: {result['error']}")
                            
                            if use_llama3 and classifier.models['llama3']:
                                with st.status("Processing with Llama 3 (cached)..."):
                                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'llama3'))
                                    if 'error' not in result:
                                        results.append(result)
                                        st.success("✅ Llama 3 complete")
                                    else:
                                        st.error(f"❌ Llama 3: {result['error']}")
                        
                        # Display results
                        if results:
                            for result in results:
                                with st.expander(f"🤖 {result['model']} Results", expanded=True):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Age Group", result['age_group'].title())
                                    with col2:
                                        st.metric("Confidence Level", result['confidence_level'].title())
                                    with col3:
                                        st.metric("Confidence Score", f"{result['confidence_score']:.2f}")
                                    
                                    st.markdown(f"**Reasoning:** {result['reasoning']}")
        
        with analysis_tab2:
            st.subheader("Batch Analysis with Smart Caching")
            
            sample_size = st.slider("Number of posts to analyze:", 10, min(200, len(df)), 25)
            
            st.info("💡 **Smart Caching**: Previously analyzed posts will be served from cache, significantly reducing costs and time!")
            
            if st.button("🚀 Run Cached Batch Analysis"):
                
                # Sample data
                sample_df = df.sample(n=sample_size, random_state=42)
                
                results_list = []
                cache_hits = 0
                api_calls_made = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                cache_status = st.empty()
                
                for i, (_, row) in enumerate(sample_df.iterrows()):
                    progress = (i + 1) / len(sample_df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing post {i+1}/{len(sample_df)}...")
                    
                    if 'Post Content' in row:
                        text = row['Post Content']
                        
                        # Use enabled model (prefer cheapest for batch)
                        model_to_use = None
                        if use_llama3 and classifier.models['llama3']:
                            model_to_use = 'llama3'
                        elif use_openai and classifier.models['openai']:
                            model_to_use = 'openai'
                        elif use_anthropic and classifier.models['anthropic']:
                            model_to_use = 'anthropic'
                        
                        if model_to_use:
                            # Check if this will be served from cache
                            cached_result = classifier.cache.get_api_response(text, model_to_use)
                            if cached_result:
                                cache_hits += 1
                            else:
                                api_calls_made += 1
                            
                            result = asyncio.run(classifier.classify_with_caching(text, model_to_use))
                            
                            if 'error' not in result:
                                result.update({
                                    'post_id': row.get('Post ID', i),
                                    'platform': row.get('Platform', 'Unknown'),
                                    'likes': row.get('Likes', 0),
                                    'comments': row.get('Comments', 0),
                                    'post_content': text[:100] + '...' if len(text) > 100 else text
                                })
                                results_list.append(result)
                    
                    # Update cache status
                    cache_status.text(f"Cache hits: {cache_hits}, API calls: {api_calls_made}")
                    
                    # Small delay
                    time.sleep(0.05)
                
                progress_bar.empty()
                status_text.empty()
                cache_status.empty()
                
                if results_list:
                    results_df = pd.DataFrame(results_list)
                    st.session_state['batch_results'] = results_df
                    
                    # Success metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Posts Analyzed", len(results_df))
                    with col2:
                        cache_rate = (cache_hits / max(cache_hits + api_calls_made, 1)) * 100
                        st.metric("Cache Hit Rate", f"{cache_rate:.1f}%")
                    with col3:
                        estimated_cost = api_calls_made * classifier.cost_per_1k.get(model_to_use, 0.002)
                        st.metric("Estimated Cost", f"${estimated_cost:.3f}")
                    with col4:
                        saved_cost = cache_hits * classifier.cost_per_1k.get(model_to_use, 0.002)
                        st.metric("Cost Saved", f"${saved_cost:.3f}")
                    
                    st.success(f"✅ Batch analysis complete! {cache_hits} results served from cache, {api_calls_made} new API calls made.")
    
    with tab3:
        st.header("📈 Analysis Results")
        
        if 'batch_results' in st.session_state:
            results_df = st.session_state['batch_results']
            
            # Results overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age_dist = results_df['age_group'].value_counts()
                fig_age = px.pie(values=age_dist.values, names=age_dist.index, 
                               title='Age Group Distribution')
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                conf_dist = results_df['confidence_level'].value_counts()
                fig_conf = px.pie(values=conf_dist.values, names=conf_dist.index,
                                title='Confidence Level Distribution')
                st.plotly_chart(fig_conf, use_container_width=True)
            
            with col3:
                # Engagement vs Demographics
                if 'likes' in results_df.columns:
                    fig_engagement = px.box(results_df, x='age_group', y='likes',
                                          title='Likes by Age Group')
                    st.plotly_chart(fig_engagement, use_container_width=True)
            
            # Detailed results table
            st.subheader("📊 Detailed Results")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                age_filter = st.multiselect("Filter by Age Group:", results_df['age_group'].unique())
            with col2:
                conf_filter = st.multiselect("Filter by Confidence Level:", results_df['confidence_level'].unique())
            
            filtered_df = results_df.copy()
            if age_filter:
                filtered_df = filtered_df[filtered_df['age_group'].isin(age_filter)]
            if conf_filter:
                filtered_df = filtered_df[filtered_df['confidence_level'].isin(conf_filter)]
            
            st.dataframe(
                filtered_df[['post_content', 'age_group', 'confidence_level', 'confidence_score', 'platform', 'likes', 'comments']],
                use_container_width=True
            )
            
            # Download results
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"social_media_classification_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("👆 Run batch analysis to see results here")
    
    with tab4:
        display_cache_dashboard(classifier.cache)
    
    # Footer with cache info
    st.markdown("---")
    cache_stats = classifier.cache.get_cache_stats()
    st.markdown(f"🚀 **Cached Classifier** | Hit Rate: {cache_stats['api_hit_rate']:.1f}% | Cost Saved: ${cache_stats['cost_saved']:.3f} | Built with Streamlit")

if __name__ == "__main__":
    main()
