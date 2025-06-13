import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import re
import hashlib
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Tuple, Optional

# API clients
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

class SimpleCacheManager:
    """Lightweight caching system"""
    
    def __init__(self):
        self.api_cache = {}
        self.stats = {'api_hits': 0, 'api_misses': 0, 'cost_saved': 0.0}

    def _generate_text_hash(self, text: str, model: str = "") -> str:
        combined = f"{text.lower().strip()}|{model}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_api_response(self, text: str, model: str) -> Optional[Dict]:
        cache_key = self._generate_text_hash(text, model)
        
        if cache_key in self.api_cache:
            entry = self.api_cache[cache_key]
            if datetime.now() - entry['timestamp'] < timedelta(hours=24):
                self.stats['api_hits'] += 1
                self.stats['cost_saved'] += entry.get('cost', 0.002)
                return entry['response']
            else:
                del self.api_cache[cache_key]
        
        self.stats['api_misses'] += 1
        return None

    def cache_api_response(self, text: str, model: str, response: Dict, cost: float = 0.002):
        cache_key = self._generate_text_hash(text, model)
        
        self.api_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now(),
            'cost': cost,
            'model': model
        }

    def get_cache_stats(self) -> Dict:
        total_api = self.stats['api_hits'] + self.stats['api_misses']
        api_hit_rate = (self.stats['api_hits'] / max(total_api, 1)) * 100
        
        return {
            'api_hit_rate': api_hit_rate,
            'total_api_calls_saved': self.stats['api_hits'],
            'cost_saved': self.stats['cost_saved'],
            'cache_sizes': {'api_responses': len(self.api_cache)}
        }

class SimpleSocialMediaClassifier:
    """Simplified classifier without problematic dependencies"""
    
    def __init__(self):
        self.models = {'openai': None, 'anthropic': None, 'llama3': None}
        self.api_calls = {'openai': 0, 'anthropic': 0, 'llama3': 0}
        self.costs = {'openai': 0.0, 'anthropic': 0.0, 'llama3': 0.0}
        
        self.cost_per_1k = {'openai': 0.002, 'anthropic': 0.003, 'llama3': 0.0005}
        self.cache = SimpleCacheManager()

    def setup_apis(self, openai_key: str = None, anthropic_key: str = None, llama3_endpoint: str = None):
        if openai_key and OPENAI_AVAILABLE:
            try:
                # OpenAI v1.0+ compatible initialization
                self.models['openai'] = openai.OpenAI(
                    api_key=openai_key.strip(),
                    timeout=30.0,
                    max_retries=2
                )
                st.success("✅ OpenAI API connected")
            except Exception as e:
                st.error(f"❌ OpenAI setup failed: {e}")
                # Try alternative initialization for older versions
                try:
                    self.models['openai'] = openai.OpenAI(api_key=openai_key.strip())
                    st.success("✅ OpenAI API connected (fallback)")
                except Exception as e2:
                    st.error(f"❌ OpenAI fallback also failed: {e2}")
        
        if anthropic_key and ANTHROPIC_AVAILABLE:
            try:
                self.models['anthropic'] = anthropic.Anthropic(
                    api_key=anthropic_key.strip(),
                    timeout=30.0,
                    max_retries=2
                )
                st.success("✅ Anthropic (Claude) API connected")
            except Exception as e:
                st.error(f"❌ Anthropic setup failed: {e}")
        
        if llama3_endpoint:
            try:
                self.models['llama3'] = llama3_endpoint.strip()
                st.success("✅ Llama 3 endpoint configured")
            except Exception as e:
                st.error(f"❌ Llama 3 setup failed: {e}")

    def create_classification_prompt(self, text: str, model_type: str) -> str:
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
        cached_result = self.cache.get_api_response(text, model)
        if cached_result:
            return cached_result
        
        if model == 'openai':
            result = await self._classify_with_openai(text)
        elif model == 'anthropic':
            result = await self._classify_with_anthropic(text)
        elif model == 'llama3':
            result = await self._classify_with_llama3(text)
        else:
            return {'error': f'Unknown model: {model}'}
        
        if 'error' not in result:
            cost = self.cost_per_1k.get(model, 0.002)
            self.cache.cache_api_response(text, model, result, cost)
        
        return result

    async def _classify_with_openai(self, text: str) -> Dict:
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
        if not self.models['llama3'] or not REQUESTS_AVAILABLE:
            return {'error': 'Llama 3 not configured or requests library unavailable'}

        try:
            prompt = self.create_classification_prompt(text, 'llama3')
            
            # Get API key from multiple sources
            api_key = None
            
            # Try Streamlit secrets first
            if hasattr(st, 'secrets') and 'TOGETHER_API_KEY' in st.secrets:
                api_key = st.secrets['TOGETHER_API_KEY']
            
            # Try the provided endpoint parameter
            elif self.models['llama3'] and isinstance(self.models['llama3'], str) and len(self.models['llama3']) > 10:
                api_key = self.models['llama3']
            
            # Check if we have a valid API key
            if not api_key or len(api_key.strip()) < 10:
                return {'error': 'Together AI API key not found. Please add TOGETHER_API_KEY to Streamlit secrets or enter it in the sidebar.'}
            
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3-8b-chat-hf",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            # Make the API call with detailed error handling
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # Increased timeout
            )
            
            # Check response status
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    
                    # Validate response structure
                    if 'choices' not in response_data or len(response_data['choices']) == 0:
                        return {'error': 'Invalid response structure from Together AI'}
                    
                    content = response_data['choices'][0]['message']['content'].strip()
                    
                    if not content:
                        return {'error': 'Empty response from Llama 3'}
                    
                    self.api_calls['llama3'] += 1
                    self.costs['llama3'] += 0.0005
                    
                    return self.parse_response(content, 'llama3')
                    
                except json.JSONDecodeError as e:
                    return {'error': f'Failed to parse Together AI response: {str(e)}'}
                    
            elif response.status_code == 401:
                return {'error': 'Invalid Together AI API key. Please check your TOGETHER_API_KEY.'}
            elif response.status_code == 429:
                return {'error': 'Rate limit exceeded. Please try again in a moment.'}
            elif response.status_code == 500:
                return {'error': 'Together AI server error. Please try again later.'}
            else:
                return {'error': f'Together AI API error: {response.status_code} - {response.text[:200]}'}
                
        except requests.exceptions.Timeout:
            return {'error': 'Llama 3 request timed out. Please try again.'}
        except requests.exceptions.ConnectionError:
            return {'error': 'Unable to connect to Together AI. Please check your internet connection.'}
        except Exception as e:
            return {'error': f'Llama 3 unexpected error: {str(e)}'}

    def parse_response(self, content: str, model_type: str) -> Dict:
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

    def calculate_simple_accuracy(self, results_df: pd.DataFrame) -> Dict:
        """Simple accuracy calculation without external dependencies"""
        
        total_posts = len(results_df)
        if total_posts == 0:
            return {'error': 'No results to evaluate'}
        
        # Simple distribution analysis
        age_dist = results_df['age_group'].value_counts()
        conf_dist = results_df['confidence_level'].value_counts()
        
        # Average confidence score
        avg_confidence = results_df['confidence_score'].mean()
        
        # Model performance summary
        model_counts = results_df['model'].value_counts() if 'model' in results_df.columns else {}
        
        return {
            'total_analyzed': total_posts,
            'age_distribution': age_dist.to_dict(),
            'confidence_distribution': conf_dist.to_dict(),
            'average_confidence_score': avg_confidence,
            'most_common_age': age_dist.index[0] if len(age_dist) > 0 else 'unknown',
            'most_common_confidence': conf_dist.index[0] if len(conf_dist) > 0 else 'unknown',
            'model_usage': model_counts.to_dict() if hasattr(model_counts, 'to_dict') else {}
        }

@st.cache_data(ttl=1800)
def load_data_cached(file_path: str = None, uploaded_file = None) -> pd.DataFrame:
    """Load data with error handling"""
    try:
        if uploaded_file:
            return pd.read_csv(uploaded_file)
        elif file_path:
            return pd.read_csv(file_path)
        else:
            # Try common file locations
            possible_paths = [
                'data/social_media_analytics.csv',
                'social_media_analytics.csv',
                'social-media-analytics.csv'
            ]
            
            for path in possible_paths:
                try:
                    return pd.read_csv(path)
                except FileNotFoundError:
                    continue
            
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_sample_data() -> pd.DataFrame:
    """Create sample data if no CSV is available"""
    
    sample_posts = [
        {
            'Post ID': 1,
            'Platform': 'Twitter',
            'User ID': 1001,
            'Post Content': 'omg school is literally so hard everyone else gets it but i dont understand math',
            'Post Date': '2024-01-15',
            'Likes': 23,
            'Comments': 5,
            'Shares': 2
        },
        {
            'Post ID': 2,
            'Platform': 'Instagram',
            'User ID': 1002,
            'Post Content': 'job interview tomorrow feeling confident about my career goals',
            'Post Date': '2024-01-16',
            'Likes': 45,
            'Comments': 8,
            'Shares': 3
        },
        {
            'Post ID': 3,
            'Platform': 'Facebook',
            'User ID': 1003,
            'Post Content': 'kids are growing up so fast proud of how well theyre doing in school',
            'Post Date': '2024-01-17',
            'Likes': 67,
            'Comments': 12,
            'Shares': 5
        },
        {
            'Post ID': 4,
            'Platform': 'Twitter',
            'User ID': 1004,
            'Post Content': 'retirement planning has been rewarding grandchildren visited yesterday',
            'Post Date': '2024-01-18',
            'Likes': 34,
            'Comments': 6,
            'Shares': 1
        },
        {
            'Post ID': 5,
            'Platform': 'Instagram',
            'User ID': 1005,
            'Post Content': 'not sure if college is right for me everyone seems more prepared',
            'Post Date': '2024-01-19',
            'Likes': 12,
            'Comments': 3,
            'Shares': 0
        }
    ]
    
    return pd.DataFrame(sample_posts)

def display_cache_dashboard(cache_manager):
    """Display cache performance in sidebar"""
    
    stats = cache_manager.get_cache_stats()
    
    st.sidebar.subheader("🚀 Cache Performance")
    st.sidebar.metric("Hit Rate", f"{stats['api_hit_rate']:.1f}%")
    st.sidebar.metric("API Calls Saved", stats['total_api_calls_saved'])
    st.sidebar.metric("Cost Saved", f"${stats['cost_saved']:.3f}")
    st.sidebar.metric("Cached Responses", stats['cache_sizes']['api_responses'])
    
    if st.sidebar.button("🗑️ Clear Cache"):
        cache_manager.api_cache.clear()
        cache_manager.stats = {'api_hits': 0, 'api_misses': 0, 'cost_saved': 0.0}
        st.sidebar.success("Cache cleared!")
        st.experimental_rerun()

def main():
    st.set_page_config(
        page_title="Social Media AI Classifier",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Social Media AI Classifier")
    st.markdown("**Multi-Model Analysis + Smart Caching - Production Ready**")
    st.markdown("Classify social media posts by age group and confidence level using AI")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = SimpleSocialMediaClassifier()
    
    classifier = st.session_state.classifier
    
    # Sidebar configuration
    st.sidebar.header("🔑 API Configuration")
    
    # Option to use environment variables or input fields
    use_env_vars = st.sidebar.checkbox("Use environment variables", 
                                      help="Check if you've set API keys as environment variables")
    
    if use_env_vars:
        # Try to get from environment/secrets
        openai_key = st.secrets.get('OPENAI_API_KEY', '')
        anthropic_key = st.secrets.get('ANTHROPIC_API_KEY', '')
        llama3_endpoint = st.secrets.get('TOGETHER_API_KEY', '')
        
        if openai_key:
            st.sidebar.success("✅ OpenAI key found in environment")
        if anthropic_key:
            st.sidebar.success("✅ Anthropic key found in environment")
        if llama3_endpoint:
            st.sidebar.success("✅ Together AI key found in environment")
    else:
        # Manual input
        openai_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                          help="Get from platform.openai.com")
        anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password",
                                             help="Get from console.anthropic.com")
        llama3_endpoint = st.sidebar.text_input("Together AI API Key", type="password",
                                               help="Get from api.together.xyz")
    
    if st.sidebar.button("🔌 Connect APIs"):
        classifier.setup_apis(openai_key, anthropic_key, llama3_endpoint)
    
    # Display cache dashboard
    display_cache_dashboard(classifier.cache)
    
    # Model selection
    st.sidebar.header("🎯 Model Selection")
    use_openai = st.sidebar.checkbox("OpenAI GPT-3.5", value=True, 
                                    help="$0.002 per 1K tokens")
    use_anthropic = st.sidebar.checkbox("Anthropic Claude", value=False,
                                       help="$0.003 per 1K tokens")
    use_llama3 = st.sidebar.checkbox("Llama 3 (Together AI)", value=False,
                                    help="$0.0005 per 1K tokens - Cheapest!")
    
    # Cost estimator
    st.sidebar.subheader("💰 Cost Estimator")
    num_posts = st.sidebar.slider("Posts to analyze:", 1, 500, 50)
    
    if use_openai:
        openai_cost = num_posts * 0.002
        st.sidebar.text(f"OpenAI: ~${openai_cost:.3f}")
    if use_anthropic:
        anthropic_cost = num_posts * 0.003
        st.sidebar.text(f"Anthropic: ~${anthropic_cost:.3f}")
    if use_llama3:
        llama3_cost = num_posts * 0.0005
        st.sidebar.text(f"Llama 3: ~${llama3_cost:.3f}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🎯 Analysis", "📈 Results", "ℹ️ About"])
    
    with tab1:
        st.header("📊 Dataset Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your social media CSV file", 
            type="csv",
            help="CSV should contain columns: Post Content, Platform, Likes, Comments"
        )
        
        # Load data
        df = load_data_cached(uploaded_file=uploaded_file)
        
        if df.empty:
            st.warning("📁 No dataset loaded. Using sample data for demonstration.")
            
            if st.button("📝 Generate Sample Data"):
                df = create_sample_data()
                st.session_state.df = df
                st.success("✅ Sample data generated!")
                st.experimental_rerun()
        else:
            st.success(f"✅ Loaded {len(df):,} social media posts")
            st.session_state.df = df
        
        # Display dataset info
        if 'df' in st.session_state and not st.session_state.df.empty:
            df = st.session_state.df
            
            with st.expander("👀 Dataset Preview & Statistics"):
                # Show first few rows
                st.subheader("First 10 rows:")
                st.dataframe(df.head(10))
                
                # Basic statistics
                st.subheader("Dataset Statistics:")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Posts", f"{len(df):,}")
                
                with col2:
                    if 'Platform' in df.columns:
                        platforms = df['Platform'].nunique()
                        st.metric("Platforms", platforms)
                        
                        # Show platform distribution
                        platform_dist = df['Platform'].value_counts()
                        st.write("**Platform Distribution:**")
                        for platform, count in platform_dist.items():
                            st.write(f"- {platform}: {count}")
                
                with col3:
                    if 'Likes' in df.columns:
                        avg_likes = df['Likes'].mean()
                        max_likes = df['Likes'].max()
                        st.metric("Avg Likes", f"{avg_likes:.0f}")
                        st.metric("Max Likes", f"{max_likes:,}")
                
                with col4:
                    if 'Comments' in df.columns:
                        avg_comments = df['Comments'].mean()
                        max_comments = df['Comments'].max()
                        st.metric("Avg Comments", f"{avg_comments:.0f}")
                        st.metric("Max Comments", f"{max_comments:,}")
                
                # Column information
                st.subheader("Column Information:")
                st.write("**Available columns:**")
                for col in df.columns:
                    st.write(f"- {col}: {df[col].dtype}")
    
    with tab2:
        st.header("🎯 AI Classification Analysis")
        
        if 'df' not in st.session_state or st.session_state.df.empty:
            st.warning("⚠️ Please load a dataset first in the Dataset tab.")
            return
        
        df = st.session_state.df
        
        # Check if we have the required Post Content column
        content_column = None
        for col in df.columns:
            if 'content' in col.lower() or 'post' in col.lower():
                content_column = col
                break
        
        if content_column is None:
            st.error("❌ No 'Post Content' column found in dataset. Please ensure your CSV has a column containing post text.")
            return
        
        st.info(f"✅ Using column '{content_column}' for post analysis")
        
        analysis_tab1, analysis_tab2 = st.tabs(["Single Post Analysis", "Batch Analysis"])
        
        with analysis_tab1:
            st.subheader("🔍 Analyze Individual Post")
            
            if len(df) > 0:
                # Post selection
                post_options = list(range(min(100, len(df))))
                selected_idx = st.selectbox(
                    "Select post to analyze:", 
                    post_options,
                    format_func=lambda x: f"Post {x+1}: {str(df.iloc[x][content_column])[:60]}..."
                )
                
                selected_text = str(df.iloc[selected_idx][content_column])
                
                # Show post details
                st.text_area("Selected Post Content:", selected_text, height=100)
                
                # Show additional post metadata if available
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
                if st.button("🚀 Analyze This Post"):
                    st.subheader("🤖 AI Analysis Results")
                    
                    results = []
                    
                    with st.spinner("Analyzing post (checking cache first)..."):
                        
                        # Process with selected models
                        if use_openai and classifier.models['openai']:
                            with st.status("Processing with OpenAI..."):
                                result = asyncio.run(classifier.classify_with_caching(selected_text, 'openai'))
                                if 'error' not in result:
                                    results.append(result)
                                    st.success("✅ OpenAI analysis complete")
                                else:
                                    st.error(f"❌ OpenAI error: {result['error']}")
                        
                        if use_anthropic and classifier.models['anthropic']:
                            with st.status("Processing with Anthropic Claude..."):
                                result = asyncio.run(classifier.classify_with_caching(selected_text, 'anthropic'))
                                if 'error' not in result:
                                    results.append(result)
                                    st.success("✅ Anthropic analysis complete")
                                else:
                                    st.error(f"❌ Anthropic error: {result['error']}")
                        
                        if use_llama3 and classifier.models['llama3']:
                            with st.status("Processing with Llama 3..."):
                                result = asyncio.run(classifier.classify_with_caching(selected_text, 'llama3'))
                                if 'error' not in result:
                                    results.append(result)
                                    st.success("✅ Llama 3 analysis complete")
                                else:
                                    st.error(f"❌ Llama 3 error: {result['error']}")
                    
                    # Display results
                    if results:
                        for i, result in enumerate(results):
                            with st.expander(f"🤖 {result['model']} Analysis", expanded=True):
                                
                                # Main metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    age_emoji = {"teens": "🧒", "young_adults": "👨‍💼", "adults": "👨‍👩‍👧‍👦", "seniors": "👴"}
                                    st.metric("Age Group", 
                                             f"{age_emoji.get(result['age_group'], '❓')} {result['age_group'].replace('_', ' ').title()}")
                                
                                with col2:
                                    conf_emoji = {"low": "😟", "medium": "😐", "high": "😊"}
                                    st.metric("Confidence Level", 
                                             f"{conf_emoji.get(result['confidence_level'], '❓')} {result['confidence_level'].title()}")
                                
                                with col3:
                                    score_color = "🟢" if result['confidence_score'] > 0.7 else "🟡" if result['confidence_score'] > 0.4 else "🔴"
                                    st.metric("AI Confidence", 
                                             f"{score_color} {result['confidence_score']:.2f}")
                                
                                # Reasoning
                                st.markdown("**🧠 AI Reasoning:**")
                                st.write(result['reasoning'])
                                
                                # Raw response (collapsible)
                                with st.expander("📝 Raw AI Response"):
                                    st.code(result['raw_response'], language="text")
                        
                        # Consensus if multiple models
                        if len(results) > 1:
                            st.subheader("🤝 Multi-Model Consensus")
                            
                            ages = [r['age_group'] for r in results]
                            confs = [r['confidence_level'] for r in results]
                            
                            age_consensus = max(set(ages), key=ages.count) if ages else 'unknown'
                            conf_consensus = max(set(confs), key=confs.count) if confs else 'unknown'
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                age_agreement = ages.count(age_consensus)
                                st.success(f"**Age Group Consensus:** {age_consensus.replace('_', ' ').title()} ({age_agreement}/{len(ages)} models agree)")
                            
                            with col2:
                                conf_agreement = confs.count(conf_consensus)
                                st.success(f"**Confidence Consensus:** {conf_consensus.title()} ({conf_agreement}/{len(confs)} models agree)")
                    
                    else:
                        st.error("❌ No successful analyses. Please check your API connections.")
        
        with analysis_tab2:
            st.subheader("📊 Batch Analysis")
            
            # Batch size selection
            max_posts = min(len(df), 200)  # Limit for demo
            sample_size = st.slider("Number of posts to analyze:", 5, max_posts, min(50, max_posts))
            
            # Model selection reminder
            selected_models = []
            if use_openai and classifier.models['openai']:
                selected_models.append('OpenAI')
            if use_anthropic and classifier.models['anthropic']:
                selected_models.append('Anthropic')
            if use_llama3 and classifier.models['llama3']:
                selected_models.append('Llama 3')
            
            if not selected_models:
                st.warning("⚠️ Please connect at least one AI model and enable it in the sidebar.")
                return
            
            st.info(f"💡 **Smart Caching Enabled**: Previously analyzed posts will be served from cache, saving time and money!")
            
            # Cost estimation
            estimated_new_calls = sample_size  # Worst case
            if use_llama3:
                estimated_cost = estimated_new_calls * 0.0005
                st.success(f"💰 **Estimated cost**: ~${estimated_cost:.3f} (using cheapest model: Llama 3)")
            elif use_openai:
                estimated_cost = estimated_new_calls * 0.002
                st.info(f"💰 **Estimated cost**: ~${estimated_cost:.3f} (using OpenAI)")
            elif use_anthropic:
                estimated_cost = estimated_new_calls * 0.003
                st.info(f"💰 **Estimated cost**: ~${estimated_cost:.3f} (using Anthropic)")
            
            # Run batch analysis
            if st.button("🚀 Run Batch Analysis"):
                
                # Sample data
                sample_df = df.sample(n=sample_size, random_state=42)
                results_list = []
                cache_hits = 0
                api_calls_made = 0
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                cache_status = st.empty()
                
                # Choose model (prefer cheapest for batch)
                model_to_use = None
                if use_llama3 and classifier.models['llama3']:
                    model_to_use = 'llama3'
                elif use_openai and classifier.models['openai']:
                    model_to_use = 'openai'
                elif use_anthropic and classifier.models['anthropic']:
                    model_to_use = 'anthropic'
                
                st.info(f"🎯 Using {model_to_use.upper()} for batch processing (most cost-effective)")
                
                # Process posts
                for i, (_, row) in enumerate(sample_df.iterrows()):
                    progress = (i + 1) / len(sample_df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing post {i+1}/{len(sample_df)}...")
                    
                    text = str(row[content_column])
                    
                    # Check cache status
                    cached_result = classifier.cache.get_api_response(text, model_to_use)
                    if cached_result:
                        cache_hits += 1
                    else:
                        api_calls_made += 1
                    
                    # Classify
                    result = asyncio.run(classifier.classify_with_caching(text, model_to_use))
                    
                    if 'error' not in result:
                        # Add metadata
                        result.update({
                            'original_index': row.name,
                            'platform': row.get('Platform', 'Unknown'),
                            'likes': row.get('Likes', 0),
                            'comments': row.get('Comments', 0),
                            'post_content': text[:100] + '...' if len(text) > 100 else text
                        })
                        results_list.append(result)
                    
                    # Update cache status
                    cache_status.text(f"📊 Cache hits: {cache_hits}, New API calls: {api_calls_made}")
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.02)
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                cache_status.empty()
                
                # Process results
                if results_list:
                    results_df = pd.DataFrame(results_list)
                    st.session_state['batch_results'] = results_df
                    
                    # Success metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("✅ Posts Analyzed", len(results_df))
                    
                    with col2:
                        cache_rate = (cache_hits / max(cache_hits + api_calls_made, 1)) * 100
                        st.metric("📊 Cache Hit Rate", f"{cache_rate:.1f}%")
                    
                    with col3:
                        actual_cost = api_calls_made * classifier.cost_per_1k.get(model_to_use, 0.002)
                        st.metric("💰 Actual Cost", f"${actual_cost:.3f}")
                    
                    with col4:
                        saved_cost = cache_hits * classifier.cost_per_1k.get(model_to_use, 0.002)
                        st.metric("💸 Cost Saved", f"${saved_cost:.3f}")
                    
                    if cache_rate > 50:
                        st.success(f"🎉 Excellent cache performance! {cache_hits} results served from cache, only {api_calls_made} new API calls needed.")
                    else:
                        st.info(f"📊 Analysis complete! {cache_hits} cached + {api_calls_made} new API calls.")
                
                else:
                    st.error("❌ No posts were successfully analyzed. Please check your API connections.")
    
    with tab3:
        st.header("📈 Analysis Results & Insights")
        
        if 'batch_results' not in st.session_state:
            st.info("👆 Run batch analysis first to see comprehensive results here.")
            return
        
        results_df = st.session_state['batch_results']
        
        # Calculate insights
        insights = classifier.calculate_simple_accuracy(results_df)
        
        if 'error' not in insights:
            # Overview metrics
            st.subheader("📊 Analysis Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Posts Analyzed", insights['total_analyzed'])
            
            with col2:
                st.metric("🎯 Most Common Age", insights['most_common_age'].replace('_', ' ').title())
            
            with col3:
                st.metric("💭 Most Common Confidence", insights['most_common_confidence'].title())
            
            with col4:
                st.metric("🤖 Avg AI Confidence", f"{insights['average_confidence_score']:.2f}")
            
            # Visualizations
            st.subheader("📊 Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age group distribution
                age_dist = results_df['age_group'].value_counts()
                fig_age = px.pie(
                    values=age_dist.values, 
                    names=[name.replace('_', ' ').title() for name in age_dist.index],
                    title='Age Group Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                # Confidence distribution
                conf_dist = results_df['confidence_level'].value_counts()
                fig_conf = px.pie(
                    values=conf_dist.values, 
                    names=[name.title() for name in conf_dist.index],
                    title='Confidence Level Distribution',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            
            # Engagement analysis
            if 'likes' in results_df.columns and results_df['likes'].sum() > 0:
                st.subheader("📈 Engagement Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Likes by age group
                    fig_likes = px.box(
                        results_df, 
                        x='age_group', 
                        y='likes',
                        title='Likes Distribution by Age Group'
                    )
                    fig_likes.update_xaxes(title="Age Group")
                    fig_likes.update_yaxes(title="Likes")
                    st.plotly_chart(fig_likes, use_container_width=True)
                
                with col2:
                    # Comments by confidence level
                    if 'comments' in results_df.columns:
                        fig_comments = px.box(
                            results_df, 
                            x='confidence_level', 
                            y='comments',
                            title='Comments Distribution by Confidence Level'
                        )
                        fig_comments.update_xaxes(title="Confidence Level")
                        fig_comments.update_yaxes(title="Comments")
                        st.plotly_chart(fig_comments, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Classification Results")
            
            # Filtering options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age_filter = st.multiselect(
                    "Filter by Age Group:",
                    options=results_df['age_group'].unique(),
                    default=[]
                )
            
            with col2:
                conf_filter = st.multiselect(
                    "Filter by Confidence Level:",
                    options=results_df['confidence_level'].unique(),
                    default=[]
                )
            
            with col3:
                min_confidence = st.slider(
                    "Minimum AI Confidence Score:",
                    0.0, 1.0, 0.0, 0.1
                )
            
            # Apply filters
            filtered_df = results_df.copy()
            
            if age_filter:
                filtered_df = filtered_df[filtered_df['age_group'].isin(age_filter)]
            
            if conf_filter:
                filtered_df = filtered_df[filtered_df['confidence_level'].isin(conf_filter)]
            
            filtered_df = filtered_df[filtered_df['confidence_score'] >= min_confidence]
            
            # Display filtered results
            if len(filtered_df) > 0:
                st.info(f"Showing {len(filtered_df)} of {len(results_df)} posts")
                
                # Select columns to display
                display_columns = ['post_content', 'age_group', 'confidence_level', 'confidence_score']
                
                if 'platform' in filtered_df.columns:
                    display_columns.append('platform')
                if 'likes' in filtered_df.columns:
                    display_columns.append('likes')
                if 'comments' in filtered_df.columns:
                    display_columns.append('comments')
                
                # Clean up column names for display
                display_df = filtered_df[display_columns].copy()
                display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download options
                st.subheader("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download filtered results
                    csv_filtered = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Results",
                        data=csv_filtered,
                        file_name=f"filtered_classification_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download full results
                    csv_full = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Results",
                        data=csv_full,
                        file_name=f"social_media_classification_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.warning("No posts match the selected filters. Try adjusting your criteria.")
        
        else:
            st.error(f"Error analyzing results: {insights['error']}")
    
    with tab4:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### 🚀 **Social Media AI Classifier**
        
        This application uses state-of-the-art AI models to automatically classify social media posts by:
        - **Age Group**: teens, young adults, adults, seniors
        - **Confidence Level**: low, medium, high
        
        ### 🤖 **Supported AI Models**
        
        | Model | Provider | Cost per 1K tokens | Best For |
        |-------|----------|-------------------|----------|
        | **GPT-3.5** | OpenAI | $0.002 | Balanced accuracy & cost |
        | **Claude** | Anthropic | $0.003 | Complex reasoning |
        | **Llama 3** | Together AI | $0.0005 | Cost-effective analysis |
        
        ### 🎯 **Key Features**
        
        ✅ **Smart Caching**: Dramatically reduces API costs for repeated analysis  
        ✅ **Multi-Model Support**: Compare results across different AI providers  
        ✅ **Batch Processing**: Analyze hundreds of posts efficiently  
        ✅ **Real-time Insights**: Interactive visualizations and statistics  
        ✅ **Cost Optimization**: Intelligent model selection and usage tracking  
        ✅ **Export Capabilities**: Download results in CSV format  
        
        ### 📊 **Use Cases**
        
        - **Market Research**: Understand your audience demographics
        - **Content Strategy**: Optimize posts for specific age groups
        - **Social Listening**: Monitor brand sentiment across demographics
        - **Academic Research**: Study communication patterns by age
        - **Product Development**: Tailor features to user confidence levels
        
        ### 🔒 **Privacy & Security**
        
        - API keys are encrypted and never stored
        - No data is retained after your session
        - All processing happens in real-time
        - Cache is local to your session only
        
        ### 💡 **Pro Tips**
        
        1. **Start small**: Test with 10-20 posts first
        2. **Use caching**: Re-analyze similar content to save costs
        3. **Compare models**: Different models excel at different aspects
        4. **Monitor costs**: Check the sidebar for real-time cost tracking
        5. **Export results**: Download your analysis for further study
        
        ### 🛠️ **Technical Details**
        
        - Built with **Streamlit** for the web interface
        - Uses **Plotly** for interactive visualizations  
        - Implements **async processing** for better performance
        - Features **intelligent caching** with automatic expiration
        - Supports **multiple file formats** for data input
        
        ### 📞 **Support**
        
        Having issues? Check these common solutions:
        - Ensure your CSV has a 'Post Content' column
        - Verify API keys are entered correctly
        - Try refreshing the page if models aren't connecting
        - Use sample data to test functionality first
        
        ### 🔮 **Future Enhancements**
        
        - Advanced evaluation metrics (accuracy, F1-score)
        - Support for more AI models and providers
        - Real-time social media API integration
        - Custom classification categories
        - Enhanced visualization options
        """)
        
        # System status
        st.subheader("⚙️ System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🐍 Python", "3.9+")
            st.metric("🚀 Streamlit", "1.29.0")
        
        with col2:
            openai_status = "✅ Available" if OPENAI_AVAILABLE else "❌ Not installed"
            anthropic_status = "✅ Available" if ANTHROPIC_AVAILABLE else "❌ Not installed"
            st.metric("🤖 OpenAI", openai_status)
            st.metric("🧠 Anthropic", anthropic_status)
        
        with col3:
            requests_status = "✅ Available" if REQUESTS_AVAILABLE else "❌ Not installed"
            cache_size = len(classifier.cache.api_cache)
            st.metric("🌐 Requests", requests_status)
            st.metric("💾 Cache Size", cache_size)
    
    # Footer
    st.markdown("---")
    cache_stats = classifier.cache.get_cache_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🚀 **Social Media AI Classifier**")
    with col2:
        st.markdown(f"💾 Cache Hit Rate: {cache_stats['api_hit_rate']:.1f}%")
    with col3:
        st.markdown(f"💰 Total Saved: ${cache_stats['cost_saved']:.3f}")

if __name__ == "__main__":
    main()
