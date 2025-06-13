import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import json
import re
import hashlib
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Optional

# API clients with safe imports
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

class SocialMediaClassifier:
    def __init__(self):
        self.models = {'openai': None, 'anthropic': None, 'llama3': None}
        self.api_calls = {'openai': 0, 'anthropic': 0, 'llama3': 0}
        self.costs = {'openai': 0.0, 'anthropic': 0.0, 'llama3': 0.0}
        self.cost_per_1k = {'openai': 0.002, 'anthropic': 0.003, 'llama3': 0.0005}
        self.cache = CacheManager()

    def setup_apis(self, openai_key: str = None, anthropic_key: str = None, llama3_key: str = None):
        if openai_key and OPENAI_AVAILABLE:
            try:
                self.models['openai'] = openai.OpenAI(api_key=openai_key.strip(), timeout=30.0)
                st.success("OpenAI API connected successfully")
            except Exception as e:
                st.error(f"OpenAI setup failed: {e}")
        
        if anthropic_key and ANTHROPIC_AVAILABLE:
            try:
                self.models['anthropic'] = anthropic.Anthropic(api_key=anthropic_key.strip(), timeout=30.0)
                st.success("Anthropic API connected successfully")
            except Exception as e:
                st.error(f"Anthropic setup failed: {e}")
        
        if llama3_key:
            try:
                self.models['llama3'] = llama3_key.strip()
                st.success("Llama 3 endpoint configured")
            except Exception as e:
                st.error(f"Llama 3 setup failed: {e}")

    def create_classification_prompt(self, text: str, model_type: str) -> str:
        base_prompt = f"""Analyze this social media post and classify the author.

TASK: Determine AGE GROUP and CONFIDENCE LEVEL

AGE GROUPS:
- teens: 13-19 years (school, slang, peers)
- young_adults: 20-30 years (career, independence)  
- adults: 31-55 years (family, responsibilities)
- seniors: 55+ years (retirement, grandchildren)

CONFIDENCE LEVELS:
- low: self-doubt, negative self-talk
- medium: balanced perspective
- high: self-assured, confident

POST: "{text}"

RESPONSE FORMAT:
Age Group: [teens/young_adults/adults/seniors]
Confidence Level: [low/medium/high]
Reasoning: [Brief explanation]
Score: [0.0-1.0]"""

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
            return self.parse_response(content, 'OpenAI GPT-3.5')
                
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
            return self.parse_response(content, 'Anthropic Claude')
                
        except Exception as e:
            return {'error': f'Anthropic error: {str(e)}'}

    async def _classify_with_llama3(self, text: str) -> Dict:
        if not self.models['llama3'] or not REQUESTS_AVAILABLE:
            return {'error': 'Llama 3 not configured'}

        try:
            prompt = self.create_classification_prompt(text, 'llama3')
            
            api_key = self.models['llama3']
            if len(api_key) < 10:
                return {'error': 'Invalid Llama 3 API key'}
            
            headers = {
                "Authorization": f"Bearer {api_key}",
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
                timeout=60
            )
            
            if response.status_code == 200:
                self.api_calls['llama3'] += 1
                self.costs['llama3'] += 0.0005
                
                response_data = response.json()
                content = response_data['choices'][0]['message']['content'].strip()
                return self.parse_response(content, 'Llama 3')
            else:
                return {'error': f'Llama 3 API error: {response.status_code}'}
                
        except Exception as e:
            return {'error': f'Llama 3 error: {str(e)}'}

    def parse_response(self, content: str, model_name: str) -> Dict:
        result = {
            'model': model_name,
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
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                result['reasoning'] = match.group(1).strip()[:200]
                break
        
        # Extract confidence score
        score_pattern = r"score:?\s*([0-9]*\.?[0-9]+)"
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

    def create_ground_truth(self, df: pd.DataFrame, content_column: str) -> pd.DataFrame:
        """Create ground truth using keyword heuristics"""
        
        sample_df = df.copy()
        sample_df['true_age_group'] = 'unknown'
        sample_df['true_confidence_level'] = 'unknown'
        
        for idx, row in sample_df.iterrows():
            text = str(row[content_column]).lower()
            
            # Age group rules
            if any(word in text for word in ['school', 'homework', 'class', 'omg', 'literally']):
                sample_df.at[idx, 'true_age_group'] = 'teens'
            elif any(word in text for word in ['college', 'job', 'career', 'apartment']):
                sample_df.at[idx, 'true_age_group'] = 'young_adults'
            elif any(word in text for word in ['kids', 'family', 'mortgage', 'parenting']):
                sample_df.at[idx, 'true_age_group'] = 'adults'
            elif any(word in text for word in ['retirement', 'grandchildren', 'health']):
                sample_df.at[idx, 'true_age_group'] = 'seniors'
            else:
                sample_df.at[idx, 'true_age_group'] = 'young_adults'
            
            # Confidence rules
            if any(word in text for word in ['terrible', 'awful', 'scared', 'worried', 'confused']):
                sample_df.at[idx, 'true_confidence_level'] = 'low'
            elif any(word in text for word in ['confident', 'excellent', 'proud', 'sure']):
                sample_df.at[idx, 'true_confidence_level'] = 'high'
            else:
                sample_df.at[idx, 'true_confidence_level'] = 'medium'
        
        return sample_df

    def calculate_accuracy(self, results_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        """Calculate accuracy metrics"""
        
        if len(results_df) != len(ground_truth_df):
            return {'error': 'Results and ground truth have different lengths'}
        
        # Merge based on index
        results_df = results_df.reset_index(drop=True)
        ground_truth_df = ground_truth_df.reset_index(drop=True)
        
        age_correct = (results_df['age_group'] == ground_truth_df['true_age_group']).sum()
        conf_correct = (results_df['confidence_level'] == ground_truth_df['true_confidence_level']).sum()
        total = len(results_df)
        
        age_accuracy = age_correct / total if total > 0 else 0
        conf_accuracy = conf_correct / total if total > 0 else 0
        
        return {
            'sample_size': total,
            'age_accuracy': age_accuracy,
            'confidence_accuracy': conf_accuracy,
            'age_correct': age_correct,
            'conf_correct': conf_correct
        }

@st.cache_data(ttl=1800)
def load_data_cached(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        # Return sample data
        sample_data = [
            {
                'Post ID': 1,
                'Platform': 'Twitter',
                'Post Content': 'omg school is literally so hard everyone else gets it but i dont understand math',
                'Likes': 23,
                'Comments': 5
            },
            {
                'Post ID': 2,
                'Platform': 'Instagram', 
                'Post Content': 'job interview tomorrow feeling confident about my career goals',
                'Likes': 45,
                'Comments': 8
            },
            {
                'Post ID': 3,
                'Platform': 'Facebook',
                'Post Content': 'kids are growing up so fast proud of how well theyre doing in school',
                'Likes': 67,
                'Comments': 12
            },
            {
                'Post ID': 4,
                'Platform': 'LinkedIn',
                'Post Content': 'retirement planning has been rewarding grandchildren visited yesterday',
                'Likes': 34,
                'Comments': 6
            },
            {
                'Post ID': 5,
                'Platform': 'Twitter',
                'Post Content': 'not sure if college is right for me everyone seems more prepared',
                'Likes': 12,
                'Comments': 3
            }
        ]
        return pd.DataFrame(sample_data)

def main():
    st.set_page_config(
        page_title="Social Media AI Classifier",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Social Media AI Classifier")
    st.markdown("**Multi-Model Analysis with Smart Caching and Evaluation**")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = SocialMediaClassifier()
    
    classifier = st.session_state.classifier
    
    # Sidebar
    st.sidebar.header("API Configuration")
    
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password")
    llama3_key = st.sidebar.text_input("Together AI API Key", type="password")
    
    if st.sidebar.button("Connect APIs"):
        classifier.setup_apis(openai_key, anthropic_key, llama3_key)
    
    # Cache stats
    stats = classifier.cache.get_cache_stats()
    st.sidebar.metric("Cache Hit Rate", f"{stats['api_hit_rate']:.1f}%")
    st.sidebar.metric("Cost Saved", f"${stats['cost_saved']:.3f}")
    
    # Model selection
    st.sidebar.header("Model Selection")
    use_openai = st.sidebar.checkbox("OpenAI GPT-3.5", value=True)
    use_anthropic = st.sidebar.checkbox("Anthropic Claude", value=False)
    use_llama3 = st.sidebar.checkbox("Llama 3", value=False)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Analysis", "Results", "Evaluation"])
    
    with tab1:
        st.header("Dataset Management")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        df = load_data_cached(uploaded_file)
        
        st.success(f"Loaded {len(df)} posts")
        st.dataframe(df.head())
        
        st.session_state.df = df
    
    with tab2:
        st.header("AI Classification Analysis")
        
        if 'df' not in st.session_state:
            st.warning("Please load data first")
            return
        
        df = st.session_state.df
        
        # Find content column
        content_column = None
        for col in df.columns:
            if 'content' in col.lower():
                content_column = col
                break
        
        if not content_column:
            st.error("No content column found")
            return
        
        st.info(f"Using column: {content_column}")
        
        # Single post analysis
        st.subheader("Single Post Analysis")
        
        post_idx = st.selectbox("Select post:", range(len(df)))
        selected_text = df.iloc[post_idx][content_column]
        
        st.text_area("Post content:", selected_text, height=100)
        
        if st.button("Analyze Post"):
            results = []
            
            with st.spinner("Analyzing..."):
                if use_openai and classifier.models['openai']:
                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'openai'))
                    if 'error' not in result:
                        results.append(result)
                
                if use_anthropic and classifier.models['anthropic']:
                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'anthropic'))
                    if 'error' not in result:
                        results.append(result)
                
                if use_llama3 and classifier.models['llama3']:
                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'llama3'))
                    if 'error' not in result:
                        results.append(result)
            
            for result in results:
                st.subheader(f"{result['model']} Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age Group", result['age_group'].replace('_', ' ').title())
                with col2:
                    st.metric("Confidence Level", result['confidence_level'].title())
                with col3:
                    st.metric("AI Confidence", f"{result['confidence_score']:.2f}")
                
                st.write("**Reasoning:**", result['reasoning'])
                
                if st.checkbox(f"Show raw response ({result['model']})", key=f"raw_{result['model']}"):
                    st.text(result['raw_response'])
        
        # Batch analysis
        st.subheader("Batch Analysis")
        
        sample_size = st.slider("Posts to analyze:", 5, min(50, len(df)), 10)
        
        if st.button("Run Batch Analysis"):
            sample_df = df.sample(n=sample_size, random_state=42)
            results_list = []
            
            progress_bar = st.progress(0)
            
            # Choose model
            model_to_use = None
            if use_llama3 and classifier.models['llama3']:
                model_to_use = 'llama3'
            elif use_openai and classifier.models['openai']:
                model_to_use = 'openai'
            elif use_anthropic and classifier.models['anthropic']:
                model_to_use = 'anthropic'
            
            if not model_to_use:
                st.error("Please connect and select at least one model")
                return
            
            for i, (_, row) in enumerate(sample_df.iterrows()):
                progress_bar.progress((i + 1) / len(sample_df))
                
                text = row[content_column]
                result = asyncio.run(classifier.classify_with_caching(text, model_to_use))
                
                if 'error' not in result:
                    result.update({
                        'original_index': row.name,
                        'post_content': text[:100] + '...' if len(text) > 100 else text
                    })
                    results_list.append(result)
                
                time.sleep(0.02)
            
            progress_bar.empty()
            
            if results_list:
                results_df = pd.DataFrame(results_list)
                st.session_state['batch_results'] = results_df
                st.success(f"Analyzed {len(results_df)} posts")
    
    with tab3:
        st.header("Analysis Results")
        
        if 'batch_results' not in st.session_state:
            st.info("Run batch analysis first")
            return
        
        results_df = st.session_state['batch_results']
        
        # Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posts Analyzed", len(results_df))
        with col2:
            most_common_age = results_df['age_group'].mode()[0]
            st.metric("Most Common Age", most_common_age.replace('_', ' ').title())
        with col3:
            most_common_conf = results_df['confidence_level'].mode()[0]
            st.metric("Most Common Confidence", most_common_conf.title())
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            age_dist = results_df['age_group'].value_counts()
            fig_age = px.pie(values=age_dist.values, names=age_dist.index, title='Age Groups')
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            conf_dist = results_df['confidence_level'].value_counts()
            fig_conf = px.pie(values=conf_dist.values, names=conf_dist.index, title='Confidence Levels')
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Results table
        st.subheader("Detailed Results")
        display_columns = ['post_content', 'age_group', 'confidence_level', 'confidence_score']
        st.dataframe(results_df[display_columns])
        
        # Download
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Results",
            csv,
            f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    with tab4:
        st.header("Model Evaluation")
        
        if 'batch_results' not in st.session_state:
            st.warning("Run batch analysis first")
            return
        
        results_df = st.session_state['batch_results']
        original_df = st.session_state['df']
        
        # Find content column
        content_column = None
        for col in original_df.columns:
            if 'content' in col.lower():
                content_column = col
                break
        
        st.info("This demo uses keyword-based ground truth. In production, use human-annotated labels.")
        
        if st.button("Generate Ground Truth"):
            with st.spinner("Generating ground truth labels..."):
                ground_truth_df = classifier.create_ground_truth(original_df, content_column)
                st.session_state['ground_truth'] = ground_truth_df
                st.success("Ground truth generated")
        
        if 'ground_truth' in st.session_state:
            ground_truth_df = st.session_state['ground_truth']
            
            if st.button("Calculate Evaluation Metrics"):
                with st.spinner("Calculating metrics..."):
                    eval_results = classifier.calculate_accuracy(results_df, ground_truth_df)
                    
                    if 'error' in eval_results:
                        st.error(eval_results['error'])
                    else:
                        st.session_state['eval_results'] = eval_results
                        st.success("Evaluation complete")
            
            if 'eval_results' in st.session_state:
                eval_results = st.session_state['eval_results']
                
                st.subheader("Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sample Size", eval_results['sample_size'])
                
                with col2:
                    age_acc = eval_results['age_accuracy']
                    st.metric("Age Accuracy", f"{age_acc:.3f}")
                
                with col3:
                    conf_acc = eval_results['confidence_accuracy']
                    st.metric("Confidence Accuracy", f"{conf_acc:.3f}")
                
                with col4:
                    overall = (age_acc + conf_acc) / 2
                    st.metric("Overall Score", f"{overall:.3f}")
                
                # Performance breakdown
                st.subheader("Detailed Results")
                
                st.write(f"**Age Group Classification:**")
                st.write(f"- Correct predictions: {eval_results['age_correct']}/{eval_results['sample_size']}")
                st.write(f"- Accuracy: {eval_results['age_accuracy']:.1%}")
                
                st.write(f"**Confidence Level Classification:**")
                st.write(f"- Correct predictions: {eval_results['conf_correct']}/{eval_results['sample_size']}")
                st.write(f"- Accuracy: {eval_results['confidence_accuracy']:.1%}")
                
                # Performance insights
                if overall >= 0.8:
                    st.success("Excellent performance! The model is working very well.")
                elif overall >= 0.6:
                    st.info("Good performance. Consider fine-tuning prompts for improvement.")
                else:
                    st.warning("Performance could be improved. Review prompt engineering and ground truth quality.")
    
    # Footer
    st.markdown("---")
    st.markdown("Social Media AI Classifier - Production Ready with Evaluation")

if __name__ == "__main__":
    main()
