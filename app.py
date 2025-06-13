import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import time
import json
import re
import hashlib
import pickle
import os
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Tuple, Optional

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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

class EvaluationMetrics:
    """Comprehensive evaluation metrics for classification models"""
    
    def __init__(self):
        self.age_groups = ['teens', 'young_adults', 'adults', 'seniors']
        self.confidence_levels = ['low', 'medium', 'high']
    
    def calculate_metrics(self, y_true: List, y_pred: List, labels: List = None) -> Dict:
        """Calculate comprehensive classification metrics"""
        
        if not labels:
            labels = list(set(y_true + y_pred))
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted metrics (accounts for class imbalance)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class detailed metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'per_class_metrics': {
                'precision': dict(zip(labels, precision_per_class)),
                'recall': dict(zip(labels, recall_per_class)),
                'f1': dict(zip(labels, f1_per_class)),
                'support': dict(zip(labels, support_per_class))
            },
            'labels': labels,
            'classification_report': classification_report(y_true, y_pred, labels=labels, output_dict=True)
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix") -> go.Figure:
        """Create interactive confusion matrix plot"""
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        # Create annotations
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append({
                    'x': j, 'y': i,
                    'text': f"{cm[i,j]}<br>({cm_normalized[i,j]:.2%})",
                    'showarrow': False,
                    'font': {'color': 'white' if cm_normalized[i,j] > 0.5 else 'black'}
                })
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            colorscale='Blues',
            colorbar=dict(title="Proportion"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            annotations=annotations,
            width=500,
            height=500
        )
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict: Dict) -> go.Figure:
        """Create metrics comparison chart"""
        
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [metrics_dict[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=models,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_per_class_metrics(self, metrics: Dict, metric_type: str = 'f1') -> go.Figure:
        """Plot per-class performance metrics"""
        
        labels = metrics['labels']
        values = [metrics['per_class_metrics'][metric_type][label] for label in labels]
        
        fig = go.Figure(data=go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Per-Class {metric_type.upper()} Scores",
            xaxis_title="Classes",
            yaxis_title=f"{metric_type.upper()} Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig

class CacheManager:
    """Advanced caching system for API responses and computations"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir) if 'Path' in globals() else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'api_hits': 0,
            'api_misses': 0,
            'cost_saved': 0.0
        }
        
        self.api_cache = {}
        self.feature_cache = {}

    def _generate_text_hash(self, text: str, model: str = "") -> str:
        """Generate hash for text + model combination"""
        combined = f"{text.lower().strip()}|{model}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_api_response(self, text: str, model: str) -> Optional[Dict]:
        """Get cached API response"""
        cache_key = self._generate_text_hash(text, model)
        
        if cache_key in self.api_cache:
            entry = self.api_cache[cache_key]
            # Simple expiration check (24 hours)
            if datetime.now() - entry['timestamp'] < timedelta(hours=24):
                self.stats['api_hits'] += 1
                self.stats['cost_saved'] += entry.get('cost', 0.002)
                return entry['response']
            else:
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

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_api = self.stats['api_hits'] + self.stats['api_misses']
        api_hit_rate = (self.stats['api_hits'] / max(total_api, 1)) * 100
        
        return {
            'api_hit_rate': api_hit_rate,
            'total_api_calls_saved': self.stats['api_hits'],
            'cost_saved': self.stats['cost_saved'],
            'cache_sizes': {
                'api_responses': len(self.api_cache),
                'features': len(self.feature_cache)
            }
        }

class EnhancedSocialMediaClassifier:
    """Enhanced classifier with evaluation metrics"""
    
    def __init__(self):
        self.models = {
            'openai': None,
            'anthropic': None,
            'llama3': None
        }
        self.api_calls = {'openai': 0, 'anthropic': 0, 'llama3': 0}
        self.costs = {'openai': 0.0, 'anthropic': 0.0, 'llama3': 0.0}
        
        self.cost_per_1k = {
            'openai': 0.002,
            'anthropic': 0.003,
            'llama3': 0.0005
        }
        
        self.cache = CacheManager()
        self.evaluator = EvaluationMetrics()

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
        """OpenAI classification"""
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
        """Anthropic classification"""
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
        """Llama 3 classification"""
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

    def create_synthetic_ground_truth(self, df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
        """Create synthetic ground truth labels for evaluation"""
        
        # Sample data for ground truth creation
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
        
        # Simple heuristic-based ground truth (for demo purposes)
        # In real scenarios, you'd have human-annotated labels
        
        sample_df['true_age_group'] = 'unknown'
        sample_df['true_confidence_level'] = 'unknown'
        
        for idx, row in sample_df.iterrows():
            if 'Post Content' in row:
                text = str(row['Post Content']).lower()
                
                # Age group heuristics
                if any(word in text for word in ['school', 'homework', 'class', 'teacher', 'omg', 'literally']):
                    sample_df.at[idx, 'true_age_group'] = 'teens'
                elif any(word in text for word in ['college', 'job', 'career', 'interview', 'apartment']):
                    sample_df.at[idx, 'true_age_group'] = 'young_adults'
                elif any(word in text for word in ['kids', 'family', 'mortgage', 'parenting', 'work']):
                    sample_df.at[idx, 'true_age_group'] = 'adults'
                elif any(word in text for word in ['retirement', 'grandchildren', 'health', 'doctor']):
                    sample_df.at[idx, 'true_age_group'] = 'seniors'
                else:
                    sample_df.at[idx, 'true_age_group'] = 'young_adults'  # default
                
                # Confidence level heuristics
                if any(word in text for word in ['not good', 'terrible', 'awful', 'dont know', 'scared', 'worried']):
                    sample_df.at[idx, 'true_confidence_level'] = 'low'
                elif any(word in text for word in ['confident', 'sure', 'excellent', 'great', 'amazing', 'proud']):
                    sample_df.at[idx, 'true_confidence_level'] = 'high'
                else:
                    sample_df.at[idx, 'true_confidence_level'] = 'medium'  # default
        
        return sample_df

    def evaluate_model_performance(self, results_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        """Evaluate model performance against ground truth"""
        
        # Merge results with ground truth
        merged_df = results_df.merge(
            ground_truth_df[['post_id', 'true_age_group', 'true_confidence_level']], 
            on='post_id', 
            how='inner'
        )
        
        if len(merged_df) == 0:
            return {'error': 'No matching posts found between results and ground truth'}
        
        # Calculate metrics for age group classification
        age_metrics = self.evaluator.calculate_metrics(
            merged_df['true_age_group'].tolist(),
            merged_df['age_group'].tolist(),
            self.evaluator.age_groups
        )
        
        # Calculate metrics for confidence level classification
        conf_metrics = self.evaluator.calculate_metrics(
            merged_df['true_confidence_level'].tolist(),
            merged_df['confidence_level'].tolist(),
            self.evaluator.confidence_levels
        )
        
        return {
            'age_group_metrics': age_metrics,
            'confidence_level_metrics': conf_metrics,
            'sample_size': len(merged_df),
            'merged_data': merged_df
        }

@st.cache_data(ttl=1800)
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

def display_evaluation_dashboard(evaluation_results: Dict):
    """Display comprehensive evaluation dashboard"""
    
    st.subheader("📊 Model Evaluation Dashboard")
    
    if 'error' in evaluation_results:
        st.error(f"Evaluation Error: {evaluation_results['error']}")
        return
    
    age_metrics = evaluation_results['age_group_metrics']
    conf_metrics = evaluation_results['confidence_level_metrics']
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sample Size", evaluation_results['sample_size'])
    with col2:
        st.metric("Age Accuracy", f"{age_metrics['accuracy']:.3f}")
    with col3:
        st.metric("Confidence Accuracy", f"{conf_metrics['accuracy']:.3f}")
    with col4:
        avg_f1 = (age_metrics['f1_weighted'] + conf_metrics['f1_weighted']) / 2
        st.metric("Avg F1-Score", f"{avg_f1:.3f}")
    
    # Detailed metrics tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Age Group Classification")
        
        # Metrics table
        age_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 
                      'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
            'Score': [
                age_metrics['accuracy'],
                age_metrics['precision_macro'],
                age_metrics['recall_macro'],
                age_metrics['f1_macro'],
                age_metrics['precision_weighted'],
                age_metrics['recall_weighted'],
                age_metrics['f1_weighted']
            ]
        })
        age_metrics_df['Score'] = age_metrics_df['Score'].round(3)
        st.dataframe(age_metrics_df, use_container_width=True)
        
        # Per-class metrics
        st.subheader("Per-Class Metrics (Age)")
        age_per_class_df = pd.DataFrame(age_metrics['per_class_metrics']).T
        age_per_class_df = age_per_class_df.round(3)
        st.dataframe(age_per_class_df, use_container_width=True)
    
    with col2:
        st.subheader("💭 Confidence Level Classification")
        
        # Metrics table
        conf_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 
                      'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
            'Score': [
                conf_metrics['accuracy'],
                conf_metrics['precision_macro'],
                conf_metrics['recall_macro'],
                conf_metrics['f1_macro'],
                conf_metrics['precision_weighted'],
                conf_metrics['recall_weighted'],
                conf_metrics['f1_weighted']
            ]
        })
        conf_metrics_df['Score'] = conf_metrics_df['Score'].round(3)
        st.dataframe(conf_metrics_df, use_container_width=True)
        
        # Per-class metrics
        st.subheader("Per-Class Metrics (Confidence)")
        conf_per_class_df = pd.DataFrame(conf_metrics['per_class_metrics']).T
        conf_per_class_df = conf_per_class_df.round(3)
        st.dataframe(conf_per_class_df, use_container_width=True)
    
    # Confusion matrices
    st.subheader("🔄 Confusion Matrices")
    
    col1, col2 = st.columns(2)
    
    evaluator = EvaluationMetrics()
    
    with col1:
        age_cm_fig = evaluator.plot_confusion_matrix(
            age_metrics['confusion_matrix'],
            age_metrics['labels'],
            "Age Group Classification"
        )
        st.plotly_chart(age_cm_fig, use_container_width=True)
    
    with col2:
        conf_cm_fig = evaluator.plot_confusion_matrix(
            conf_metrics['confusion_matrix'],
            conf_metrics['labels'],
            "Confidence Level Classification"
        )
        st.plotly_chart(conf_cm_fig, use_container_width=True)
    
    # Per-class F1 scores
    st.subheader("📈 Per-Class Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_f1_fig = evaluator.plot_per_class_metrics(age_metrics, 'f1')
        age_f1_fig.update_layout(title="Age Group F1-Scores")
        st.plotly_chart(age_f1_fig, use_container_width=True)
    
    with col2:
        conf_f1_fig = evaluator.plot_per_class_metrics(conf_metrics, 'f1')
        conf_f1_fig.update_layout(title="Confidence Level F1-Scores")
        st.plotly_chart(conf_f1_fig, use_container_width=True)
    
    # Classification reports
    with st.expander("📋 Detailed Classification Reports"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Group Report")
            age_report_df = pd.DataFrame(age_metrics['classification_report']).T
            age_report_df = age_report_df.round(3)
            st.dataframe(age_report_df, use_container_width=True)
        
        with col2:
            st.subheader("Confidence Level Report")
            conf_report_df = pd.DataFrame(conf_metrics['classification_report']).T
            conf_report_df = conf_report_df.round(3)
            st.dataframe(conf_report_df, use_container_width=True)

def compare_model_performance(model_results: Dict[str, Dict]) -> None:
    """Compare performance across multiple models"""
    
    st.subheader("🏆 Model Performance Comparison")
    
    if len(model_results) < 2:
        st.info("Need at least 2 model results for comparison. Run evaluation on multiple models first.")
        return
    
    # Prepare comparison data
    comparison_data = []
    
    for model_name, results in model_results.items():
        if 'error' not in results:
            age_metrics = results['age_group_metrics']
            conf_metrics = results['confidence_level_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'Age Accuracy': age_metrics['accuracy'],
                'Age F1 (Macro)': age_metrics['f1_macro'],
                'Age F1 (Weighted)': age_metrics['f1_weighted'],
                'Conf Accuracy': conf_metrics['accuracy'],
                'Conf F1 (Macro)': conf_metrics['f1_macro'],
                'Conf F1 (Weighted)': conf_metrics['f1_weighted'],
                'Sample Size': results['sample_size']
            })
    
    if not comparison_data:
        st.warning("No valid model results found for comparison.")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(comparison_df.round(3), use_container_width=True)
    
    # Create comparison charts
    evaluator = EvaluationMetrics()
    
    # Metrics comparison chart
    metrics_dict = {}
    for model_name, results in model_results.items():
        if 'error' not in results:
            age_metrics = results['age_group_metrics']
            conf_metrics = results['confidence_level_metrics']
            
            # Average metrics across age and confidence tasks
            metrics_dict[model_name] = {
                'accuracy': (age_metrics['accuracy'] + conf_metrics['accuracy']) / 2,
                'precision_macro': (age_metrics['precision_macro'] + conf_metrics['precision_macro']) / 2,
                'recall_macro': (age_metrics['recall_macro'] + conf_metrics['recall_macro']) / 2,
                'f1_macro': (age_metrics['f1_macro'] + conf_metrics['f1_macro']) / 2
            }
    
    if metrics_dict:
        comparison_fig = evaluator.plot_metrics_comparison(metrics_dict)
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Best model recommendations
    if len(comparison_df) > 0:
        st.subheader("🥇 Model Rankings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_age_model = comparison_df.loc[comparison_df['Age F1 (Weighted)'].idxmax(), 'Model']
            best_age_score = comparison_df['Age F1 (Weighted)'].max()
            st.metric("Best for Age Classification", best_age_model, f"F1: {best_age_score:.3f}")
        
        with col2:
            best_conf_model = comparison_df.loc[comparison_df['Conf F1 (Weighted)'].idxmax(), 'Model']
            best_conf_score = comparison_df['Conf F1 (Weighted)'].max()
            st.metric("Best for Confidence Classification", best_conf_model, f"F1: {best_conf_score:.3f}")
        
        with col3:
            # Overall best (average of both tasks)
            comparison_df['Overall F1'] = (comparison_df['Age F1 (Weighted)'] + comparison_df['Conf F1 (Weighted)']) / 2
            best_overall_model = comparison_df.loc[comparison_df['Overall F1'].idxmax(), 'Model']
            best_overall_score = comparison_df['Overall F1'].max()
            st.metric("Best Overall", best_overall_model, f"Avg F1: {best_overall_score:.3f}")

def main():
    """Main Streamlit application with evaluation metrics"""
    
    st.set_page_config(
        page_title="AI Classifier with Metrics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Social Media AI Classifier with Evaluation Metrics")
    st.markdown("**Advanced Caching + Multi-Model Analysis + Comprehensive Evaluation**")
    st.markdown("Complete model evaluation with accuracy, F1-score, recall, precision, and confusion matrices")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = EnhancedSocialMediaClassifier()
    
    classifier = st.session_state.classifier
    
    # Sidebar configuration
    st.sidebar.header("🔑 API Configuration")
    
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password")
    llama3_endpoint = st.sidebar.text_input("Together API Key", type="password")
    
    if st.sidebar.button("🔌 Connect APIs"):
        classifier.setup_apis(openai_key, anthropic_key, llama3_endpoint)
    
    # Cache status
    st.sidebar.header("📊 Cache Status")
    stats = classifier.cache.get_cache_stats()
    st.sidebar.metric("API Hit Rate", f"{stats['api_hit_rate']:.1f}%")
    st.sidebar.metric("Cost Saved", f"${stats['cost_saved']:.3f}")
    
    # Model selection
    st.sidebar.header("🎯 Model Selection")
    use_openai = st.sidebar.checkbox("OpenAI GPT-3.5", value=True)
    use_anthropic = st.sidebar.checkbox("Anthropic Claude", value=False)
    use_llama3 = st.sidebar.checkbox("Llama 3", value=False)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dataset", "🎯 Analysis", "📈 Results", "🔬 Evaluation", "🏆 Model Comparison"])
    
    with tab1:
        st.header("📊 Dataset Loading")
        
        uploaded_file = st.file_uploader("Upload social_media_analytics.csv", type="csv")
        df = load_data_cached(uploaded_file=uploaded_file)
        
        if not df.empty:
            st.success(f"✅ Loaded {len(df):,} social media posts")
            
            with st.expander("👀 Preview Dataset"):
                st.dataframe(df.head(10))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Posts", f"{len(df):,}")
                with col2:
                    platforms = df['Platform'].nunique() if 'Platform' in df.columns else 0
                    st.metric("Platforms", platforms)
                with col3:
                    avg_likes = df['Likes'].mean() if 'Likes' in df.columns else 0
                    st.metric("Avg Likes", f"{avg_likes:.0f}")
                with col4:
                    avg_comments = df['Comments'].mean() if 'Comments' in df.columns else 0
                    st.metric("Avg Comments", f"{avg_comments:.0f}")
            
            st.session_state.df = df
        else:
            st.warning("📁 No dataset loaded. Please upload a CSV file.")
    
    with tab2:
        st.header("🎯 AI Classification Analysis")
        
        if 'df' not in st.session_state or st.session_state.df.empty:
            st.warning("Please load a dataset first in the Dataset tab.")
            return
        
        df = st.session_state.df
        
        analysis_tab1, analysis_tab2 = st.tabs(["Single Post", "Batch Analysis"])
        
        with analysis_tab1:
            st.subheader("Analyze Individual Post")
            
            if len(df) > 0 and 'Post Content' in df.columns:
                post_options = list(range(min(100, len(df))))
                selected_idx = st.selectbox(
                    "Select post:", 
                    post_options,
                    format_func=lambda x: f"Post {x+1}: {df.iloc[x]['Post Content'][:60]}..."
                )
                
                selected_text = df.iloc[selected_idx]['Post Content']
                st.text_area("Selected Post:", selected_text, height=100)
                
                if st.button("🔍 Analyze with Caching"):
                    st.subheader("Analysis Results")
                    
                    results = []
                    
                    with st.spinner("Analyzing (checking cache first)..."):
                        
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
                    
                    # Display results
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
            
            sample_size = st.slider("Number of posts to analyze:", 10, min(200, len(df)), 50)
            
            if st.button("🚀 Run Cached Batch Analysis"):
                
                sample_df = df.sample(n=sample_size, random_state=42)
                results_list = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (_, row) in enumerate(sample_df.iterrows()):
                    progress = (i + 1) / len(sample_df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing post {i+1}/{len(sample_df)}...")
                    
                    if 'Post Content' in row:
                        text = row['Post Content']
                        
                        # Use enabled model
                        model_to_use = None
                        if use_llama3 and classifier.models['llama3']:
                            model_to_use = 'llama3'
                        elif use_openai and classifier.models['openai']:
                            model_to_use = 'openai'
                        elif use_anthropic and classifier.models['anthropic']:
                            model_to_use = 'anthropic'
                        
                        if model_to_use:
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
                    
                    time.sleep(0.02)
                
                progress_bar.empty()
                status_text.empty()
                
                if results_list:
                    results_df = pd.DataFrame(results_list)
                    st.session_state['batch_results'] = results_df
                    st.success(f"✅ Analyzed {len(results_df)} posts successfully!")
    
    with tab3:
        st.header("📈 Analysis Results")
        
        if 'batch_results' in st.session_state:
            results_df = st.session_state['batch_results']
            
            # Basic visualizations
            col1, col2 = st.columns(2)
            
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
            
            # Results table
            st.subheader("📊 Detailed Results")
            st.dataframe(
                results_df[['post_content', 'age_group', 'confidence_level', 'confidence_score', 'platform', 'likes', 'comments']],
                use_container_width=True
            )
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"social_media_classification_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("👆 Run batch analysis to see results here")
    
    with tab4:
        st.header("🔬 Model Evaluation")
        
        if 'batch_results' not in st.session_state:
            st.warning("Please run batch analysis first to generate evaluation data.")
            return
        
        results_df = st.session_state['batch_results']
        
        # Create ground truth
        st.subheader("📋 Ground Truth Generation")
        st.info("💡 **Note**: In production, you'd have human-annotated labels. This demo uses heuristic-based ground truth for demonstration.")
        
        if st.button("🎯 Generate Ground Truth Labels"):
            with st.spinner("Generating synthetic ground truth labels..."):
                # Create ground truth using heuristics
                ground_truth_df = classifier.create_synthetic_ground_truth(
                    st.session_state.df, 
                    sample_size=len(results_df)
                )
                
                st.session_state['ground_truth'] = ground_truth_df
                st.success(f"✅ Generated ground truth for {len(ground_truth_df)} posts")
        
        if 'ground_truth' in st.session_state:
            ground_truth_df = st.session_state['ground_truth']
            
            # Run evaluation
            if st.button("📊 Evaluate Model Performance"):
                with st.spinner("Calculating evaluation metrics..."):
                    evaluation_results = classifier.evaluate_model_performance(
                        results_df, 
                        ground_truth_df
                    )
                    
                    st.session_state['evaluation_results'] = evaluation_results
                    
                    if 'error' not in evaluation_results:
                        st.success(f"✅ Evaluation complete! Analyzed {evaluation_results['sample_size']} posts.")
            
            # Display evaluation results
            if 'evaluation_results' in st.session_state:
                display_evaluation_dashboard(st.session_state['evaluation_results'])
    
    with tab5:
        st.header("🏆 Multi-Model Performance Comparison")
        
        st.info("💡 **To compare models**: Run batch analysis with different models enabled and evaluate each one separately.")
        
        # Example of how to store multiple model results
        if 'model_evaluations' not in st.session_state:
            st.session_state['model_evaluations'] = {}
        
        # Option to save current evaluation
        if 'evaluation_results' in st.session_state:
            evaluation_results = st.session_state['evaluation_results']
            
            if 'error' not in evaluation_results:
                current_model = st.selectbox(
                    "Save current evaluation as:",
                    ["OpenAI GPT-3.5", "Anthropic Claude", "Llama 3", "Custom Model"]
                )
                
                if st.button(f"💾 Save {current_model} Results"):
                    st.session_state['model_evaluations'][current_model] = evaluation_results
                    st.success(f"✅ Saved evaluation results for {current_model}")
        
        # Display comparison if multiple models available
        if len(st.session_state['model_evaluations']) > 0:
            st.subheader("📊 Saved Model Evaluations")
            
            for model_name in st.session_state['model_evaluations'].keys():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_{model_name}"):
                        del st.session_state['model_evaluations'][model_name]
                        st.experimental_rerun()
            
            if len(st.session_state['model_evaluations']) >= 2:
                compare_model_performance(st.session_state['model_evaluations'])
        else:
            st.info("No saved model evaluations yet. Complete an evaluation in the Evaluation tab first.")
    
    # Footer
    st.markdown("---")
    cache_stats = classifier.cache.get_cache_stats()
    st.markdown(f"📊 **AI Classifier with Metrics** | Cache Hit: {cache_stats['api_hit_rate']:.1f}% | Cost Saved: ${cache_stats['cost_saved']:.3f}")

if __name__ == "__main__":
    main()
