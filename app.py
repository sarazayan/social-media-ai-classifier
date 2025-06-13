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
import os

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
        self.openai_method = None  # Track which OpenAI method worked

    def clear_proxy_settings(self):
        """Clear any proxy settings that might interfere with API calls"""
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        for var in proxy_vars:
            os.environ.pop(var, None)
    
    def _make_openai_request(self, api_key: str, **kwargs):
        """Manual OpenAI API request for cases where the client fails"""
        import requests
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        if 'messages' in kwargs:
            # Chat completion request
            data = {
                'model': kwargs.get('model', 'gpt-3.5-turbo'),
                'messages': kwargs['messages'],
                'temperature': kwargs.get('temperature', 0.1),
                'max_tokens': kwargs.get('max_tokens', 300)
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        else:
            raise Exception("Unsupported request type")

    def setup_apis(self, openai_key: str = None, anthropic_key: str = None, llama3_key: str = None):
        """Enhanced API setup with targeted proxies fix"""
        
        if openai_key and OPENAI_AVAILABLE:
            self.clear_proxy_settings()  # Clear proxy settings first
            success = False
            
            # Check OpenAI version first
            try:
                openai_version = openai.__version__
                st.info(f"🔍 Detected OpenAI version: {openai_version}")
            except:
                openai_version = "unknown"
            
            # Method 1: Ultra-basic approach (no optional parameters)
            if not success:
                try:
                    # Create client with absolute minimum parameters
                    client = openai.OpenAI()
                    client.api_key = openai_key.strip()
                    # Test the connection
                    client.models.list()
                    self.models['openai'] = client
                    self.openai_method = 'basic'
                    st.success("✅ OpenAI API connected successfully (basic method)")
                    success = True
                except Exception as e:
                    st.warning(f"Basic method failed: {e}")
            
            # Method 2: Environment variable only
            if not success:
                try:
                    os.environ['OPENAI_API_KEY'] = openai_key.strip()
                    # Remove any proxy-related environment variables
                    for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
                        os.environ.pop(proxy_var, None)
                    
                    # Create client without any parameters that might cause issues
                    self.models['openai'] = openai.OpenAI()
                    # Test the connection
                    self.models['openai'].models.list()
                    self.openai_method = 'env'
                    st.success("✅ OpenAI API connected successfully (environment method)")
                    success = True
                except Exception as e:
                    st.warning(f"Environment method failed: {e}")
            
            # Method 3: Direct attribute setting (for newer versions)
            if not success:
                try:
                    client = openai.OpenAI.__new__(openai.OpenAI)
                    client.__dict__.update({
                        'api_key': openai_key.strip(),
                        '_base_url': 'https://api.openai.com/v1',
                        '_timeout': 60,
                        '_max_retries': 2
                    })
                    # Initialize the client
                    openai.OpenAI.__init__(client, api_key=openai_key.strip())
                    # Test the connection
                    client.models.list()
                    self.models['openai'] = client
                    self.openai_method = 'direct'
                    st.success("✅ OpenAI API connected successfully (direct method)")
                    success = True
                except Exception as e:
                    st.warning(f"Direct method failed: {e}")
            
            # Method 4: Legacy OpenAI (for 0.x versions)
            if not success and openai_version.startswith('0.'):
                try:
                    # Legacy version - completely different approach
                    openai.api_key = openai_key.strip()
                    # Clear any base that might have proxy settings
                    if hasattr(openai, 'api_base'):
                        openai.api_base = 'https://api.openai.com/v1'
                    # Test with a simple call
                    openai.Model.list()
                    self.models['openai'] = openai
                    self.openai_method = 'legacy'
                    st.success("✅ OpenAI API connected successfully (legacy v0.x)")
                    success = True
                except Exception as e:
                    st.warning(f"Legacy method failed: {e}")
            
            # Method 5: Manual client creation (bypass constructor completely)
            if not success:
                try:
                    # Create a minimal client object manually
                    import types
                    client = types.SimpleNamespace()
                    client.api_key = openai_key.strip()
                    client.base_url = 'https://api.openai.com/v1'
                    
                    # Try to use the openai module functions directly
                    import openai
                    original_api_key = getattr(openai, 'api_key', None)
                    openai.api_key = openai_key.strip()
                    
                    # Test if we can make a simple request
                    if hasattr(openai, 'Model'):
                        openai.Model.list()
                        self.models['openai'] = openai
                        self.openai_method = 'module'
                        st.success("✅ OpenAI API connected successfully (module method)")
                        success = True
                    else:
                        # For newer versions, try direct API call
                        import requests
                        headers = {
                            'Authorization': f'Bearer {openai_key.strip()}',
                            'Content-Type': 'application/json'
                        }
                        response = requests.get('https://api.openai.com/v1/models', headers=headers, timeout=10)
                        if response.status_code == 200:
                            client.make_request = lambda **kwargs: self._make_openai_request(openai_key.strip(), **kwargs)
                            self.models['openai'] = client
                            self.openai_method = 'manual'
                            st.success("✅ OpenAI API connected successfully (manual method)")
                            success = True
                        else:
                            raise Exception(f"API test failed with status {response.status_code}")
                except Exception as e:
                    st.warning(f"Manual method failed: {e}")
            
            if not success:
                st.error("❌ All OpenAI connection methods failed.")
                st.error("🔍 This suggests either:")
                st.error("   • Invalid API key")
                st.error("   • Network connectivity issues") 
                st.error("   • Incompatible OpenAI library version")
                st.info("💡 Try: pip uninstall openai && pip install openai==1.3.7")
                st.info("💡 Or check your API key at: https://platform.openai.com/api-keys")
        
        if anthropic_key and ANTHROPIC_AVAILABLE:
            try:
                self.models['anthropic'] = anthropic.Anthropic(api_key=anthropic_key.strip())
                st.success("✅ Anthropic API connected successfully")
            except Exception as e:
                st.error(f"❌ Anthropic setup failed: {e}")
        
        if llama3_key:
            try:
                self.models['llama3'] = llama3_key.strip()
                st.success("✅ Llama 3 endpoint configured")
            except Exception as e:
                st.error(f"❌ Llama 3 setup failed: {e}")

    def create_classification_prompt(self, text: str, model_type: str) -> str:
        base_prompt = f"""You are an expert social media analyst. Analyze this post and classify the author.

POST TO ANALYZE: "{text}"

TASK 1 - AGE GROUP (look for these specific indicators):
- TEENS (13-19): "school", "homework", "class", "teacher", "omg", "literally", "high school"
- YOUNG_ADULTS (20-30): "college", "job", "career", "interview", "apartment", "university", "dating"  
- ADULTS (31-55): "kids", "children", "family", "parenting", "mortgage", "work", "business"
- SENIORS (55+): "retirement", "grandchildren", "health", "retired", "years ago", "back then"

TASK 2 - CONFIDENCE LEVEL (look for these exact phrases and words):
- LOW confidence: "not good enough", "terrible at", "everyone else is better", "don't know", "confused", "scared", "awful", "struggling", "worse than", "can't do"
- HIGH confidence: "confident", "excellent", "proud", "sure", "great at", "amazing", "definitely", "successful", "love doing", "fantastic"  
- MEDIUM confidence: "sometimes", "usually", "okay", "decent", "learning", "pretty good", "depends", "mixed feelings"

CRITICAL INSTRUCTIONS:
1. Read the post carefully for EXACT words and phrases above
2. If you see LOW confidence words → classify as "low"
3. If you see HIGH confidence words → classify as "high"  
4. If unclear or neutral → classify as "medium"
5. For age, match the strongest indicators present

RESPOND IN EXACTLY THIS FORMAT:
Age Group: [teens/young_adults/adults/seniors]
Confidence Level: [low/medium/high]
Reasoning: [Explain which specific words/phrases you found]
Score: [0.8]"""

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
            
            # Handle different OpenAI client types
            if self.openai_method == 'legacy':
                # Legacy OpenAI v0.x
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                content = response.choices[0].message.content.strip()
            
            elif self.openai_method == 'module':
                # Using openai module directly
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                content = response['choices'][0]['message']['content'].strip()
            
            elif self.openai_method == 'manual':
                # Manual request method
                response_data = self.models['openai'].make_request(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                content = response_data['choices'][0]['message']['content'].strip()
            
            else:
                # Modern OpenAI v1.x (basic, env, direct methods)
                response = self.models['openai'].chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                content = response.choices[0].message.content.strip()
            
            self.api_calls['openai'] += 1
            self.costs['openai'] += 0.002
            
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
        """Create ground truth using enhanced keyword heuristics"""
        
        sample_df = df.copy().reset_index(drop=True)
        sample_df['true_age_group'] = 'unknown'
        sample_df['true_confidence_level'] = 'unknown'
        
        for idx, row in sample_df.iterrows():
            text = str(row[content_column]).lower()
            
            # Age group rules
            if any(word in text for word in ['school', 'homework', 'class', 'omg', 'literally', 'teen', 'high school', 'grade', 'teacher', 'exam', 'student']):
                sample_df.at[idx, 'true_age_group'] = 'teens'
            elif any(word in text for word in ['college', 'job', 'career', 'apartment', 'university', 'interview', 'graduate', 'dating', 'single', 'relationship']):
                sample_df.at[idx, 'true_age_group'] = 'young_adults'
            elif any(word in text for word in ['kids', 'family', 'mortgage', 'parenting', 'children', 'marriage', 'spouse', 'work', 'business', 'project']):
                sample_df.at[idx, 'true_age_group'] = 'adults'
            elif any(word in text for word in ['retirement', 'grandchildren', 'health', 'retired', 'doctor', 'medication', 'years ago', 'back then']):
                sample_df.at[idx, 'true_age_group'] = 'seniors'
            else:
                sample_df.at[idx, 'true_age_group'] = 'young_adults'
            
            # Enhanced confidence rules
            low_confidence_phrases = [
                'not good enough', 'terrible at', 'awful at', 'bad at', 'struggling with',
                'everyone else', 'better than me', 'smarter than', 'worse than',
                'scared', 'worried', 'confused', 'lost', 'dont know', "don't know",
                'not sure', 'uncertain', 'anxious', 'nervous', 'overwhelmed',
                'failing', 'useless', 'hopeless', 'can\'t do', 'unable to'
            ]
            
            high_confidence_phrases = [
                'confident', 'excellent at', 'great at', 'good at', 'proud of',
                'sure about', 'certain', 'definitely', 'absolutely', 'successful',
                'accomplished', 'achieved', 'capable', 'skilled', 'talented',
                'amazing', 'fantastic', 'outstanding', 'brilliant', 'perfect',
                'love doing', 'passionate about', 'excited about'
            ]
            
            medium_confidence_phrases = [
                'sometimes', 'usually', 'often', 'okay with', 'decent at',
                'learning', 'improving', 'getting better', 'working on',
                'mixed feelings', 'ups and downs', 'depends', 'varies',
                'pretty good', 'not bad', 'alright', 'fine with'
            ]
            
            # Check for confidence indicators
            confidence_assigned = False
            
            for phrase in low_confidence_phrases:
                if phrase in text:
                    sample_df.at[idx, 'true_confidence_level'] = 'low'
                    confidence_assigned = True
                    break
            
            if not confidence_assigned:
                for phrase in high_confidence_phrases:
                    if phrase in text:
                        sample_df.at[idx, 'true_confidence_level'] = 'high'
                        confidence_assigned = True
                        break
            
            if not confidence_assigned:
                for phrase in medium_confidence_phrases:
                    if phrase in text:
                        sample_df.at[idx, 'true_confidence_level'] = 'medium'
                        confidence_assigned = True
                        break
            
            # Fallback to individual words
            if not confidence_assigned:
                if any(word in text for word in ['terrible', 'awful', 'scared', 'worried', 'confused', 'lost', 'failing']):
                    sample_df.at[idx, 'true_confidence_level'] = 'low'
                elif any(word in text for word in ['confident', 'excellent', 'proud', 'sure', 'great', 'amazing', 'successful']):
                    sample_df.at[idx, 'true_confidence_level'] = 'high'
                else:
                    sample_df.at[idx, 'true_confidence_level'] = 'medium'
        
        return sample_df

    def calculate_accuracy(self, results_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        """Calculate accuracy metrics without any external libraries"""
        
        if len(results_df) == 0:
            return {'error': 'No results to evaluate'}
        
        if len(ground_truth_df) == 0:
            return {'error': 'No ground truth data available'}
        
        min_length = min(len(results_df), len(ground_truth_df))
        
        if min_length == 0:
            return {'error': 'No data to compare'}
        
        results_df = results_df.reset_index(drop=True).head(min_length)
        ground_truth_df = ground_truth_df.reset_index(drop=True).head(min_length)
        
        try:
            age_correct = (results_df['age_group'] == ground_truth_df['true_age_group']).sum()
            conf_correct = (results_df['confidence_level'] == ground_truth_df['true_confidence_level']).sum()
            total = min_length
            
            age_accuracy = age_correct / total if total > 0 else 0
            conf_accuracy = conf_correct / total if total > 0 else 0
            
            return {
                'sample_size': total,
                'age_accuracy': age_accuracy,
                'confidence_accuracy': conf_accuracy,
                'age_correct': age_correct,
                'conf_correct': conf_correct
            }
        except Exception as e:
            return {'error': f'Error calculating accuracy: {str(e)}'}

@st.cache_data(ttl=1800)
def load_data_cached(uploaded_file=None) -> pd.DataFrame:
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            sample_data = [
                {'Post ID': 1, 'Platform': 'Twitter', 'Post Content': 'omg school is literally so hard everyone else gets it but i dont understand math at all', 'Likes': 23, 'Comments': 5},
                {'Post ID': 2, 'Platform': 'Instagram', 'Post Content': 'just aced my history exam feeling confident about my grades this semester', 'Likes': 45, 'Comments': 8},
                {'Post ID': 3, 'Platform': 'TikTok', 'Post Content': 'homework is okay sometimes hard sometimes easy depends on the subject', 'Likes': 67, 'Comments': 12},
                {'Post ID': 4, 'Platform': 'LinkedIn', 'Post Content': 'job interview tomorrow feeling confident about my career goals and skills', 'Likes': 34, 'Comments': 6},
                {'Post ID': 5, 'Platform': 'Twitter', 'Post Content': 'not sure if college is right for me everyone seems more prepared than me', 'Likes': 12, 'Comments': 3},
                {'Post ID': 6, 'Platform': 'Instagram', 'Post Content': 'university life is usually pretty good learning a lot about myself', 'Likes': 28, 'Comments': 7},
                {'Post ID': 7, 'Platform': 'Facebook', 'Post Content': 'parenting is so hard everyone else seems like better parents than me', 'Likes': 15, 'Comments': 9},
                {'Post ID': 8, 'Platform': 'LinkedIn', 'Post Content': 'proud of how well our family project turned out great teamwork with the kids', 'Likes': 52, 'Comments': 14},
                {'Post ID': 9, 'Platform': 'Facebook', 'Post Content': 'work life balance is challenging but generally managing okay most days', 'Likes': 31, 'Comments': 8},
                {'Post ID': 10, 'Platform': 'Facebook', 'Post Content': 'retirement planning has been confusing dont know if im doing it right', 'Likes': 19, 'Comments': 4},
                {'Post ID': 11, 'Platform': 'Facebook', 'Post Content': 'absolutely love spending time with grandchildren feeling blessed and grateful', 'Likes': 43, 'Comments': 11},
                {'Post ID': 12, 'Platform': 'LinkedIn', 'Post Content': 'health is pretty decent for my age some good days some not so good', 'Likes': 27, 'Comments': 6},
                {'Post ID': 13, 'Platform': 'Instagram', 'Post Content': 'terrible at cooking everything i make turns out awful compared to others', 'Likes': 8, 'Comments': 2},
                {'Post ID': 14, 'Platform': 'Twitter', 'Post Content': 'excellent presentation today definitely nailed it feeling accomplished', 'Likes': 67, 'Comments': 15},
                {'Post ID': 15, 'Platform': 'Facebook', 'Post Content': 'kids are growing up so fast sometimes proud sometimes worried about choices', 'Likes': 29, 'Comments': 7},
                {'Post ID': 16, 'Platform': 'TikTok', 'Post Content': 'omg literally everyone at school is smarter than me feeling so lost', 'Likes': 14, 'Comments': 3},
                {'Post ID': 17, 'Platform': 'Instagram', 'Post Content': 'career goals are clear and im confident about my professional path ahead', 'Likes': 38, 'Comments': 9},
                {'Post ID': 18, 'Platform': 'LinkedIn', 'Post Content': 'business project went well usually pretty good at managing teams', 'Likes': 45, 'Comments': 12},
                {'Post ID': 19, 'Platform': 'Facebook', 'Post Content': 'grandchildren visited yesterday absolutely amazing watching them grow up', 'Likes': 56, 'Comments': 18},
                {'Post ID': 20, 'Platform': 'Twitter', 'Post Content': 'college applications are overwhelming everyone else seems more qualified', 'Likes': 11, 'Comments': 4},
                {'Post ID': 21, 'Platform': 'Instagram', 'Post Content': 'family vacation planning going okay decent at organizing these things', 'Likes': 33, 'Comments': 8},
                {'Post ID': 22, 'Platform': 'Facebook', 'Post Content': 'retirement savings looking good feeling secure about financial future', 'Likes': 41, 'Comments': 10},
                {'Post ID': 23, 'Platform': 'TikTok', 'Post Content': 'school dance was amazing had such a great time feeling fantastic', 'Likes': 73, 'Comments': 22},
                {'Post ID': 24, 'Platform': 'LinkedIn', 'Post Content': 'job search is tough not sure what employers want feeling uncertain', 'Likes': 16, 'Comments': 5},
                {'Post ID': 25, 'Platform': 'Facebook', 'Post Content': 'parenting teenagers is usually challenging but sometimes very rewarding', 'Likes': 39, 'Comments': 13}
            ]
            df = pd.DataFrame(sample_data)
        
        # Clean the DataFrame
        if df.empty:
            return df
        
        df = df.reset_index(drop=True)
        
        # Ensure proper data types
        numeric_columns = ['Post ID', 'Likes', 'Comments']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        text_columns = ['Post Content', 'Platform']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('')
        
        # Remove empty content
        content_cols = [col for col in df.columns if 'content' in col.lower()]
        if content_cols:
            content_col = content_cols[0]
            df = df[df[content_col].str.strip() != '']
            df = df[df[content_col] != 'nan']
            df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame([{
            'Post ID': 1,
            'Platform': 'Twitter',
            'Post Content': 'This is a test post for debugging',
            'Likes': 10,
            'Comments': 2
        }])

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
    st.sidebar.header("🔑 API Configuration")
    
    # Show library versions
    with st.sidebar.expander("📋 System Info"):
        if OPENAI_AVAILABLE:
            try:
                st.write(f"OpenAI: {openai.__version__}")
            except:
                st.write("OpenAI: Version unknown")
        else:
            st.write("OpenAI: Not installed")
        
        if ANTHROPIC_AVAILABLE:
            try:
                st.write(f"Anthropic: {anthropic.__version__}")
            except:
                st.write("Anthropic: Version unknown")
        else:
            st.write("Anthropic: Not installed")
        
        st.write(f"Requests: {'Available' if REQUESTS_AVAILABLE else 'Not available'}")
        
        # Debug info for proxy issues
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        active_proxies = [f"{var}: {os.environ.get(var, 'None')}" for var in proxy_vars if os.environ.get(var)]
        if active_proxies:
            st.write("⚠️ Active proxy settings:")
            for proxy in active_proxies:
                st.write(f"  {proxy}")
        else:
            st.write("✅ No proxy settings detected")
    
    # Debug section for troubleshooting
    with st.sidebar.expander("🐛 Debug Tools"):
        if st.button("Clear All Proxy Settings"):
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
            cleared = []
            for var in proxy_vars:
                if var in os.environ:
                    os.environ.pop(var)
                    cleared.append(var)
            if cleared:
                st.success(f"Cleared: {', '.join(cleared)}")
            else:
                st.info("No proxy settings to clear")
        
        if st.button("Test OpenAI Import"):
            try:
                import openai
                st.success(f"✅ OpenAI imported successfully: {openai.__version__}")
                
                # Test basic client creation without any parameters
                try:
                    test_client = openai.OpenAI.__new__(openai.OpenAI)
                    st.success("✅ OpenAI client object created")
                except Exception as e:
                    st.error(f"❌ Client creation failed: {e}")
                    
            except Exception as e:
                st.error(f"❌ OpenAI import failed: {e}")
    
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key")
    llama3_key = st.sidebar.text_input("Together AI API Key", type="password", help="Enter your Together AI API key for Llama 3")
    
    if st.sidebar.button("🔗 Connect APIs"):
        classifier.setup_apis(openai_key, anthropic_key, llama3_key)
    
    # Cache stats
    stats = classifier.cache.get_cache_stats()
    st.sidebar.metric("📊 Cache Hit Rate", f"{stats['api_hit_rate']:.1f}%")
    st.sidebar.metric("💰 Cost Saved", f"${stats['cost_saved']:.3f}")
    
    # Model selection
    st.sidebar.header("🤖 Model Selection")
    use_openai = st.sidebar.checkbox("OpenAI GPT-3.5", value=True, disabled=not classifier.models['openai'])
    use_anthropic = st.sidebar.checkbox("Anthropic Claude", value=False, disabled=not classifier.models['anthropic'])
    use_llama3 = st.sidebar.checkbox("Llama 3", value=False, disabled=not classifier.models['llama3'])
    
    # Show connection status
    if classifier.models['openai']:
        st.sidebar.success("✅ OpenAI Connected")
    if classifier.models['anthropic']:
        st.sidebar.success("✅ Anthropic Connected")
    if classifier.models['llama3']:
        st.sidebar.success("✅ Llama 3 Connected")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🔍 Analysis", "📈 Results", "📋 Evaluation"])
    
    with tab1:
        st.header("📊 Dataset Management")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv", help="Upload a CSV file with social media posts")
        df = load_data_cached(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} posts")
        st.dataframe(df.head(), use_container_width=True)
        
        # Show column info
        st.info(f"📋 Columns: {', '.join(df.columns.tolist())}")
        
        st.session_state.df = df
    
    with tab2:
        st.header("🔍 AI Classification Analysis")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Please load data first in the Dataset tab")
            return
        
        df = st.session_state.df
        
        # Find content column
        content_column = None
        for col in df.columns:
            if 'content' in col.lower():
                content_column = col
                break
        
        if not content_column:
            st.error("❌ No content column found. Make sure your CSV has a column with 'content' in the name.")
            return
        
        st.info(f"📝 Using column: **{content_column}**")
        
        # Single post analysis
        st.subheader("🎯 Single Post Analysis")
        
        post_idx = st.selectbox("Select post:", range(min(len(df), 100)), format_func=lambda x: f"Post {x+1}: {str(df.iloc[x][content_column])[:50]}...")
        
        if post_idx >= len(df):
            st.error("❌ Selected post index out of range")
            return
            
        selected_text = str(df.iloc[post_idx][content_column])
        st.text_area("📝 Post content:", selected_text, height=100)
        
        if st.button("🚀 Analyze Post"):
            if not any([use_openai and classifier.models['openai'], 
                       use_anthropic and classifier.models['anthropic'], 
                       use_llama3 and classifier.models['llama3']]):
                st.error("❌ Please connect and select at least one model")
                return
            
            results = []
            
            with st.spinner("🔄 Analyzing..."):
                if use_openai and classifier.models['openai']:
                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'openai'))
                    if 'error' not in result:
                        results.append(result)
                    else:
                        st.error(f"OpenAI Error: {result['error']}")
                
                if use_anthropic and classifier.models['anthropic']:
                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'anthropic'))
                    if 'error' not in result:
                        results.append(result)
                    else:
                        st.error(f"Anthropic Error: {result['error']}")
                
                if use_llama3 and classifier.models['llama3']:
                    result = asyncio.run(classifier.classify_with_caching(selected_text, 'llama3'))
                    if 'error' not in result:
                        results.append(result)
                    else:
                        st.error(f"Llama 3 Error: {result['error']}")
            
            for result in results:
                st.subheader(f"🤖 {result['model']} Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👥 Age Group", result['age_group'].replace('_', ' ').title())
                with col2:
                    st.metric("💪 Confidence Level", result['confidence_level'].title())
                with col3:
                    st.metric("🎯 AI Confidence", f"{result['confidence_score']:.2f}")
                
                st.write("**🧠 Reasoning:**", result['reasoning'])
                
                if st.checkbox(f"Show raw response ({result['model']})", key=f"raw_{result['model']}"):
                    st.code(result['raw_response'])
        
        # Batch analysis
        st.subheader("📦 Batch Analysis")
        
        sample_size = st.slider("📊 Posts to analyze:", 1, min(100, len(df)), min(20, len(df)))
        
        if st.button("🚀 Run Batch Analysis"):
            if not any([use_openai and classifier.models['openai'], 
                       use_anthropic and classifier.models['anthropic'], 
                       use_llama3 and classifier.models['llama3']]):
                st.error("❌ Please connect and select at least one model")
                return
                
            max_available = len(df)
            actual_sample_size = min(sample_size, max_available)
            
            if actual_sample_size < sample_size:
                st.warning(f"⚠️ Requested {sample_size} posts, but only {max_available} available. Using {actual_sample_size} posts.")
            
            if actual_sample_size == 0:
                st.error("❌ No data available for analysis")
                return
            
            try:
                df_clean = df.copy().reset_index(drop=True)
                df_clean = df_clean.dropna(subset=[content_column])
                
                if len(df_clean) == 0:
                    st.error("❌ No valid posts found after cleaning data")
                    return
                
                actual_sample_size = min(actual_sample_size, len(df_clean))
                
                if actual_sample_size >= len(df_clean):
                    sample_df = df_clean.copy()
                    st.info(f"ℹ️ Using all {len(df_clean)} available posts (no sampling needed)")
                else:
                    try:
                        sample_df = df_clean.sample(n=actual_sample_size, random_state=42, replace=False)
                    except ValueError as ve:
                        st.error(f"❌ Sampling error: {ve}")
                        sample_df = df_clean.head(actual_sample_size)
                        st.warning("⚠️ Using first rows instead of random sample due to sampling issue")
                    except Exception as e:
                        st.error(f"❌ Unexpected sampling error: {e}")
                        return
                
            except Exception as e:
                st.error(f"❌ Error preparing data: {e}")
                return
            
            results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Choose model
            model_to_use = None
            if use_llama3 and classifier.models['llama3']:
                model_to_use = 'llama3'
            elif use_openai and classifier.models['openai']:
                model_to_use = 'openai'
            elif use_anthropic and classifier.models['anthropic']:
                model_to_use = 'anthropic'
            
            if not model_to_use:
                st.error("❌ Please connect and select at least one model")
                return
            
            st.info(f"🤖 Using {model_to_use.upper()} for batch analysis")
            
            for i, (_, row) in enumerate(sample_df.iterrows()):
                progress_bar.progress((i + 1) / len(sample_df))
                status_text.text(f"Processing post {i+1}/{len(sample_df)}")
                
                text = row[content_column]
                result = asyncio.run(classifier.classify_with_caching(text, model_to_use))
                
                if 'error' not in result:
                    result.update({
                        'original_index': row.name,
                        'post_content': text[:100] + '...' if len(text) > 100 else text
                    })
                    results_list.append(result)
                else:
                    st.warning(f"⚠️ Error processing post {i+1}: {result['error']}")
                
                time.sleep(0.02)
            
            progress_bar.empty()
            status_text.empty()
            
            if results_list:
                results_df = pd.DataFrame(results_list)
                st.session_state['batch_results'] = results_df
                st.success(f"✅ Successfully analyzed {len(results_df)} posts")
            else:
                st.error("❌ No successful results from batch analysis")
    
    with tab3:
        st.header("📈 Analysis Results")
        
        if 'batch_results' not in st.session_state:
            st.info("ℹ️ Run batch analysis first in the Analysis tab")
            return
        
        results_df = st.session_state['batch_results']
        
        if len(results_df) == 0:
            st.error("❌ No results found. Please run batch analysis first.")
            return
        
        # Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Posts Analyzed", len(results_df))
        with col2:
            try:
                most_common_age = results_df['age_group'].mode()[0] if len(results_df) > 0 else 'Unknown'
            except:
                most_common_age = results_df['age_group'].value_counts().index[0] if len(results_df) > 0 else 'Unknown'
            st.metric("👥 Most Common Age", most_common_age.replace('_', ' ').title())
        with col3:
            try:
                most_common_conf = results_df['confidence_level'].mode()[0] if len(results_df) > 0 else 'Unknown'
            except:
                most_common_conf = results_df['confidence_level'].value_counts().index[0] if len(results_df) > 0 else 'Unknown'
            st.metric("💪 Most Common Confidence", most_common_conf.title())
        
        # Visualizations
        if len(results_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                age_dist = results_df['age_group'].value_counts()
                if len(age_dist) > 0:
                    fig_age = px.pie(values=age_dist.values, names=age_dist.index, title='👥 Age Groups Distribution')
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("ℹ️ No age group data to display")
            
            with col2:
                conf_dist = results_df['confidence_level'].value_counts()
                if len(conf_dist) > 0:
                    fig_conf = px.pie(values=conf_dist.values, names=conf_dist.index, title='💪 Confidence Levels Distribution')
                    st.plotly_chart(fig_conf, use_container_width=True)
                else:
                    st.info("ℹ️ No confidence level data to display")
        else:
            st.warning("⚠️ No data available for visualization")
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        if len(results_df) > 0:
            available_columns = []
            desired_columns = ['post_content', 'age_group', 'confidence_level', 'confidence_score']
            
            for col in desired_columns:
                if col in results_df.columns:
                    available_columns.append(col)
            
            if available_columns:
                st.dataframe(results_df[available_columns], use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )
            else:
                st.error("❌ No valid columns found in results")
        else:
            st.warning("⚠️ No results to display")
    
    with tab4:
        st.header("📋 Model Evaluation")
        
        if 'batch_results' not in st.session_state:
            st.warning("⚠️ Run batch analysis first in the Analysis tab")
            return
        
        results_df = st.session_state['batch_results']
        original_df = st.session_state['df']
        
        if len(results_df) == 0:
            st.error("❌ No results found. Please run batch analysis first.")
            return
        
        # Find content column
        content_column = None
        for col in original_df.columns:
            if 'content' in col.lower():
                content_column = col
                break
        
        st.info("ℹ️ This demo uses keyword-based ground truth. In production, use human-annotated labels.")
        
        if st.button("🎯 Generate Ground Truth"):
            with st.spinner("🔄 Generating ground truth labels..."):
                try:
                    if len(results_df) == 0:
                        st.error("❌ No results to generate ground truth for")
                        return
                    
                    original_indices = results_df.get('original_index', range(len(results_df)))
                    
                    try:
                        subset_df = original_df.iloc[original_indices].copy()
                    except:
                        subset_df = original_df.head(len(results_df)).copy()
                    
                    ground_truth_df = classifier.create_ground_truth(subset_df, content_column)
                    st.session_state['ground_truth'] = ground_truth_df
                    st.success(f"✅ Ground truth generated for {len(ground_truth_df)} posts")
                    
                except Exception as e:
                    st.error(f"❌ Error generating ground truth: {str(e)}")
                    st.info("💡 Try running batch analysis again first")
        
        if 'ground_truth' in st.session_state:
            ground_truth_df = st.session_state['ground_truth']
            
            if st.button("📊 Calculate Evaluation Metrics"):
                with st.spinner("🔄 Calculating metrics..."):
                    eval_results = classifier.calculate_accuracy(results_df, ground_truth_df)
                    
                    if 'error' in eval_results:
                        st.error(f"❌ {eval_results['error']}")
                    else:
                        st.session_state['eval_results'] = eval_results
                        st.success("✅ Evaluation complete")
            
            if 'eval_results' in st.session_state:
                eval_results = st.session_state['eval_results']
                
                st.subheader("📊 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Sample Size", eval_results['sample_size'])
                
                with col2:
                    age_acc = eval_results['age_accuracy']
                    st.metric("👥 Age Accuracy", f"{age_acc:.3f}")
                
                with col3:
                    conf_acc = eval_results['confidence_accuracy']
                    st.metric("💪 Confidence Accuracy", f"{conf_acc:.3f}")
                
                with col4:
                    overall = (age_acc + conf_acc) / 2
                    st.metric("🎯 Overall Score", f"{overall:.3f}")
                
                # Performance breakdown
                st.subheader("📋 Detailed Results")
                
                st.write(f"**👥 Age Group Classification:**")
                st.write(f"- ✅ Correct predictions: {eval_results['age_correct']}/{eval_results['sample_size']}")
                st.write(f"- 📊 Accuracy: {eval_results['age_accuracy']:.1%}")
                
                st.write(f"**💪 Confidence Level Classification:**")
                st.write(f"- ✅ Correct predictions: {eval_results['conf_correct']}/{eval_results['sample_size']}")
                st.write(f"- 📊 Accuracy: {eval_results['confidence_accuracy']:.1%}")
                
                if overall >= 0.8:
                    st.success("🎉 Excellent performance! The model is working very well.")
                elif overall >= 0.6:
                    st.info("👍 Good performance. Consider fine-tuning prompts for improvement.")
                else:
                    st.warning("⚠️ Performance could be improved. Review prompt engineering and ground truth quality.")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **Social Media AI Classifier** - Enhanced with improved error handling and multiple API connection methods")

if __name__ == "__main__":
    main()
