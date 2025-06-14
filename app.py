import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import time
import re
from datetime import datetime

# Simple API clients with minimal setup
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

class SimpleClassifier:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.api_calls = 0
        
    def setup_openai(self, api_key: str):
        """Simple OpenAI setup - no complex fallbacks"""
        if not OPENAI_AVAILABLE:
            st.error("OpenAI library not installed. Run: pip install openai")
            return False
            
        try:
            # Simple, direct initialization
            self.openai_client = openai.OpenAI(api_key=api_key)
            
            # Test the connection with a simple call
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            st.success("✅ OpenAI connected successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ OpenAI setup failed: {str(e)}")
            st.info("💡 Make sure you have a valid API key and sufficient credits")
            return False
    
    def setup_anthropic(self, api_key: str):
        """Simple Anthropic setup"""
        if not ANTHROPIC_AVAILABLE:
            st.error("Anthropic library not installed. Run: pip install anthropic")
            return False
            
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            st.success("✅ Anthropic connected successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Anthropic setup failed: {str(e)}")
            return False
    
    def create_prompt(self, text: str) -> str:
        """Create classification prompt"""
        return f"""Analyze this social media post and classify the author's age group and confidence level.

POST: "{text}"

INSTRUCTIONS:
1. AGE GROUP - Choose ONE:
   - teens: mentions school, homework, class, high school, omg, literally
   - young_adults: mentions college, job, career, university, dating, apartment
   - adults: mentions kids, family, work, parenting, mortgage, business
   - seniors: mentions retirement, grandchildren, health issues, "years ago"

2. CONFIDENCE LEVEL - Choose ONE:
   - low: "not good enough", "terrible", "everyone else is better", "scared", "confused", "struggling"
   - high: "confident", "excellent", "proud", "amazing", "successful", "great at"
   - medium: "sometimes", "okay", "decent", "learning", "pretty good"

RESPOND EXACTLY LIKE THIS:
Age Group: [teens/young_adults/adults/seniors]
Confidence Level: [low/medium/high]
Reasoning: [Brief explanation of key words found]"""

    def classify_with_openai(self, text: str) -> dict:
        """Classify using OpenAI"""
        if not self.openai_client:
            return {'error': 'OpenAI not configured'}
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": self.create_prompt(text)}],
                temperature=0.1,
                max_tokens=200
            )
            
            self.api_calls += 1
            content = response.choices[0].message.content
            return self.parse_response(content, "OpenAI GPT-3.5")
            
        except Exception as e:
            return {'error': f'OpenAI API error: {str(e)}'}

    def classify_with_anthropic(self, text: str) -> dict:
        """Classify using Anthropic"""
        if not self.anthropic_client:
            return {'error': 'Anthropic not configured'}
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": self.create_prompt(text)}]
            )
            
            self.api_calls += 1
            content = response.content[0].text
            return self.parse_response(content, "Anthropic Claude")
            
        except Exception as e:
            return {'error': f'Anthropic API error: {str(e)}'}

    def parse_response(self, content: str, model_name: str) -> dict:
        """Parse AI response into structured format"""
        result = {
            'model': model_name,
            'age_group': 'unknown',
            'confidence_level': 'unknown', 
            'reasoning': 'Could not parse',
            'raw_response': content
        }
        
        content_lower = content.lower()
        
        # Extract age group
        if 'teens' in content_lower:
            result['age_group'] = 'teens'
        elif 'young_adults' in content_lower or 'young adults' in content_lower:
            result['age_group'] = 'young_adults'
        elif 'adults' in content_lower and 'young' not in content_lower:
            result['age_group'] = 'adults'
        elif 'seniors' in content_lower:
            result['age_group'] = 'seniors'
        
        # Extract confidence level
        if 'low' in content_lower:
            result['confidence_level'] = 'low'
        elif 'high' in content_lower:
            result['confidence_level'] = 'high'
        elif 'medium' in content_lower:
            result['confidence_level'] = 'medium'
        
        # Extract reasoning
        reasoning_match = re.search(r'reasoning:?\s*(.+)', content, re.IGNORECASE)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()[:150]
        
        return result

def create_ground_truth(df: pd.DataFrame, content_column: str) -> pd.DataFrame:
    """Create ground truth using keyword rules"""
    df = df.copy()
    df['true_age_group'] = 'young_adults'  # default
    df['true_confidence_level'] = 'medium'  # default
    
    for idx, row in df.iterrows():
        text = str(row[content_column]).lower()
        
        # Age classification rules
        if any(word in text for word in ['school', 'homework', 'class', 'omg', 'literally', 'high school', 'teacher']):
            df.at[idx, 'true_age_group'] = 'teens'
        elif any(word in text for word in ['college', 'job', 'career', 'university', 'interview', 'dating', 'apartment']):
            df.at[idx, 'true_age_group'] = 'young_adults'
        elif any(word in text for word in ['kids', 'family', 'parenting', 'children', 'mortgage', 'work', 'business']):
            df.at[idx, 'true_age_group'] = 'adults'
        elif any(word in text for word in ['retirement', 'grandchildren', 'health', 'retired', 'years ago']):
            df.at[idx, 'true_age_group'] = 'seniors'
        
        # Confidence classification rules
        low_words = ['not good enough', 'terrible', 'awful', 'everyone else', 'better than me', 
                     'scared', 'worried', 'confused', 'lost', 'struggling', 'cant do', 'useless']
        high_words = ['confident', 'excellent', 'great', 'proud', 'sure', 'amazing', 
                      'successful', 'accomplished', 'fantastic', 'brilliant', 'love doing']
        medium_words = ['sometimes', 'usually', 'okay', 'decent', 'learning', 'pretty good', 
                        'depends', 'mixed feelings', 'not bad', 'alright']
        
        if any(phrase in text for phrase in low_words):
            df.at[idx, 'true_confidence_level'] = 'low'
        elif any(phrase in text for phrase in high_words):
            df.at[idx, 'true_confidence_level'] = 'high'
        elif any(phrase in text for phrase in medium_words):
            df.at[idx, 'true_confidence_level'] = 'medium'
    
    return df

def calculate_metrics(y_true, y_pred, labels):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    # Overall metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'labels': labels
    }

def plot_confusion_matrix(y_true, y_pred, labels, title):
    """Plot confusion matrix using matplotlib only"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    return fig

@st.cache_data
def load_sample_data():
    """Load sample data with clear patterns"""
    data = [
        # Teens - Low confidence
        {'content': 'omg school is literally so hard everyone else gets it but i dont understand math at all', 'id': 1},
        {'content': 'homework is terrible im awful at everything compared to my classmates', 'id': 2},
        
        # Teens - High confidence  
        {'content': 'aced my history exam feeling confident about my grades this semester', 'id': 3},
        {'content': 'omg literally amazing at this subject love doing presentations in class', 'id': 4},
        
        # Young adults - Low confidence
        {'content': 'job interview tomorrow not sure if college prepared me everyone seems more qualified', 'id': 5},
        {'content': 'university is overwhelming everyone else seems to know what theyre doing', 'id': 6},
        
        # Young adults - High confidence
        {'content': 'confident about my career goals and excellent at networking in college', 'id': 7},
        {'content': 'apartment hunting going great proud of my job search success', 'id': 8},
        
        # Adults - Low confidence
        {'content': 'parenting is so hard everyone else seems like better parents than me with their kids', 'id': 9},
        {'content': 'work project struggling cant seem to manage family and business responsibilities', 'id': 10},
        
        # Adults - High confidence
        {'content': 'proud of how well our family project turned out excellent at balancing work and kids', 'id': 11},
        {'content': 'business meeting went fantastic confident about my parenting and career success', 'id': 12},
        
        # Seniors - Low confidence
        {'content': 'retirement planning confusing dont know if im doing it right for my health', 'id': 13},
        {'content': 'years ago things were simpler now everything seems overwhelming with grandchildren', 'id': 14},
        
        # Seniors - High confidence
        {'content': 'retirement is amazing love spending time with grandchildren feeling blessed', 'id': 15},
        {'content': 'years of experience paid off excellent at managing health and enjoying retirement', 'id': 16},
        
        # Additional mixed examples
        {'content': 'sometimes good at math class usually okay with homework depends on the teacher', 'id': 17},
        {'content': 'college job search is decent pretty good at interviews learning about career options', 'id': 18},
        {'content': 'family life has mixed feelings work is okay kids are usually well behaved', 'id': 19},
        {'content': 'retirement health is alright not bad for my age grandchildren visits are decent', 'id': 20},
    ]
    return pd.DataFrame(data)

def main():
    st.set_page_config(page_title="Social Media Classifier", page_icon="🎯", layout="wide")
    
    st.title("🎯 Social Media AI Classifier - Simplified")
    st.markdown("**Fast, simple classification with comprehensive evaluation metrics**")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = SimpleClassifier()
    
    classifier = st.session_state.classifier
    
    # Sidebar - API Setup
    st.sidebar.header("🔧 API Configuration")
    
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key")
    
    if openai_key and st.sidebar.button("Connect OpenAI"):
        classifier.setup_openai(openai_key)
    
    if anthropic_key and st.sidebar.button("Connect Anthropic"):
        classifier.setup_anthropic(anthropic_key)
    
    # Status
    st.sidebar.header("📊 Status")
    openai_status = "✅ Connected" if classifier.openai_client else "❌ Not connected"
    anthropic_status = "✅ Connected" if classifier.anthropic_client else "❌ Not connected"
    
    st.sidebar.write(f"OpenAI: {openai_status}")
    st.sidebar.write(f"Anthropic: {anthropic_status}")
    st.sidebar.write(f"API Calls Made: {classifier.api_calls}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Data & Analysis", "📈 Results", "🎯 Evaluation"])
    
    with tab1:
        st.header("📊 Data Management")
        
        # Data loading
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload CSV file (optional)", type="csv")
            
        with col2:
            use_sample = st.checkbox("Use sample data", value=True)
        
        # Load data
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            content_col = st.selectbox("Select content column:", df.columns)
        elif use_sample:
            df = load_sample_data()
            content_col = 'content'
        else:
            st.warning("Please upload a file or use sample data")
            return
        
        st.success(f"✅ Loaded {len(df)} posts")
        st.dataframe(df.head(10))
        
        # Model selection
        st.header("🤖 Classification")
        
        model_choice = st.radio("Choose model:", 
                               ["OpenAI GPT-3.5", "Anthropic Claude"], 
                               disabled=[not classifier.openai_client, not classifier.anthropic_client])
        
        sample_size = st.slider("Number of posts to analyze:", 1, min(50, len(df)), min(20, len(df)))
        
        if st.button("🚀 Run Classification", type="primary"):
            if model_choice == "OpenAI GPT-3.5" and not classifier.openai_client:
                st.error("❌ OpenAI not connected. Please add API key and connect.")
                return
            elif model_choice == "Anthropic Claude" and not classifier.anthropic_client:
                st.error("❌ Anthropic not connected. Please add API key and connect.")
                return
            
            # Sample data
            sample_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Run classification
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in sample_df.iterrows():
                status_text.text(f"Processing post {i+1}/{len(sample_df)}")
                progress_bar.progress((i + 1) / len(sample_df))
                
                text = row[content_col]
                
                if model_choice == "OpenAI GPT-3.5":
                    result = classifier.classify_with_openai(text)
                else:
                    result = classifier.classify_with_anthropic(text)
                
                if 'error' not in result:
                    result['post_content'] = text
                    results.append(result)
                else:
                    st.warning(f"Error processing post {i+1}: {result['error']}")
                
                time.sleep(0.1)  # Rate limiting
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                results_df = pd.DataFrame(results)
                st.session_state['results_df'] = results_df
                st.session_state['sample_df'] = sample_df
                st.session_state['content_col'] = content_col
                st.success(f"🎉 Successfully classified {len(results)} posts!")
            else:
                st.error("❌ No successful classifications")
    
    with tab2:
        st.header("📈 Classification Results")
        
        if 'results_df' not in st.session_state:
            st.info("👆 Please run classification first")
            return
        
        results_df = st.session_state['results_df']
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Posts Classified", len(results_df))
        
        with col2:
            age_mode = results_df['age_group'].mode()[0] if len(results_df) > 0 else 'Unknown'
            st.metric("👥 Most Common Age", age_mode.replace('_', ' ').title())
        
        with col3:
            conf_mode = results_df['confidence_level'].mode()[0] if len(results_df) > 0 else 'Unknown'
            st.metric("💪 Most Common Confidence", conf_mode.title())
        
        with col4:
            model_name = results_df['model'].iloc[0] if len(results_df) > 0 else 'Unknown'
            st.metric("🤖 Model Used", model_name.split()[0])
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            age_counts = results_df['age_group'].value_counts()
            fig_age = px.pie(values=age_counts.values, names=age_counts.index, 
                           title="👥 Age Group Distribution")
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            conf_counts = results_df['confidence_level'].value_counts()
            fig_conf = px.pie(values=conf_counts.values, names=conf_counts.index,
                            title="💪 Confidence Level Distribution")
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Results table
        st.subheader("📋 Detailed Results")
        display_df = results_df[['post_content', 'age_group', 'confidence_level', 'reasoning']].copy()
        display_df['post_content'] = display_df['post_content'].str[:100] + '...'
        st.dataframe(display_df)
        
        # Download option
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results", csv, f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    
    with tab3:
        st.header("🎯 Model Evaluation")
        
        if 'results_df' not in st.session_state:
            st.info("👆 Please run classification first")
            return
        
        results_df = st.session_state['results_df']
        sample_df = st.session_state['sample_df']
        content_col = st.session_state['content_col']
        
        # Generate ground truth
        if st.button("🔄 Generate Ground Truth"):
            with st.spinner("Generating ground truth labels..."):
                ground_truth_df = create_ground_truth(sample_df, content_col)
                st.session_state['ground_truth_df'] = ground_truth_df
                st.success("✅ Ground truth generated!")
        
        if 'ground_truth_df' not in st.session_state:
            st.info("👆 Please generate ground truth first")
            return
        
        ground_truth_df = st.session_state['ground_truth_df']
        
        # Calculate evaluation metrics
        st.subheader("📊 Evaluation Metrics")
        
        # Ensure same length
        min_len = min(len(results_df), len(ground_truth_df))
        results_subset = results_df.head(min_len).reset_index(drop=True)
        truth_subset = ground_truth_df.head(min_len).reset_index(drop=True)
        
        # Age group evaluation
        age_labels = ['teens', 'young_adults', 'adults', 'seniors']
        age_metrics = calculate_metrics(truth_subset['true_age_group'], 
                                      results_subset['age_group'], 
                                      age_labels)
        
        # Confidence level evaluation
        conf_labels = ['low', 'medium', 'high']
        conf_metrics = calculate_metrics(truth_subset['true_confidence_level'], 
                                       results_subset['confidence_level'], 
                                       conf_labels)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Age Group Metrics")
            st.metric("Accuracy", f"{age_metrics['accuracy']:.3f}")
            st.metric("Macro Precision", f"{age_metrics['macro_precision']:.3f}")
            st.metric("Macro Recall", f"{age_metrics['macro_recall']:.3f}")
            st.metric("Macro F1-Score", f"{age_metrics['macro_f1']:.3f}")
        
        with col2:
            st.subheader("💪 Confidence Level Metrics")
            st.metric("Accuracy", f"{conf_metrics['accuracy']:.3f}")
            st.metric("Macro Precision", f"{conf_metrics['macro_precision']:.3f}")
            st.metric("Macro Recall", f"{conf_metrics['macro_recall']:.3f}")
            st.metric("Macro F1-Score", f"{conf_metrics['macro_f1']:.3f}")
        
        # Detailed metrics tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Age Group - Per Class Metrics")
            age_detailed = pd.DataFrame({
                'Class': age_labels,
                'Precision': age_metrics['precision'],
                'Recall': age_metrics['recall'],
                'F1-Score': age_metrics['f1'],
                'Support': age_metrics['support']
            })
            st.dataframe(age_detailed.round(3))
        
        with col2:
            st.subheader("💪 Confidence Level - Per Class Metrics")
            conf_detailed = pd.DataFrame({
                'Class': conf_labels,
                'Precision': conf_metrics['precision'],
                'Recall': conf_metrics['recall'],
                'F1-Score': conf_metrics['f1'],
                'Support': conf_metrics['support']
            })
            st.dataframe(conf_detailed.round(3))
        
        # Confusion matrices
        st.subheader("🔥 Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age_cm = plot_confusion_matrix(truth_subset['true_age_group'], 
                                             results_subset['age_group'],
                                             age_labels,
                                             "Age Group Confusion Matrix")
            st.pyplot(fig_age_cm)
        
        with col2:
            fig_conf_cm = plot_confusion_matrix(truth_subset['true_confidence_level'], 
                                              results_subset['confidence_level'],
                                              conf_labels,
                                              "Confidence Level Confusion Matrix")
            st.pyplot(fig_conf_cm)
        
        # Classification report
        st.subheader("📋 Detailed Classification Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Age Group Classification Report:")
            age_report = classification_report(truth_subset['true_age_group'], 
                                             results_subset['age_group'],
                                             target_names=age_labels)
            st.text(age_report)
        
        with col2:
            st.text("Confidence Level Classification Report:")
            conf_report = classification_report(truth_subset['true_confidence_level'], 
                                              results_subset['confidence_level'],
                                              target_names=conf_labels)
            st.text(conf_report)
        
        # Summary interpretation
        st.subheader("🎯 Performance Summary")
        
        overall_age_acc = age_metrics['accuracy']
        overall_conf_acc = conf_metrics['accuracy']
        combined_score = (overall_age_acc + overall_conf_acc) / 2
        
        if combined_score >= 0.8:
            st.success(f"🎉 Excellent Performance! Combined accuracy: {combined_score:.1%}")
        elif combined_score >= 0.65:
            st.info(f"👍 Good Performance! Combined accuracy: {combined_score:.1%}")
        else:
            st.warning(f"⚠️ Performance needs improvement. Combined accuracy: {combined_score:.1%}")
        
        # Error analysis
        if st.checkbox("🔍 Show Error Analysis"):
            st.subheader("🔍 Error Analysis")
            
            # Find misclassified examples
            age_errors = truth_subset['true_age_group'] != results_subset['age_group']
            conf_errors = truth_subset['true_confidence_level'] != results_subset['confidence_level']
            
            if age_errors.any():
                st.write("**Age Group Misclassifications:**")
                age_error_df = pd.DataFrame({
                    'Post': sample_df.loc[age_errors, content_col].str[:100] + '...',
                    'True Age': truth_subset.loc[age_errors, 'true_age_group'],
                    'Predicted Age': results_subset.loc[age_errors, 'age_group'],
                    'AI Reasoning': results_subset.loc[age_errors, 'reasoning']
                })
                st.dataframe(age_error_df.head(5))
            
            if conf_errors.any():
                st.write("**Confidence Level Misclassifications:**")
                conf_error_df = pd.DataFrame({
                    'Post': sample_df.loc[conf_errors, content_col].str[:100] + '...',
                    'True Confidence': truth_subset.loc[conf_errors, 'true_confidence_level'],
                    'Predicted Confidence': results_subset.loc[conf_errors, 'confidence_level'],
                    'AI Reasoning': results_subset.loc[conf_errors, 'reasoning']
                })
                st.dataframe(conf_error_df.head(5))
    
    st.markdown("---")
    st.markdown("🎯 **Simplified Social Media Classifier** - Clean, fast, and reliable!")

if __name__ == "__main__":
    main()
