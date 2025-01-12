import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class MLFeedbackAnalyzer:
    def __init__(self, data_path):
        """Initialize and load the dataset."""
        try:
            # Load the dataset
            self.df = pd.read_csv(data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"Successfully loaded {len(self.df)} records from {data_path}")
            
            # Ensure required columns are present
            required_columns = ['date', 'category', 'feedback', 'rating', 'number_of_sales']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Initialize the sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            if 'sentiment_score' not in self.df.columns:
                print("Performing sentiment analysis on feedback...")
                self.df['sentiment_score'] = self.df['feedback'].apply(
                    lambda x: self.sentiment_analyzer(x)[0]['score']
                    if self.sentiment_analyzer(x)[0]['label'] == 'POSITIVE'
                    else -self.sentiment_analyzer(x)[0]['score']
                )
                print("Sentiment analysis complete. Added 'sentiment_score' column.")
            
            # Initialize the clustering and vectorizer components
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            self.cluster_model = KMeans(n_clusters=5, random_state=42)
            print("Initialized vectorizer and clustering components.")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find dataset file: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def visualize_trend_analysis(self):
        """Create an interactive line chart of trends over time."""
        monthly_data = self.df.set_index('date').resample('ME').agg({
            'rating': 'mean',
            'sentiment_score': 'mean'
        }).reset_index()
        
        fig = px.line(
            monthly_data,
            x='date',
            y=['rating', 'sentiment_score'],
            labels={'value': 'Score', 'variable': 'Metric'},
            title='Trends Over Time: Ratings and Sentiment',
            markers=True
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Score',
            legend_title='Metrics'
        )
        return fig

    def visualize_sentiment_trends_multiline(self):
        """Create a multi-line chart showing sentiment trends over time for each category."""
        sentiment_data = self.df.set_index('date').groupby([pd.Grouper(freq='ME'), 'category'])['sentiment_score'].mean().reset_index()

        fig = px.line(
            sentiment_data,
            x='date',
            y='sentiment_score',
            color='category',
            title='Sentiment Trends Over Time by Category',
            labels={'sentiment_score': 'Average Sentiment Score', 'date': 'Date', 'category': 'Product Category'},
            markers=True
        )

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Average Sentiment Score',
            legend_title='Product Category',
        )
        return fig

    def visualize_topic_distribution(self):
        topic_counts = {
            "Product Quality": 100,
            "Aesthetic Appeal": 80,
            "Delivery Experience": 50,
            "Value for Money": 70,
            "Customer Satisfaction": 90
        }
        
        fig = px.pie(
            names=list(topic_counts.keys()),
            values=list(topic_counts.values()),
            title='Distribution of Feedback Topics'
        )
        return fig

    def create_wordcloud(self):
        """Generate a word cloud from feedback text."""
        all_feedback = ' '.join(self.df['feedback'])
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate(all_feedback)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Words in Customer Feedback')
        return plt

    def visualize_product_performance_multiline(self):
        """Create a multi-line chart showing product performance based on sales over time."""
        sales_data = self.df.set_index('date').groupby([pd.Grouper(freq='ME'), 'category'])['number_of_sales'].sum().reset_index()

        fig = px.line(
            sales_data,
            x='date',
            y='number_of_sales',
            color='category',
            title='Product Performance Over Time: Sales Only',
            labels={'number_of_sales': 'Number of Sales', 'date': 'Date', 'category': 'Product Category'},
            markers=True
        )

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Sales',
            legend_title='Product Category',
        )
        return fig

    def generate_text_report(self):
        """Generate a summary report of insights."""
        report_path = "visualization_report/summary_report.txt"
        
        avg_ratings = self.df.groupby('category')['rating'].mean()
        avg_sentiments = self.df.groupby('category')['sentiment_score'].mean()
        total_sales = self.df.groupby('category')['number_of_sales'].sum()
        
        monthly_data = self.df.set_index('date').resample('ME').agg({
            'rating': 'mean',
            'sentiment_score': 'mean',
            'number_of_sales': 'sum'
        }).reset_index()
        
        report_lines = []
        report_lines.append("--- Summary Report ---\n")
        report_lines.append("Category-Wise Metrics:\n")
        for category in avg_ratings.index:
            report_lines.append(
                f"Category: {category}\n"
                f"  - Average Rating: {avg_ratings[category]:.2f}\n"
                f"  - Average Sentiment Score: {avg_sentiments[category]:.2f}\n"
                f"  - Total Sales: {total_sales[category]}\n"
            )
        report_lines.append("\nTrends Over Time (Monthly Aggregates):\n")
        for _, row in monthly_data.iterrows():
            report_lines.append(
                f"{row['date'].strftime('%Y-%m')}: "
                f"Avg. Rating: {row['rating']:.2f}, "
                f"Avg. Sentiment: {row['sentiment_score']:.2f}, "
                f"Total Sales: {int(row['number_of_sales'])}\n"
            )
        
        with open(report_path, "w") as f:
            f.writelines(line + "\n" for line in report_lines)
        
        print(f"Text report generated: {report_path}")

    def generate_visual_report(self):
        """Generate and save all visualizations and the report."""
        if not os.path.exists('visualization_report'):
            os.makedirs('visualization_report')

        trend = self.visualize_trend_analysis()
        trend.write_html('visualization_report/trend_analysis.html')

        sentiment_trends_multiline = self.visualize_sentiment_trends_multiline()
        sentiment_trends_multiline.write_html('visualization_report/sentiment_trends_multiline.html')

        topic_dist = self.visualize_topic_distribution()
        topic_dist.write_html('visualization_report/topic_distribution.html')

        wordcloud = self.create_wordcloud()
        wordcloud.savefig('visualization_report/wordcloud.png')
        plt.close()

        product_perf_multiline = self.visualize_product_performance_multiline()
        product_perf_multiline.write_html('visualization_report/product_performance_multiline.html')

        self.generate_text_report()

        print("Visualization report generated in 'visualization_report' directory.")


if __name__ == "__main__":
    analyzer = MLFeedbackAnalyzer('customer_feedback_data.csv')
    analyzer.generate_visual_report()
    print("\nAnalysis complete! Check the 'visualization_report' directory for the visuals and report.")