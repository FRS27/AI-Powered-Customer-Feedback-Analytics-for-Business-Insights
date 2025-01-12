**# AI-Powered-Customer-Feedback-Analytics-for-Business-Insights**

This project implements an advanced NLP and machine learning pipeline for end-to-end customer feedback analysis, leveraging state-of-the-art tools like Hugging Face Transformers, TF-IDF Vectorization, and K-Means Clustering. The pipeline delivers actionable insights through sentiment analysis, topic modeling, and interactive visualizations, enabling businesses to uncover sentiment trends, customer feedback patterns, and sales performance with cutting-edge AI technologies.

**🚀 Features**

**1. Advanced Sentiment Analysis**

Built with Hugging Face Transformers using the distilbert-base-uncased-finetuned-sst-2-english model.
Performs granular sentiment classification, providing a sentiment polarity (positive/negative) and sentiment intensity scores for each customer feedback entry.

**2. High-Performance Feature Extraction**

**TF-IDF Vectorization:**

Extracts high-value terms from customer feedback for dimensionality reduction and identifying critical features.
Utilizes stop-word removal and term frequency-inverse document frequency techniques to highlight the most relevant words.

**3. Clustering with K-Means**

Applies K-Means Clustering to group feedback into distinct thematic clusters such as:

Product Quality
Delivery Experience
Customer Satisfaction
Provides a scalable approach to topic modeling, enabling quick identification of recurring themes in feedback.

**4. Comprehensive Interactive Visualizations**

**Built using Plotly and Matplotlib for dynamic, high-quality visualizations:**

**Trends Over Time:**

Interactive multi-line charts visualizing sentiment trends, ratings, and sales data across product categories.

**Topic Distribution:**

Pie charts breaking down feedback topics into proportional contributions.

**Word Cloud:**

Visual representation of frequent terms in customer feedback using WordCloud.

**Sales Performance:**

Interactive sales trend visualizations by product category over time.

**5. Automated Reporting**

**Text Report:**

Generates category-wise summaries of sentiment scores, ratings, and sales performance.

**Visualization Report:**

Includes interactive HTML visualizations and PNG word cloud outputs for business-ready insights.

**6. Streamlined Data Processing**

Handles structured datasets with pandas, ensuring robust preprocessing for columns like:

**date:** Feedback timestamp.

**feedback:** Textual feedback data.

**rating:** Customer-provided ratings.

**number_of_sales:** Sales data for product performance tracking.

Fully automates missing value checks and column verifications.

**📊 Key Results**

**1. Sentiment Trends Over Time**

**Insights:** Monthly sentiment scores and ratings help identify trends in customer satisfaction for each product category.

**2. Topic Distribution**

**Insights:** Key themes in customer feedback (e.g., "Product Quality," "Delivery Experience") are identified and visualized using clustering and topic modeling.

**3. Word Cloud**

**Insights:** Common terms in customer feedback provide insights into recurring concerns and positive features.

**4. Product Performance**

**Insights:** Sales trends over time reveal product success rates and customer demand patterns.
