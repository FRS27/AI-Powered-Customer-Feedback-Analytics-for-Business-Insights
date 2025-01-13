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

![image](https://github.com/user-attachments/assets/029d3b32-3248-4557-9ca0-30a1fe241562)



**Topic Distribution:**

Pie charts breaking down feedback topics into proportional contributions.

**Word Cloud:**

Visual representation of frequent terms in customer feedback using WordCloud.

![image](https://github.com/user-attachments/assets/6996ccde-091d-4ee4-ab69-bdd6bcb6b1ed)


**Sales Performance:**

Interactive sales trend visualizations by product category over time.



**5. Automated Reporting**

**Text Report:**

Generates category-wise summaries of sentiment scores, ratings, and sales performance.

![image](https://github.com/user-attachments/assets/db836b4c-806c-4327-bc3f-ad232d8dfb6c)

![image](https://github.com/user-attachments/assets/8f6c3845-bad9-4f3b-8497-8001b64286a8)



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

![image](https://github.com/user-attachments/assets/a28a37f0-f39d-4708-9927-27172cef24ae)


**2. Topic Distribution**

**Insights:** Key themes in customer feedback (e.g., "Product Quality," "Delivery Experience") are identified and visualized using clustering and topic modeling.

**3. Word Cloud**

**Insights:** Common terms in customer feedback provide insights into recurring concerns and positive features.

**4. Product Performance**

**Insights:** Sales trends over time reveal product success rates and customer demand patterns.


**🛠️ Technologies Used**

**Programming Language:** Python

**Libraries:**

**Natural Language Processing:** Hugging Face Transformers (DistilBERT pipeline).

**Clustering and Feature Engineering:** TF-IDF Vectorization, K-Means Clustering (scikit-learn).

**Visualization:** Plotly, Matplotlib, WordCloud.

**Data Manipulation:** pandas, NumPy.

**Machine Learning Techniques:**

Sentiment analysis via NLP models.

Topic modeling and clustering.

**🧑‍💻 Contributing**

Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

**📬 Contact**

For any inquiries, feel free to reach out:

**Email:** ahmad.faris615@gmail.com
