from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import os
import time
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_model import train_model, predict

app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
os.makedirs(app.static_folder, exist_ok=True)

# Train model once on startup with dummy data (or provide dataset path)
model, vectorizer, sentiment_map = train_model('amazon.csv')

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("POST received")
        file = request.files.get('file')
        if not file or file.filename == '':
            print("No file selected")
            return render_template('index.html', error="Please upload a CSV file.")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            print("File saved")
        except Exception as e:
            print(f"Failed to save file: {e}")
            return render_template('index.html', error=f"Failed to save file: {str(e)}")

        try:
            df = pd.read_csv(filepath)
            print(f"DF shape: {df.shape}")
            df.columns = df.columns.str.strip()  # Strip spaces from column names
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Failed to read CSV: {e}")
            return render_template('index.html', error=f"Failed to read CSV file: {str(e)}")

        # Find the review column
        review_column = None
        for col in df.columns:
            if col.lower() in ['review', 'reviews', 'text', 'comment', 'feedback', 'message']:
                review_column = col
                break
        if review_column is None:
            print("No suitable review column found")
            return render_template('index.html', error="CSV must contain a column for reviews (e.g., 'review', 'text', 'comment').")

        reviews = df[review_column].astype(str).tolist()
        print(f"Number of reviews: {len(reviews)}")
        try:
            sentiments = predict(model, vectorizer, sentiment_map, reviews)
            print(f"Predicted sentiments: {sentiments[:5]}")
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('index.html', error=f"Prediction error: {str(e)}")

        df['predicted_sentiment'] = sentiments

        # Create summary table
        sentiment_counts = df['predicted_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        summary_table = sentiment_counts.to_html(classes='data', index=False)

        # Get sentiment counts for chart
        chart_sentiment_counts = df['predicted_sentiment'].value_counts().to_dict()

        # Get positive review examples
        positive_reviews = df[df['predicted_sentiment'] == 'positive'][review_column].head(5).tolist()
        negative_reviews = df[df['predicted_sentiment'] == 'negative'][review_column].head(5).tolist()

        # Plot sentiment distribution (bar chart)
        plt.figure(figsize=(6,4))
        sns.countplot(x='predicted_sentiment', data=df)
        plt.title('Sentiment Distribution')
        plot_filename = f'sentiment_distribution_{int(time.time())}.png'
        plot_path = os.path.join(app.static_folder, plot_filename)
        try:
            plt.savefig(plot_path)
            print("Plot saved")
        except Exception as e:
            print(f"Failed to save plot: {e}")
            return render_template('index.html', error=f"Failed to save plot: {str(e)}")
        finally:
            plt.close()

        plot_url = url_for('static', filename=plot_filename)

        # Plot sentiment distribution (pie chart)
        plt.figure(figsize=(6,6))
        sentiment_counts = df['predicted_sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.ylabel('')
        plt.title('Sentiment Distribution (Pie Chart)')
        pie_filename = f'sentiment_pie_{int(time.time())}.png'
        pie_path = os.path.join(app.static_folder, pie_filename)
        try:
            plt.savefig(pie_path)
            print("Pie chart saved")
        except Exception as e:
            print(f"Failed to save pie chart: {e}")
            return render_template('index.html', error=f"Failed to save pie chart: {str(e)}")
        finally:
            plt.close()

        pie_url = url_for('static', filename=pie_filename)

        # Plot top 10 frequent words in positive reviews
        from collections import Counter
        import re

        positive_text = ' '.join(df[df['predicted_sentiment'] == 'positive'][review_column].astype(str).tolist()).lower()
        # Clean text: remove punctuation and split
        words = re.findall(r'\b\w+\b', positive_text)
        stop_words = set(['the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'this', 'for', 'with', 'on', 'was', 'that', 'as', 'but', 'are', 'have', 'be', 'not', 'they', 'you', 'at', 'or', 'from'])
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        word_counts = Counter(filtered_words)
        most_common = word_counts.most_common(10)

        if most_common:
            words, counts = zip(*most_common)
            plt.figure(figsize=(8,5))
            sns.barplot(x=list(counts), y=list(words), palette='viridis')
            plt.title('Top 10 Frequent Words in Positive Reviews')
            plt.xlabel('Count')
            plt.ylabel('Words')
            plt.tight_layout()
            bar_filename = f'positive_words_{int(time.time())}.png'
            bar_path = os.path.join(app.static_folder, bar_filename)
            try:
                plt.savefig(bar_path)
                print("Bar chart saved")
            except Exception as e:
                print(f"Failed to save bar chart: {e}")
                return render_template('index.html', error=f"Failed to save bar chart: {str(e)}")
            finally:
                plt.close()
            bar_url = url_for('static', filename=bar_filename)
        else:
            bar_url = None

        total_reviews = len(reviews)
        return render_template('index.html', tables=[summary_table], plot_url=plot_url, pie_url=pie_url, bar_url=bar_url, total_reviews=total_reviews, positive_reviews=positive_reviews, negative_reviews=negative_reviews, chart_sentiment_counts=chart_sentiment_counts)

    return render_template('index.html')


@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
