from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    data = request.get_json()
    text = data['text']
    language = data['language']

    if language != 'en':
        translated_text = GoogleTranslator(source=language, target='en').translate(text)
    else:
        translated_text = text

    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(translated_text)

    response = {
        'positive': sentiment_dict['pos'] * 100,
        'negative': sentiment_dict['neg'] * 100,
        'neutral': sentiment_dict['neu'] * 100,
        'compound': sentiment_dict['compound'] * 100
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
