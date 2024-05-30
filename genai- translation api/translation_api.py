from flask import Flask, request, jsonify
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get('text')
    src_lang = data.get('src_lang')
    dest_lang = data.get('dest_lang')

    if text and src_lang and dest_lang:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        translated_text = translated.text
        return jsonify({'translated_text': translated_text})
    else:
        return jsonify({'error': 'Missing parameters'}), 400

if __name__ == '__main__':
    app.run(debug=True)
