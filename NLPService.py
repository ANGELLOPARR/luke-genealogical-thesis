from flask import Flask, jsonify, request
from flask_cors import CORS
import spacy
import re
# Add neural coref to SpaCy's pipeline
import neuralcoref

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp)
doc = None
print('done loading model')

def clean_wikitext(text):
    cleaned = re.sub(r'\[.{1,2}\]', '', text)
    cleaned = re.sub(r'\(([^\)]*[^\)]*?)\)', '', cleaned)
    return cleaned

@app.route('/', methods=['GET'])
def index():
    return "Welcome to my app!"

@app.route('/weatherReport/', methods=['GET'])
def WeatherReport():
    global weather
    return jsonify([weather])

@app.route('/tokenizeText/', methods=['POST'])
def tokenize_text():
    global nlp, doc
    req_data = request.get_json()
    print(f'\n\n\n{request.get_json()}\n\n\n')
    doc = nlp(req_data['text'])
    return jsonify({'tokens' : [e.text for e in doc]})

@app.route('/getRelationships/', methods=['POST'])
def get_relationships():
    global nlp, doc
    print('running!')
    res = []
    paragraph = request.get_json()['text']
    doc = nlp(clean_wikitext(paragraph))
    doc_len = len(doc.text)
    found_entities = [(e, doc[:0]) for e in doc.ents if e.label_ == 'PERSON']
    clusters = doc._.coref_clusters
    
    # Remove all clusters that don't have a named entity in the main resolved span
    clusters = [c for c in clusters if len(c.main.ents) > 0]

    # Remove all clusters that don't contain a PERSON named entity in the main resolved span
    clusters = [c for c in clusters if len([c_ent for c_ent in c.main.ents if c_ent.label_ == 'PERSON']) > 0]

    # TODO: Find a way to parse out people from groups (older sister, Kourtney, a younger sister, Khloé)
    # and include their coreferences by part.
    
    # TODO: Might need to abstract found_entities to include who the entities are linked to (if any)
    for c in clusters:
        for ment in c.mentions:
            if ment != c.main and (ment, c.main) not in found_entities:
                found_entities.append((ment, c.main))

    MAX_RANGE = 140
    CONTEXT_CHARS = 30

    for (ent1, coref1) in found_entities:
        for (ent2, coref2) in found_entities:
            
            # if either entity is semantically the same, skip it
            if ent2 == ent1 or coref2 == ent1 or coref1 == ent2:
                continue

            # make sure that ent2 is within range of ent1 and the direction is strictly ent1 --> ent2
            if ent2.end_char <= ent1.start_char + MAX_RANGE and ent1.start_char < ent2.start_char:
                span_start = 0 if ent1.start_char < CONTEXT_CHARS else ent1.start_char - CONTEXT_CHARS
                if span_start != 0:
                    span_start = doc.text[span_start:].find(' ') + span_start
                span_end = ent2.end_char + CONTEXT_CHARS
                if span_end < doc_len:
                    span_end = doc.text[:span_end].rfind(' ')
                new_rel = {
                    'text_span' : doc.text[span_start: span_end],
                    'entity1' : {
                        'start' : ent1.start_char - span_start,
                        'end' : ent1.end_char - span_start,
                        'text' : ent1.text
                    },
                    'entity2' : {
                        'start' : ent2.start_char - span_start,
                        'end' : ent2.end_char - span_start,
                        'text' : ent2.text
                    }
                }
                
                ## The quick ][ brown fox jumped ENTITY1 the lazy dog
                
#                 print(f'ENT 1 ({ent1}), ENT 2 ({ent2})')
#                 rel_span = doc.text[ent1.start_char : ent2.end_char]
#                 encoded_e1 = f'[e1] {ent1.text} [/e1]'
#                 encoded_e2 = f'[e2] {ent2.text} [/e2]'

#                 new_span = encoded_e1 + doc.text[ent1.end_char : ent2.start_char] + encoded_e2
                res.append(new_rel)

    response = jsonify({'data': res})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

# @app.route('/getRelationships/', methods=['POST'])
# def get_relationships():
#     return jsonify({'data': 'testing'})


if __name__ == '__main__':
    app.run(debug=True)