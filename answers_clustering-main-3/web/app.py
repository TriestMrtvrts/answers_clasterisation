from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import umap.umap_ as umap
from sklearn.cluster import AgglomerativeClustering
import compress_fasttext
from torch import argmax as torchargmax

app = Flask(__name__)
app.secret_key = 'a66ee1919de09f25451412411bed2fb755f845d70f96bf0a'

agg_clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3)

fasttext = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
    'uploads/geowac_tokens_sg_300_5_2020-400K-100K-300.bin'
)

reducer2d = umap.UMAP(n_components=2)

sentiment_model_tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')
sentiment_model = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')

questions = ['Опишите одним словом ваше текущее состояние',
             'Напишите свою ассоциацию со словом "бюрократ"',
             'Что позволяет вам лично поддерживать уверенность и рабочий настрой?',
             'В чём причина стресса, по вашему мнению?',
             'Вопрос 5. В чем, по Вашему мнению, причина роста травматизма?']


@app.route('/download', methods=['GET', 'POST'])
def download():
    df = json.loads(request.form['results'])
    df = pd.DataFrame(data=df)
    filename = 'uploads/output.csv'

    df.to_csv(filename, index=False)
    return send_file(filename, as_attachment=True)


@app.route('/processing', methods=['POST', 'GET'])
def processing():
    selected_question = request.form['question']

    return render_template('processing.html', selected_question=selected_question)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', questions=questions)


@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    question = request.args.get('selected_question')

    df = pd.read_csv('uploads/ques_ans_data.csv')
    df = df[df.question == question].drop(['question'], axis=1)

    tokens = sentiment_model_tokenizer(list(df['answers']), padding=True, truncation=True, return_tensors='pt')
    sentiments = torchargmax(sentiment_model(**tokens).logits, dim=1)
    df['sentiment'] = sentiments

    df['sentiment'] = df['sentiment'].apply(lambda x: sentiment_model.config.id2label[x])

    df['embeds'] = list(fasttext[list(df['answers'])])

    umap_embeds = reducer2d.fit_transform(list(df['embeds']))
    df['embeds'] = list(umap_embeds)

    labels = agg_clusterer.fit_predict(list(df['embeds']))

    df['cluster'] = labels

    df_group = df.groupby('sentiment').count().reset_index()
    grouped_data = df.groupby('cluster').agg({'embeds': list, 'answers': list, 'sentiment': list}).reset_index()

    grouped = df.groupby('cluster').count().reset_index()

    mean_group_count = int(grouped['answers'].sum() / len(df['cluster'].unique()))
    group_count = len(df['cluster'].unique())

    max_cluster = grouped[grouped.answers == grouped.answers.max()]
    max_cluster_count = int(max_cluster['answers'].iloc[0])

    datasets = []
    cluster_count = []
    cluster_name = []
    positive = []
    for index, row in grouped_data.iterrows():
        label = row['cluster']
        xy_list = row['embeds']
        text_list = row['answers']
        positive.append(row['sentiment'].count('neutral'))
        data = []
        for xy in zip(xy_list, text_list):
            x, y = xy[0][0], xy[0][1]
            data.append({'x': float(x), 'y': float(y), 'text': xy[1]})

        cluster_count.append(len(data))
        cluster_name.append(label)
        dataset = {'label': label, 'data': data}
        datasets.append(dataset)


    return render_template('charts.html', labels=list(df_group['sentiment']), data=list(df_group['answers']),
                           df=df.to_json(), datasets=datasets, mean_group_count=mean_group_count,
                           group_count=group_count, max_cluster_count=max_cluster_count,
                           cluster_count=cluster_count, cluster_name=cluster_name, positive=positive, title=question)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
