import re
import os
import pandas as pd
import argparse
import json
import xml.etree.cElementTree as ET
import spacy

def get_label_data(article_id, label_tree):
    """ Get labels corresponding to an article
        for the hyperpartisan dataset
    """
    root = False
    for event, elem in label_tree:
        if event == 'start' and not root:
            root = elem
        if event == 'end':
            if elem.attrib.get('id') == article_id:
                label = elem.get('hyperpartisan')
                bias = elem.get('bias')
                labeled_by = elem.get('labeled-by')
                elem.clear()
                root.clear()
                return label, bias, labeled_by
            else:
                elem.clear()
                root.clear()
    raise Exception(f'article_id: {article_id} not found')

def parse_debate_dataset(filepath, nlp):
    motions = []
    with open(filepath, 'r') as f:
        line = next(f)
        while line:
            motion_id = re.findall(r'Motion ID: (\d+-\d+)', line)[0]
            line = next(f)
            instance_id = re.findall(r'Instance ID: (\d+-\d+-\d+)', line)[0]
            line = next(f)
            line = next(f)
            line = next(f)
            motion = ''
            while not line.startswith('Assertion'):
                motion += line
                line = next(f)
            assertion = ''
            line = next(f)
            while not line.startswith('Justification'):
                assertion += line
                line = next(f)
            justification = ''
            line = next(f)
            while not line.startswith('Score'):
                # skip bibliography (not sure if we should)
                if not re.match('^\[\d+\]', line):
                    justification += line
                line = next(f)
            line = next(f)
            justification = split_sentences(justification, nlp)
            scores = re.findall(r'GE:(\d)\tLO:(\d)\tIS:(\d)\tUA:(\d)\tUJ:(\d)\tPersuasiveness:(\d)', line)[0]
            motions.append([
                motion_id, 
                instance_id, 
                motion.strip(), 
                assertion.strip(), 
                justification.strip(), 
                *[int(score) for score in scores]
            ])
            try:
                while not line.startswith('Motion ID'):
                    line = next(f)
            except StopIteration: 
                break
                
    return pd.DataFrame(motions, columns=['MotionID',
                                        'InstanceID',
                                        'Motion',
                                        'Assertion', 
                                        'Justification', 
                                        'GE', 'LO', 'IS', 'UA', 'UJ', 'Persuasiveness'])

def save_debates_ds(debates_df, args):
    train_df = debates_df.sample(frac=0.6, random_state=42)
    rest_df = debates_df.drop(train_df.index)
    valid_df = rest_df.sample(frac=0.5, random_state=42)
    test_df = rest_df.drop(valid_df.index)
    train_path = os.path.join(args.debate_output_dir, f'{args.debate_output_prefix}-train.json') 
    valid_path = os.path.join(args.debate_output_dir, f'{args.debate_output_prefix}-valid.json') 
    test_path = os.path.join(args.debate_output_dir, f'{args.debate_output_prefix}-test.json') 
    train_df.to_json(train_path, orient='records', indent=2)
    valid_df.to_json(valid_path, orient='records', indent=2)
    test_df.to_json(test_path, orient='records', indent=2)

def process_article(elem, train_label_tree, nlp, f):
    article_id = elem.attrib.get('id')
    title = elem.attrib.get('title')
    # process article content
    xml = ET.tostring(elem, encoding='utf-8', method='xml').decode()
    # replace anchor tags and whitespace with single space
    text = re.sub(r'(<a.*?>|<\/a>|\s{1,})', ' ', xml) 
    paragraphs = re.findall(r'<p>(.*?)<\/p>', text)
    #split into sentences with spacy
    text = title
    for p in paragraphs:
        text = '[SEP]'.join([text, split_sentences(p, nlp)])
    label, bias, labeled_by = get_label_data(article_id, train_label_tree)
    return {
        'article_id': article_id,
        'label': label,
        'bias': bias,
        'labeled_by': labeled_by,
        'text': text,
    }

def close_hp_file(f):
    # remove trailing comma and newline
    f.seek(f.tell()-2, os.SEEK_SET)
    f.truncate()
    f.write(']')
    f.close()

def preprocess_hp_dataset(data_path, labels_path, output_file_path, nlp):
    """ Combines data and groundtruth labels xml files in a single tsv file
        Output file format:
            article_id label bias labeled_by title text
    """
    data_tree = ET.iterparse(data_path, events=('start', 'end'))
    label_tree = ET.iterparse(labels_path, events=('start', 'end'))

    max_filesize = args.hp_max_filesize * 1024**2
    idx_file = 0
    filepath = f'{output_file_path}.json'
    f = open(filepath, 'w', encoding='utf-8')
    f.write('[')
    i = 0
    root = False
    for event, elem in data_tree:  
        if event == 'start' and not root:
            root = elem
        if event == 'end' and elem.tag == 'article':
            article = process_article(elem, label_tree, nlp, f)
            json.dump(article, f, indent=2, ensure_ascii=False)
            f.write(',\n')
            # clear prev elements from memory
            elem.clear()
            root.clear()
            if os.stat(filepath).st_size > max_filesize:
                close_hp_file(f)
                idx_file += 1
                filepath = f'{output_file_path}-{idx_file}.json'
                f = open(filepath, 'w', encoding='utf-8')
                f.write('[')
            if i != 0 and i % 10000 == 0:
                print(f'Articles processed: {i}')
            i += 1
    close_hp_file(f)

def split_sentences(text, nlp):
    return '[SEP]'.join([s.text for s in nlp(text).sents])

def preprocess_fake_news(rootdir, nlp):
    for path, _, files in os.walk(rootdir):
        for name in files:
            filename = os.path.join(path, name)
            print(f'Processing {filename}')
            dataset = pd.read_csv(filename, sep='\t', names=['text', 'label'])
            dataset['text'] = dataset['text'].apply(split_sentences, args=(nlp,))
            dataset.to_csv(filename, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument('--hp_train_data', type=str, help='Hyperpartisan train data', default='data/SemEval/articles-training-20180831.xml')
    parser.add_argument('--hp_train_labels', type=str, help='Hyperpartisan train labels', default='data/SemEval/ground-truth-training-20180831.xml')
    parser.add_argument('--hp_train_output_prefix', type=str, help='Hyperpartisan train output file prefix', default='data/SemEval/articles-training')
    parser.add_argument('--hp_valid_data', type=str, help='Hyperpartisan valid data', default='data/SemEval/articles-validation-20180831.xml')
    parser.add_argument('--hp_valid_labels', type=str, help='Hyperpartisan valid labels', default='data/SemEval/ground-truth-validation-20180831.xml')
    parser.add_argument('--hp_valid_output_prefix', type=str, help='Hyperpartisan valid output file prefix', default='data/SemEval/articles-valid')
    parser.add_argument('--hp_byarticle_data', type=str, help='Hyperpartisan byarticles data', default='data/SemEval/articles-training-byarticle-20181122.xml')
    parser.add_argument('--hp_byarticle_labels', type=str, help='Hyperpartisan byarticles labels', default='data/SemEval/ground-truth-training-byarticle-20181122.xml')
    parser.add_argument('--hp_byarticle_output_prefix', type=str, help='Hyperpartisan byarticle dataset output file prefix', default='data/SemEval/dataset-byarticle')
    parser.add_argument('--hp_max_filesize', type=int, help='Maximum file size in MB for hyperpartisan dataset', default=400)
    parser.add_argument('--debate_datapath', type=str, help='Path to Debate Persuasiveness dataset', default='data/DebatePersuasiveness/DebateArguments.txt')
    parser.add_argument('--debate_output_dir', type=str, help='Path to Debate Persuasiveness output dir', default='data/DebatePersuasiveness/')
    parser.add_argument('--debate_output_prefix', type=str, help='Debate Persuasisveness output file', default='persuasiveness_dataset')
    parser.add_argument('--fake_news_rootdir', type=str, help='Path to FakeNews dataset', default='data/FakeNews')

    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["parser"]) 
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    # print('Preprocessing hyperpartisan training dataset')
    # preprocess_hp_dataset(args.hp_train_data, args.hp_train_labels, args.hp_train_output_prefix, nlp)
    # print('Preprocessing hyperpartisan validation dataset')
    # preprocess_hp_dataset(args.hp_valid_data, args.hp_valid_labels, args.hp_valid_output_prefix, nlp)
    print('Preprocessing hyperpartisan byarticle dataset')
    preprocess_hp_dataset(args.hp_byarticle_data, args.hp_byarticle_labels, args.hp_byarticle_output_prefix, nlp)

    print('Preprocessing persuasiveness datasets')
    debates_df = parse_debate_dataset(args.debate_datapath, nlp)
    save_debates_ds(debates_df, args)
    
    print('Preprocessing FakeNews datasets')
    preprocess_fake_news(args.fake_news_rootdir, nlp)