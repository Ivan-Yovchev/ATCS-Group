import re
import pandas as pd
import argparse
import xml.etree.cElementTree as ET

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

def parse_debate_dataset(filepath):
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


def preprocess_hp_dataset(data_path, labels_path, output_file_path):
    """ Combines data and groundtruth labels xml files in a single tsv file
        Output file format:
            article_id label bias labeled_by title text
    """
    train_data_tree = ET.iterparse(data_path, events=('start', 'end'))
    train_label_tree = ET.iterparse(labels_path, events=('start', 'end'))

    with open(output_file_path, 'w') as f:
        i = 0
        root = False
        for event, elem in train_data_tree:  
            if event == 'start' and not root:
                root = elem
            if event == 'end' and elem.tag == 'article':
                article_id = elem.attrib.get('id')
                title = elem.attrib.get('title')
                # process article content
                xml = ET.tostring(elem, encoding='utf-8', method='xml').decode()
                text = ' '.join(re.findall(r'<p>(.*?)<\/p>', xml))
                # replace anchor tags and whitespace with single space
                text = re.sub(r'(<a.*>|</a>|\s{1,})', ' ', text)
                label, bias, labeled_by = get_label_data(article_id, train_label_tree)
                f.write(f'{article_id}\t{label}\t{bias}\t{labeled_by}\t{title}\t{text}\n')
                # clear prev elements from memory
                elem.clear()
                root.clear()
                if i != 0 and i % 10000 == 0:
                    print(f'Articles processed: {i}')
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument('--hp_train_data', type=str, help='Hyperpartisan train data', default='data/SemEval/articles-training-20180831.xml')
    parser.add_argument('--hp_train_labels', type=str, help='Hyperpartisan train labels', default='data/SemEval/ground-truth-training-20180831.xml')
    parser.add_argument('--hp_train_output_file', type=str, help='Hyperpartisan train output file', default='data/SemEval/articles-training.tsv')
    parser.add_argument('--hp_valid_data', type=str, help='Hyperpartisan valid data', default='data/SemEval/articles-validation-20180831.xml')
    parser.add_argument('--hp_valid_labels', type=str, help='Hyperpartisan valid labels', default='data/SemEval/ground-truth-validation-20180831.xml')
    parser.add_argument('--hp_valid_output_file', type=str, help='Hyperpartisan valid output file', default='data/SemEval/articles-valid.tsv')
    parser.add_argument('--debate_datapath', type=str, help='Path to Debate Persuasiveness dataset', default='data/DebatePersuasiveness/DebateArguments.txt')
    parser.add_argument('--debate_output_file', type=str, help='Debate Persuasisveness output file', default='data/DebatePersuasiveness/persuasiveness_dataset.json')

    args = parser.parse_args()

    preprocess_hp_dataset(args.hp_train_data, args.hp_train_labels, args.hp_train_output_file)
    preprocess_hp_dataset(args.hp_valid_data, args.hp_valid_labels, args.hp_valid_output_file)

    debates_df = parse_debate_dataset(args.debate_datapath)
    debates_df.to_json(args.debate_output_file, orient='records', indent=2)