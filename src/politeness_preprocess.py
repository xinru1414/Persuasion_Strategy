import csv
import glob
import os
import pickle
from enum import Enum

import click


class AnnotationHeaders(Enum):
    CONVERSATION_ID = 'B2'
    SPEAKER_FLAG = 'B4'
    SENT = 'Unit'
    LABELS = 'Poss_Labels'

    @property
    def column_name(self):
        return self.value


class Conversation:
    def __init__(self, message_id):
        self.message_id = message_id
        self.sents = []
        self.labels = []

    def add_sent(self, sent, label):
        self.sents += [sent]
        self.labels += [label]

    def __len__(self):
        return len(self.sents)

    @property
    def dict(self):
        d = {'message_id': self.message_id,
             'text': ' '.join(self.sents),
             **{f'sent{i + 1}': sent for i, sent in enumerate(self.sents)},
             **{f'label{i + 1}': label for i, label in enumerate(self.labels)}}
        return d


@click.command()
@click.option('-i', '--dataset-dir', default='../data/smc_dataset')
@click.option('-o', '--output-dir', default='../data/preprocessed')
def main(dataset_dir, output_dir):
    conversations = {}
    un_conversations = {}
    annotation_filepaths = glob.glob(f'{dataset_dir}/politeness_*.csv')
    for annotation_filepath in annotation_filepaths:
        with open(annotation_filepath, newline='', mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                message_id = row[AnnotationHeaders.CONVERSATION_ID.column_name]
                if message_id not in conversations:
                    conversations[message_id] = Conversation(message_id)

                sent = row[AnnotationHeaders.SENT.column_name]
                label_header = AnnotationHeaders.LABELS
                label = row[label_header.column_name]
                conversations[message_id].add_sent(sent, label)

    un_annotation_file = dataset_dir + '/300_dialog.csv'
    with open(un_annotation_file, newline='', mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[AnnotationHeaders.CONVERSATION_ID.column_name]
            if message_id not in conversations:
                if message_id not in un_conversations:
                    un_conversations[message_id] = Conversation(message_id)
                speaker = row[AnnotationHeaders.SPEAKER_FLAG.column_name]
                if speaker == '1':
                    sent = row[AnnotationHeaders.SENT.column_name]
                    un_conversations[message_id].add_sent(sent, label)
    assert (len(conversations.keys()) + len(un_conversations.keys()) == 300), 'total instances number should be 300'
    print(f'length of the annotated data {len(conversations.keys())}')
    print(f'length of the unannotated data {len(un_conversations.keys())}')

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        click.secho(f'Warning overwriting old output in "{output_dir}"!', fg='red')

    with open(f'{output_dir}/p_annotation_dataset.csv', 'w') as csvfile:
        headers = ['message_id', 'text']
        for i in range(max(map(len, conversations.values()))):
            headers += [f'sent{i+1}', f'label{i+1}']
        writer = csv.DictWriter(csvfile, headers)
        writer.writeheader()
        max_length = max(len(item) for item in conversations.values())
        print(f'max length {max_length}')
        for conversation in conversations.values():
            writer.writerow(conversation.dict)
    with open(f'{output_dir}/politeness_dataset_text.csv', 'w') as csvfile:
        headers = ['message_id', 'text']
        writer = csv.DictWriter(csvfile, headers, extrasaction='ignore')
        writer.writeheader()
        for conversation in un_conversations.values():
            writer.writerow(conversation.dict)

    donation_labels = {}
    message_id_to_info = {}
    with open(f'{dataset_dir}/300_info.csv', 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['B4'] == '1':
                message_id_to_info[row['B2']] = row
    total = {**conversations, **un_conversations}
    for message_id in total.keys():
        info = message_id_to_info[message_id]
        b5, b6 = info['B5'], info['B6']
        if (b5 == '' and b6 == '0') or (b5 == '0' and b6 == '0'):
            donation_labels[message_id] = 0
        elif (b5 and float(b5) > 0 and float(b6) > 0) or (b5 == '' and float(b6) > 0) or (b5 == '0' and float(b6) > 0):
            donation_labels[message_id] = 1
        elif b5 and float(b5) > 0 and b6 == '0':
            donation_labels[message_id] = 0
        else:
            print(f'{message_id} does not belong to any classes {b5, b6}')

    with open(f'{output_dir}/persuasion_donation_label.pkl', 'wb') as picklefile:
        pickle.dump(donation_labels, picklefile)


if __name__ == '__main__':
    main()