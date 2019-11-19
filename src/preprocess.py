"""
Oct 2019
Xinru Yan

This script is for preprocessing data to run the persuasion model.

Input files of this script:
    smc_dataset:
        to generate the csv files:
            Combine all csvs of Annotations - name_20.csv (all.csv)
        to generate the pkl file:
            300_info.csv

Output files of this script:
    data/preprocessed/annotation_dataset.csv:
        columns:
            message_id, text, sent1, label1, ..., sent30, label30
        each row in the output file corresponds to all the rows in a Annotations-X.csv file with the same B2 value:
            message_id: the value of B2
            text: All the Unit columns with the same B2 value joined by a space in the order they appear in the input file.
            for each input rows:
                sentN: unit of the row
                labelN: er_label_1 value if B4=0 else Our Label value
    data/preprocessed/persuasion_dataset_text.csv: (the first two columns of annotation_dataset.csv)
        columns:
            message_id, text
        each filed corresponds to Annotations-X.csv:
            message_id: the value of B2
            text: All the Unit columns with the same B2 value joined by a space in the order they appear in the input file.
    data/preprocessed/persuasion_donation_label.pkl:
        a dictionary of {mid:score}
            mid: message_id = B2
            score: for each unique B2 in all.csv, compare B5 and B6 in 300_info.xlsx (B5 = proposed donation amount, B6 = actual amount)
                   0 if B5 is None and B6 is 0 (sincere_non_donors)
                   1 if B5 > 0 and B6 > 0 (sincere_donors)
                   -1 if B5 >0 and B6 = 0 (insincere_donors)

Usage:

Please test the data file format by running the DataLoader.py

"""
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
    PERSUADER = 'er_label_1'
    OUR_PERSUADEE = 'Poss_Labels'

    @property
    def column_name(self):
        return self.value


class UnAnnotationHeaders(Enum):
    CONVERSATION_ID = 'B2'
    SPEAKER_FLAG = 'B4'
    SENT = 'Unit'
    PERSUADER = 'er_label_1'
    PERSUADEE = 'ee_label_1'

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

    def add_sent_only(self, sent):
        self.sents += [sent]

    def __len__(self):
        return len(self.sents)

    @property
    def dict(self):
        d = {'message_id': self.message_id,
             'text': ' '.join(self.sents),
             **{f'sent{i + 1}': sent for i, sent in enumerate(self.sents)},
             **{f'label{i + 1}': label for i, label in enumerate(self.labels)}}
        return d


def read_unlabled(file, convs):
    un_conversations = {}
    with open(file, newline='', mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[AnnotationHeaders.CONVERSATION_ID.column_name]
            if message_id not in convs:
                if message_id not in un_conversations:
                    un_conversations[message_id] = Conversation(message_id)
                speaker = row[AnnotationHeaders.SPEAKER_FLAG.column_name]
                if speaker == '0':
                    sent = row[AnnotationHeaders.SENT.column_name]
                    un_conversations[message_id].add_sent_only(sent)
    return un_conversations


@click.command()
@click.option('-i', '--dataset-dir', default='../data/smc_dataset')
@click.option('-o', '--output-dir', default='../data/preprocessed_politeness_ER_4')
def main(dataset_dir, output_dir):
    conversations = {}
    annotation_filepaths = glob.glob(f'{dataset_dir}/ER_4.csv')
    for annotation_filepath in annotation_filepaths:
        with open(annotation_filepath, newline='', mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            turn = []
            for row in reader:
                message_id = row[AnnotationHeaders.CONVERSATION_ID.column_name]
                if message_id not in conversations:
                    conversations[message_id] = Conversation(message_id)

                speaker = row[AnnotationHeaders.SPEAKER_FLAG.column_name]

                if speaker == '0':
                    sent = row[AnnotationHeaders.SENT.column_name]
                    label_header = AnnotationHeaders.OUR_PERSUADEE
                    label = row[label_header.column_name]
                    if label == '':
                        label = "other"
                    conversations[message_id].add_sent(sent, label)
    print(f'labeled conv number {len(conversations)}')
    print(f'reading in 300 unlabeled')
    un_annotation_file_1 = dataset_dir + '/300_dialog.csv'
    un_conversations_1 = read_unlabled(un_annotation_file_1, conversations)
    print(f'reading in rest unlabeled')
    un_annotation_file_2 = dataset_dir + '/full_dialog.csv'
    un_conversations_2 = read_unlabled(un_annotation_file_2, conversations)
    un_conversations = {**un_conversations_1, **un_conversations_2}
    print(f'total unlabeled conv {len(un_conversations)}')

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        click.secho(f'Warning overwriting old output in "{output_dir}"!', fg='red')

    with open(f'{output_dir}/annotation_dataset.csv', 'w') as csvfile:
        headers = ['message_id', 'text']
        for i in range(max(map(len, conversations.values()))):
            headers += [f'sent{i+1}', f'label{i+1}']
        writer = csv.DictWriter(csvfile, headers)
        writer.writeheader()
        max_length = max(len(item) for item in conversations.values())
        print(f'max length {max_length}')
        for conversation in conversations.values():
            writer.writerow(conversation.dict)
    with open(f'{output_dir}/persuasion_dataset_text.csv', 'w') as csvfile:
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
            if row['B4'] == '0':
                message_id_to_info[row['B2']] = row
    with open(f'{dataset_dir}/full_info.csv', 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['B4'] == '0':
                message_id_to_info[row['B2']] = row

    total = {**conversations, **un_conversations}
    print(f'total number of convs {len(total)}')
    for message_id in total.keys():
        info = message_id_to_info[message_id]
        b6 = info['B6']
        if b6 == '0' or b6 == '0.0':
            donation_labels[message_id] = 0
        else:
            donation_labels[message_id] = 1
        # b5, b6 = info['B5'], info['B6']
        # if (b5 == '' and b6 == '0') or (b5 == '0' and b6 == '0'):
        #     donation_labels[message_id] = 0
        # elif (b5 and float(b5) > 0 and float(b6) > 0) or (b5 == '' and float(b6) > 0) or (b5 == '0' and float(b6) > 0):
        #     donation_labels[message_id] = 1
        # elif b5 and float(b5) > 0 and b6 == '0':
        #     donation_labels[message_id] = 0
        # else:
        #     print(f'{message_id} does not belong to any classes {b5, b6}')

    with open(f'{output_dir}/persuasion_donation_label.pkl', 'wb') as picklefile:
        pickle.dump(donation_labels, picklefile)


if __name__ == '__main__':
    main()