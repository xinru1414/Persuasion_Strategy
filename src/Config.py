sent_label_set_1 = ['direct-rejection', 'self-pity', 'deflect-responsibility', 'attack-credibility', 'organization-inquiry', 'personal-choice', 'delay-tactic', 'hesitance', 'nitpicking', 'not-a-strategy']
sent_label_set_2 = ['ERPos+', 'ERPos-', 'ERNeg+', 'ERNeg-','EEPos+', 'EEPos-', 'EENeg+', 'EENeg-']
dataset_text = "../data/preprocessed/persuasion_dataset_text.csv"
dataset_with_annotation = "../data/preprocessed/annotation_dataset.csv"


class ModelConfig:
    def __init__(self, embedding_size=128, hidden_size=256, feature_size=10, output_size=2, learning_rate=5e-5):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.learning_rate = learning_rate


class ConversationConfig:
    sent_num = 45
    conv_label_num = 2
    sent_label_set = sent_label_set_1
    sent_label_num = len(sent_label_set)
    conv_pad_label = len(sent_label_set_1) + 1


def zeroed_class_dict(elements: int):
    return {i: 0 for i in range(elements)}

