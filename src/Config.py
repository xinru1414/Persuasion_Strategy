sent_label_set_1 = ['direct-rejection', 'self-pity', 'deflect-responsibility', 'attack-credibility', 'organization-inquiry', 'personal-choice', 'delay-tactic', 'hesitance', 'nitpicking', 'not-a-strategy']
sent_label_set_EE = ['ERPos+', 'EEPos+', 'EENeg+', 'ERPos-', 'ERNeg+', 'ERNeg-', 'other']
sent_label_set_ER = ['ERPos+', 'EEPos+', 'EENeg+', 'EENeg-', 'EEPos-', 'other']


class ModelConfig:
    def __init__(self, embedding_size=128, hidden_size=256, feature_size=10, output_size=2, learning_rate=5e-5):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.learning_rate = learning_rate


class ConversationConfig:
    sent_num = 45 # 30 EE, 45 Resisting
    conv_label_num = 2
    sent_label_set = sent_label_set_1
    sent_label_num = len(sent_label_set)
    conv_pad_label = len(sent_label_set_1) + 1


class FiveFold:
    def __init__(self, i):
        self.dataset_text = f"../data/preprocessed_full_{i}/persuasion_dataset_text.csv"
        self.dataset_with_annotation = f"../data/preprocessed_full_{i}/annotation_dataset.csv"
        self.save_path = f'../model/full_{i}/model_attn_1.pkl'


def zeroed_class_dict(elements: int):
    return {i: 0 for i in range(elements)}

