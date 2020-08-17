from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

def get_answer_using_bert(question, reference_text):
    # Load fine-tuned model for QA
    bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load Vocab as well
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Apply bert_tokenizer on input text
    input_ids = bert_tokenizer.encode(question, reference_text)
    input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    # Search index of first [SEP] token
    sep_location = input_ids.index(bert_tokenizer.sep_token_id)
    first_seg_len, second_seg_len = sep_location + 1, len(input_ids) - (sep_location + 1)
    seg_embedding = [0] * first_seg_len + [1] * second_seg_len

    # Run our example on model
    model_scores = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([seg_embedding]))
    ans_start_loc, ans_end_loc = torch.argmax(model_scores[0]), torch.argmax(model_scores[1])
    result = ' '.join(input_tokens[ans_start_loc:ans_end_loc + 1])

    # Return final result
    result = result.replace(' ##', '')
    return result


question = "How many parameters does BERT-large have?"
reference_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
print(get_answer_using_bert(question, reference_text))