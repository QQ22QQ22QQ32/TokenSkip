import os
import json
from tqdm import tqdm
from llmlingua import PromptCompressor


def load_jsonl(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def save_jsonl(data, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for item in data:
        with open(output_path, 'a+', encoding='utf-8') as f:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

def filter_correct_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions.jsonl",
                           output_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_correct.jsonl"):
    """
    Filter the correct outputs from the data.
    """
    data = load_jsonl(input_path)
    correct_data = []
    for i in range(len(data)):
        if data[i]['accuracy']:
            correct_data.append(data[i])
    print(f"Original Samples: {len(data)}, Correct Samples: {len(correct_data)}, Accuracy: {len(correct_data) / len(data)}")
    save_jsonl(correct_data, output_path)


def filter_formatted_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_correct.jsonl",
                             output_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_formatted.jsonl", model_type="qwen"):
    """
    Filter the formatted outputs from the data. Extract COT from th outputs.
    """
    data = load_jsonl(input_path)
    formatted_data = []
    for i in range(len(data)):
        if data[i]['cot_length'] > 500:
            continue
        if model_type == "llama3":
            spans = data[i]["output"].split('\n\nThe final answer is:')
            if len(spans) == 2:
                data[i]["cot"] = spans[0]
                formatted_data.append(data[i])
        elif model_type == "qwen":
            formatted_data.append(data[i])
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")
    print(f"Original Samples: {len(data)}, Formatted Samples: {len(formatted_data)}")
    save_jsonl(formatted_data, output_path)

def _deduplicate_tokens(tokens):
    """Deduplicate tokens while preserving their original order."""
    seen = set()
    deduped = []
    for token in tokens:
        if token is None:
            continue
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def LLMLingua(data, compression_ratio=0.5, model_type="qwen",
              llmlingua_path="/your_model_path/llmlingua-2-xlm-roberta-large-meetingbank",
              entropy_threshold=None, entropy_keep_topk=None,
              token_stats_field="token_stats", cot_token_count=None,
              record_entropy_policy=False):
    """
    Compress the CoT outputs with LLMLingua-2.
    """
    if model_type == "llama3":
        cot_type = "cot"
    elif model_type == "qwen":
        cot_type = "model_output"
    else:
        raise ValueError(f"Model Type {model_type} is not supported.")

    if entropy_threshold is not None and entropy_keep_topk is not None:
        raise ValueError("Only one of entropy_threshold or entropy_keep_topk can be provided.")

    llm_lingua = PromptCompressor(
        model_name=llmlingua_path,
        use_llmlingua2=True,  # Whether to use llmlingua-2
    )
    compressed_data = []
    base_force_tokens = []
    if model_type == "llama3":
        base_force_tokens = ['Step', ':']
    elif model_type == "qwen":
        base_force_tokens = []
    else:
        raise ValueError(f"Model Type {model_type} is not supported.")
    for i in tqdm(range(len(data))):
        cot_output = data[i][cot_type]
        token_stats = data[i].get(token_stats_field) or []
        cot_limit = cot_token_count if cot_token_count is not None else data[i].get("cot_token_count")
        if isinstance(cot_limit, int):
            if cot_limit > 0:
                token_stats = token_stats[:cot_limit]
            elif cot_limit == 0:
                token_stats = []

        indexed_stats = list(enumerate(token_stats))
        selected_stats = []
        if entropy_keep_topk is not None and entropy_keep_topk > 0:
            from heapq import nlargest

            topk = nlargest(entropy_keep_topk, indexed_stats,
                            key=lambda x: x[1].get("entropy", float('-inf')))
            topk_indices = {idx for idx, _ in topk}
            selected_stats = [stat for idx, stat in indexed_stats if idx in topk_indices]
        elif entropy_threshold is not None:
            selected_stats = [stat for _, stat in indexed_stats
                              if stat.get("entropy") is not None and stat.get("entropy") >= entropy_threshold]

        dynamic_force_tokens = []
        for stat in selected_stats:
            token_piece = (stat.get("decoded_token_piece")
                           or stat.get("token_piece")
                           or stat.get("decoded_token")
                           or stat.get("token"))
            if token_piece is None:
                continue
            dynamic_force_tokens.append(token_piece)
        dynamic_force_tokens = _deduplicate_tokens(dynamic_force_tokens)

        combined_force_tokens = _deduplicate_tokens(base_force_tokens + dynamic_force_tokens)

        compress_kwargs = {
            "rate": compression_ratio,
        }
        if model_type == "llama3":
            compress_kwargs.update({
                "force_tokens": combined_force_tokens,
                "force_reserve_digit": True,
                "drop_consecutive": True,
            })
        elif model_type == "qwen":
            if combined_force_tokens:
                compress_kwargs["force_tokens"] = combined_force_tokens

        compressed_prompt = llm_lingua.compress_prompt(cot_output, **compress_kwargs)
        compressed_data_line = {
            'question': data[i]['messages'][0]['content'],
            'input': data[i]['prompt'],
            'output': data[i]['model_output'],
            'answer': data[i]['answer'],
            'model_answer': data[i]['prediction'],
            'is_correct': data[i]['accuracy'],
            'cot': data[i][cot_type],
            'compressed_cot': compressed_prompt['compressed_prompt'],
            'original_cot_tokens': compressed_prompt['origin_tokens'],
            'compressed_cot_tokens': compressed_prompt['compressed_tokens'],
            'compression_rate': compressed_prompt['rate']
        }
        if record_entropy_policy and (entropy_threshold is not None or entropy_keep_topk is not None):
            compressed_data_line['entropy_policy'] = {
                'entropy_threshold': entropy_threshold,
                'entropy_keep_topk': entropy_keep_topk,
                'forced_token_count': len(dynamic_force_tokens),
                'token_stats_field': token_stats_field,
            }
            if isinstance(cot_limit, int):
                compressed_data_line['entropy_policy']['cot_token_count'] = cot_limit
        if dynamic_force_tokens:
            compressed_data_line['dynamic_force_tokens'] = dynamic_force_tokens
        compressed_data.append(compressed_data_line)
    return compressed_data


def compress_cot_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_formatted.jsonl",
                         output_dir="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Compression", model_type="qwen",
                         llmlingua_path="llmlingua-2-xlm-roberta-large-meetingbank",
                         entropy_threshold=None, entropy_keep_topk=None,
                         token_stats_field="token_stats", cot_token_count=None,
                         record_entropy_policy=False):
    """
    Compress the CoT outputs with various compression ratios using LLMLingua-2.
    """
    data = load_jsonl(input_path)
    ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5]
    for compression_ratio in ratio_list:
        output_path = os.path.join(output_dir, f"train_outputs_compressed_ratio_{compression_ratio}.jsonl")
        compressed_data = LLMLingua(
            data,
            compression_ratio=compression_ratio,
            model_type=model_type,
            llmlingua_path=llmlingua_path,
            entropy_threshold=entropy_threshold,
            entropy_keep_topk=entropy_keep_topk,
            token_stats_field=token_stats_field,
            cot_token_count=cot_token_count,
            record_entropy_policy=record_entropy_policy,
        )
        save_jsonl(compressed_data, output_path)
        get_average_compress_rate(compressed_data)

def get_average_compress_rate(data):
    compress_rate = 0
    for i in range(len(data)):
        compress_rate += data[i]['compressed_cot_tokens'] / data[i]['original_cot_tokens']
    compress_rate = compress_rate / len(data)
    print(f"Average Compression Rate: {compress_rate}")


def data_processing_gsm8k(input_dir="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/", model_type="qwen",
                          llmlingua_path="/your_model_path/llmlingua-2-xlm-roberta-large-meetingbank",
                          entropy_threshold=None, entropy_keep_topk=None,
                          token_stats_field="token_stats", cot_token_count=None,
                          record_entropy_policy=False):
    """
    The overall pipeline to process the GSM8K data.
    """
    input_path = os.path.join(input_dir, "Original/train/samples/predictions.jsonl")
    correct_path = os.path.join(input_dir, "Original/train/samples/predictions_correct.jsonl")
    formatted_path = os.path.join(input_dir, "Original/train/samples/predictions_formatted.jsonl")
    compressed_dir = os.path.join(input_dir, "Compression")

    filter_correct_outputs(input_path=input_path, output_path=correct_path)
    filter_formatted_outputs(input_path=correct_path, output_path=formatted_path, model_type=model_type)
    compress_cot_outputs(
        input_path=formatted_path,
        output_dir=compressed_dir,
        model_type=model_type,
        llmlingua_path=llmlingua_path,
        entropy_threshold=entropy_threshold,
        entropy_keep_topk=entropy_keep_topk,
        token_stats_field=token_stats_field,
        cot_token_count=cot_token_count,
        record_entropy_policy=record_entropy_policy,
    )

if __name__ == '__main__':
    data_processing_gsm8k()


