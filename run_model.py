import torch
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
)


tokenizer = AutoTokenizer.from_pretrained(
    'microsoft/DialoGPT-small', padding_side='left')
model = AutoModelWithLMHead.from_pretrained('output-small')

chat_history_ids = []

# Let's chat for 4 lines
for step in range(20):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> User:") + tokenizer.eos_token, return_tensors='pt')
    # print(new_user_input_ids)

    # clear chat history every 3 lines
    if step % 4 == 0:
        chat_history_ids = chat_history_ids[:, -
                                            1:] if len(chat_history_ids) > 0 else torch.tensor([])

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat(
        [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids, max_length=400,  # oriignal 200
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=19,
        do_sample=True,
        top_k=300,  # original 100
        top_p=0.75,
        temperature=0.8
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # safety for when the response is unintelligible
    if (response.count("!") > 8) or (response.count("?") > 8) or (response.count(".") > 8):
        response = "I... I'm sorry, I can't think about that, it hurts."

    # pretty print last ouput tokens from bot
    print("Faye: {}".format(response))
