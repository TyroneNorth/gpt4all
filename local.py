from pyllamacpp.model import Model


def new_text_callback(text: str):
    print(text, end="", flush=True)


def get_user_input():
    user_input = input("\nUser: ")
    return user_input


def update_conversation_history(history, user_message, ai_response):
    history.append("User: " + user_message)
    history.append("DREAMAI: " + ai_response)
    return history


def main():
    conversation_history = []

    initial_prompt = ("You are DREAMAI, a solo developer assistant, expert in writing code, "
                      "explaining computer science, and planning/organization.")

    model = Model(
        ggml_model='gpt4all-lora-quantized.bin.orig', n_ctx=512)

    user_message = " I want to build a social media app, Twitter competitor, with a focus on privacy and security."
    conversation_history = update_conversation_history(
        conversation_history, user_message, "")
    prompt = initial_prompt + "\n\n" + "\n".join(conversation_history)

    model.generate(prompt, n_predict=100,
                   new_text_callback=new_text_callback, n_threads=16, verbose=False)

    while True:
        user_input = get_user_input().strip()

        if user_input.lower() == "exit":
            break

        conversation_history = update_conversation_history(
            conversation_history, user_input, "")
        prompt = initial_prompt + "\n\n" + "\n".join(conversation_history)

        model.generate(prompt, n_predict=100,
                       new_text_callback=new_text_callback, n_threads=16, verbose=False)


if __name__ == "__main__":
    main()
