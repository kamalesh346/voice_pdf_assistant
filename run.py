from audio import record_audio, speech_to_text
from rag import generate_answer_stream, load_or_create_index, load_conversation, save_conversation
from pathlib import Path
import time
from tts import interrupt_speech

def retrieve_context(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def process_voice_input():
    audio_file = record_audio()

    if audio_file:
        question = speech_to_text(audio_file)
        print("You said:", question)
        return question

    return ""


def answer_question(question, retriever, memory):

    start = time.time()

    context = retrieve_context(question, retriever)

    answer = generate_answer_stream(question, context, memory)

    end = time.time()

    print(f"\n⏱️ Response time: {end-start:.2f}s")

    return answer


# -----------------------------

pdf_file = input("Enter PDF file path (inside data/): ")

vectorstore = load_or_create_index(pdf_file)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

pdf_name = Path(pdf_file).stem

memory = load_conversation(pdf_name)


print("\n🎙️ Voice PDF Assistant")
print("=" * 40)

mode = input("Choose mode: 'v' for voice or 't' for text: ").lower()

if mode not in ["v", "t"]:
    mode = "t"


while True:

    if mode == "v":

        question = process_voice_input()
        interrupt_words = ["stop", "wait", "hold on", "cancel"]

        if any(word in question for word in interrupt_words):
            print("Interrupt detected")
            interrupt_speech()
            continue

    else:

        question = input("Ask question: ")

    if not question.strip():
        continue


    if "exit" in question.lower():

        save_conversation(pdf_name, memory)

        print("Conversation saved.")

        break


    if "switch to voice" in question.lower():

        mode = "v"

        print("Switched to voice mode.")

        continue


    if "switch to text" in question.lower():

        mode = "t"

        print("Switched to text mode.")

        continue


    print(f"\nProcessing: {question}\n")

    answer = answer_question(question, retriever, memory)

    print(f"\nAssistant:\n{answer}\n")


save_conversation(pdf_name, memory)