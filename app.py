# import streamlit as st
# from transformers import pipeline

# # تحميل موديل تحليل المشاعر
# pipe = pipeline(
#     "text-classification",
#     model="cardiffnlp/twitter-roberta-base-sentiment"
# )


# st.title("Hello, Streamlit!")

# text = st.text_input("Enter some text to analyze its sentiment:")


# if text:
#     result = pipe(text)

#     st.write("Sentiment Analysis Result:")
#     st.write(result[0]['label'])
#     st.write(f"Confidence: {result[0]['score']:.2f}")



import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "MK-Mostafa/marian-finetuned-translation-ar-to-en"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "MK-Mostafa/marian-finetuned-translation-ar-to-en"
)

st.title("Arabic to English Translator")

text = st.text_area("Enter Arabic text")

if text:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)

    translation = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    st.success(translation)