import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# إعداد واجهة التطبيق
st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="💰")
st.title("💰 Financial Sentiment Analysis")
st.subheader("Llama 3.2 1B + LoRA")

# دالة لتحميل الموديل مرة واحدة وتخزينه في الكاش لتسريع الأداء
@st.cache_resource
def load_model():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    lora_id = "Adityaaaa468/Financial-sentiment-analysis_llama_3.2_1B"
    
    # تحميل الـ Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # تحميل الموديل الأساسي (بصيغة تناسب جهازك)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" # سيقوم باختيار cuda تلقائياً إذا توفرت
    )
    
    # تحميل أوزان الـ LoRA الخاصة بتحليل المشاعر المالية
    model.load_adapter(lora_id)
    
    return tokenizer, model

# عرض رسالة تحميل
with st.spinner("جاري تحميل الموديل... قد يستغرق ذلك دقيقة"):
    tokenizer, model = load_model()

# صندوق إدخال النص
user_input = st.text_area("أدخل الجملة المالية هنا (بالإنجليزي):", 
                          placeholder="e.g., The company's revenue increased by 20% this quarter.")

if st.button("تحليل المشاعر"):
    if user_input:
        # تجهيز المدخلات للموديل
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # عرض النتيجة
        st.success("result:")
        st.write(result)
    else:
        st.warning("من فضلك أدخلي نصاً للتحليل.")