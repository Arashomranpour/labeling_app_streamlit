import streamlit as st
from pytube import YouTube
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from gensim import corpora,models
import re
import gensim
import heapq
import whisper
import os

# --------------------------------

insurance_keywords = ['actuary', 'claims', 'coverage', 'deductible', 'policyholder', 'premium', 'underwriter', 'risk assessment', 'insurable interest', 'loss ratio', 'reinsurance', 'actuarial tables', 'property damage', 'liability', 'flood insurance', 'term life insurance', 'whole life insurance', 'health insurance', 'auto insurance', 'homeowners insurance', 'marine insurance', 'crop insurance', 'catastrophe insurance', 'umbrella insurance', 'pet insurance', 'travel insurance', 'professional liability insurance', 'disability insurance', 'long-term care insurance', 'annuity', 'pension plan', 'group insurance', 'insurtech', 'insured', 'insurer', 'subrogation', 'adjuster', 'third-party administrator', 'excess and surplus lines', 'captives', 'workers compensation', 'insurance fraud', 'health savings account', 'health maintenance organization', 'preferred provider organization']

finance_keywords = ['asset', 'liability', 'equity', 'capital', 'portfolio', 'dividend', 'financial statement', 'balance sheet', 'income statement', 'cash flow statement', 'statement of retained earnings', 'financial ratio', 'valuation', 'bond', 'stock', 'mutual fund', 'exchange-traded fund', 'hedge fund', 'private equity', 'venture capital', 'mergers and acquisitions', 'initial public offering', 'secondary market', 'primary market', 'securities', 'derivative', 'option', 'futures', 'forward contract', 'swaps', 'commodities', 'credit rating', 'credit score', 'credit report', 'credit bureau', 'credit history', 'credit limit', 'credit utilization', 'credit counseling', 'credit card', 'debit card', 'ATM', 'bankruptcy', 'foreclosure', 'debt consolidation', 'taxes', 'tax return', 'tax deduction', 'tax credit', 'tax bracket', 'taxable income']

banking_capital_markets_keywords = ['bank', 'credit union', 'savings and loan association', 'commercial bank', 'investment bank', 'retail bank', 'wholesale bank', 'online bank', 'mobile banking', 'checking account', 'savings account', 'money market account', 'certificate of deposit', 'loan', 'mortgage', 'home equity loan', 'line of credit', 'credit card', 'debit card', 'ATM', 'automated clearing house', 'wire transfer', 'ACH', 'SWIFT', 'international banking', 'foreign exchange', 'forex', 'currency exchange', 'central bank', 'Federal Reserve', 'interest rate', 'inflation', 'deflation', 'monetary policy', 'fiscal policy', 'quantitative easing', 'securities', 'stock', 'bond', 'mutual fund', 'exchange-traded fund', 'hedge fund', 'private equity', 'venture capital', 'investment management', 'portfolio management', 'wealth management', 'financial planning']

healthcare_life_sciences_keywords = ['medical device', 'pharmaceutical', 'biotechnology', 'clinical trial', 'FDA', 'healthcare provider', 'healthcare plan', 'healthcare insurance', 'patient', 'doctor', 'nurse', 'pharmacist', 'hospital', 'clinic', 'healthcare system', 'healthcare policy', 'public health', 'healthcare IT', 'electronic health record', 'telemedicine', 'personalized medicine', 'genomics', 'proteomics', 'clinical research', 'drug development', 'drug discovery', 'medicine', 'health']

law_keywords = ['law', 'legal', 'attorney', 'lawyer', 'litigation', 'arbitration', 'dispute resolution', 'contract law', 'intellectual property', 'corporate law', 'labor law', 'tax law', 'real estate law', 'environmental law', 'criminal law', 'family law', 'immigration law', 'bankruptcy law']

sports_keywords = ['sports', 'football', 'basketball', 'baseball', 'hockey', 'soccer', 'golf', 'tennis', 'olympics', 'athletics', 'coaching', 'sports management', 'sports medicine', 'sports psychology', 'sports broadcasting', 'sports journalism', 'esports', 'fitness']

media_keywords = ['media', 'entertainment', 'film', 'television', 'radio', 'music', 'news', 'journalism', 'publishing', 'public relations', 'advertising', 'marketing', 'social media', 'digital media', 'animation', 'graphic design', 'web design', 'video production']

manufacturing_keywords = ['manufacturing', 'production', 'assembly', 'logistics', 'supply chain', 'quality control', 'lean manufacturing', 'six sigma', 'industrial engineering', 'process improvement', 'machinery', 'automation', 'aerospace', 'automotive', 'chemicals', 'construction materials', 'consumer goods', 'electronics', 'semiconductors']

automotive_keywords = ['automotive', 'cars', 'trucks', 'SUVs', 'electric vehicles', 'hybrid vehicles', 'autonomous vehicles', 'car manufacturing', 'automotive design', 'car dealerships', 'auto parts', 'vehicle maintenance', 'car rental', 'fleet management', 'telematics']

telecom_keywords = ['telecom', 'telecommunications', 'wireless', 'networks', 'internet', 'broadband', 'fiber optics', '5G', 'telecom infrastructure', 'telecom equipment', 'VoIP', 'satellite communications', 'mobile devices', 'smartphones', 'telecom services', 'telecom regulation', 'telecom policy']


information_technology_keywords = [
    "Artificial intelligence", "Machine learning", "Data Science", "Big Data", "Cloud Computing",
    "Cybersecurity", "Information security", "Network security", "Blockchain", "Cryptocurrency",
    "Internet of things", "IoT", "Web development", "Mobile development", "Frontend development",
    "Backend development", "Software engineering", "Software development", "Programming",
    "Database", "Data analytics", "Business intelligence", "DevOps", "Agile", "Scrum",
    "Product management", "Project management", "IT consulting", "IT service management", 
    "ERP", "CRM", "SaaS", "PaaS", "IaaS", "Virtualization", "Artificial reality", "AR", "Virtual reality",
    "VR", "Gaming", "E-commerce", "Digital marketing", "SEO", "SEM", "Content marketing",
    "Social media marketing", "User experience", "UX design", "UI design", "Cloud-native",
    "Microservices", "Serverless", "Containerization"
]

industries = {
    'Insurance': insurance_keywords,
    'Finance': finance_keywords,
    'Banking': banking_capital_markets_keywords,
    'Healthcare': healthcare_life_sciences_keywords,
    'Legal': law_keywords,
    'Sports': sports_keywords,
    'Media': media_keywords,
    'Manufacturing': manufacturing_keywords,
    'Automotive': automotive_keywords,
    'Telecom': telecom_keywords,
    'IT': information_technology_keywords
}
# --------------------------------


def label_topic(text):
    counts={}
    for industry,keywords in industries.items():
        count=sum([1 for keyword in keywords if re.search(r"\b{}\b".format(keyword),text,re.IGNORECASE)])
        counts[industry]=count
        
    top_indus=heapq.nlargest(3,iterable=counts,key=counts.get)
    if  len(top_indus)==1:
        return top_indus[0]
    else:
        return top_indus

        
       
    
    
    
def preprocess_text(text):
    tokens=gensim.utils.simple_preprocess(text)
    stop_words=gensim.parsing.preprocessing.STOPWORDS
    preprocessed_text=[[token for token in tokens if token not in stop_words]]
    return preprocessed_text    


def topic_model(transcript_text,number_topic=3,number_words=1):
    
    
    preprocessed_text=preprocess_text(transcript_text)
    diction=corpora.Dictionary(preprocessed_text)
    corpus=[diction.doc2bow(text) for text in preprocessed_text]
    model=models.LdaModel(corpus=corpus,id2word=diction,num_topics=number_topic)
    
    topics=[]
    for idx,topic in model.print_topics(-1,num_words=number_words):
        topic_word=[word.split("*")[1].replace('"',"").strip() for word in topic.split("+")]
        topics.append((f"Topic {idx}",topic_word))   
        
    return topics 
    

st.set_page_config(layout="wide")
st.title("labeling app") 


@st.cache_resource
def load_model():
    model=whisper.load_model("base")
    return model

def save_video(url,video_file):
    youtubeobjct=YouTube(url)
    youtubeobjct=youtubeobjct.streams.get_lowest_resolution()
    try:
        video_file=youtubeobjct.download()
    except:
        st.error("Error downloading video")
    return video_file

# def audio_save(url):
#     yt=YouTube(url)
#     video=yt.streams.filter(only_audio=True).first()
#     out_fule=video.download()
#     base,ext=os.path.splitext(out_fule)
#     file_name=base+".mp3"
#     try:
#         os.rename(out_fule,file_name)
#     except WindowsError:
#         os.remove(file_name)
#         os.rename(out_fule,file_name)
        
#     audio_file=Path(file_name).stem(".mp3")
#     video_file=save_video(url,Path(file_name).stem(".mp4"))
#     return yt.title,audio_file,video_file
    
def audio_save(url):
    try:
        yt = YouTube(url)
        # Get only the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        out_file = audio_stream.download()  # Download the audio
        base, ext = os.path.splitext(out_file)
        audio_file = base + ".mp3"
        
        # Rename to .mp3 if necessary
        try:
            os.rename(out_file, audio_file)
        except WindowsError:
            os.remove(audio_file)
            os.rename(out_file, audio_file)
        
        return yt.title, audio_file  # Return only the audio information
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None  # Return None on failure

    
    
def audio_trans(audio_file):
    model=load_model()
    res=model.transcribe(audio_file)
    transcript=res["text"]
    return transcript
    
choice=st.sidebar.selectbox("Select your type",["on Text","on Video","on csv"])

if choice=="on Text":
    st.subheader("labeling based on text")
    
    text_in=st.text_area("Paste here",height=400)
    if text_in is not None:
        
        if st.button("Analyze"):
            col1,col2,col3=st.columns(3)
        
            with col1:
            # st.write(text_in)
                st.info("text in below :")
                st.success(text_in)   
            with col2:
                st.info("topics :")
                
                topics=topic_model(text_in)
                for topic in topics:
                    st.success(f"{topic[0]}: {", ".join(topic[1])}")
            with col3:
                st.info("Labeled text :")
                labeling_text=text_in
                indus=label_topic(labeling_text)
                # st.success(f"Topic Labeling : {indus}")      
                st.markdown("**Topic Labeling Industry**")
                st.write(indus)          
                





               
if choice=="on Video":
    st.subheader("labeling based on video")
    # uploaded_vid=st.file_uploader("upload your video",type=["mp4"])
    url=st.text_input("Enter YT URL")
    # if uploaded_vid is not None:
    # if st.button("Analyze Video"):
    #     col1,col2,col3=st.columns(3)
    #     with col1:
    #         st.info("Video Uploaded")
    #         video_tile,audio_filename,video_filename=audio_save(url)
    #         st.video(video_filename)
    #     with col2:
    #         st.info("Transcript")
    #         trans_res=audio_trans(audio_filename)
    #         st.success(trans_res)
    #     with col3:
    #         st.info("Topic modeling")
    #         labeling_trs=trans_res
    #         indist=label_topic(labeling_trs)
    #         st.markdown("**Topic Labeling**")
    #         st.write(indist)
    if st.button("Analyze Video"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Audio from YouTube")
            video_title, audio_filename = audio_save(url)  # No video, just audio
            if audio_filename:
                st.success(f"Audio file: {audio_filename}")
            else:
                st.error("Failed to download audio.")
        with col2:
            if audio_filename:
                st.info("Transcription")
                transcript = audio_trans(audio_filename)  # Transcribe the audio
                st.success(transcript)
        with col3:
            if audio_filename:
                st.info("Topic Modeling")
                labeling_text = transcript
                industry_labels = label_topic(labeling_text)
                st.markdown("**Topic Labeling**")
                st.write(industry_labels)

    
    
if choice=="on csv":
    st.subheader("labeling based on csv")
    uploaded=st.file_uploader("Upload your csv file",type=["csv"])
    if uploaded is not None:
        if st.button("Analysis csv"):
            col1,col2=st.columns(2)
            with col1:
                st.info("CSV file uploaded")
                csv_file=uploaded.name
                with open(os.path.join(csv_file),"wb") as f:
                    f.write(uploaded.getbuffer())
                print(csv_file)
                df=pd.read_csv(csv_file,encoding="unicode_escape")
                st.dataframe(df)
                
            with col2:    
                data_list=df["Data"].to_list()
                indust_list=[]
                for i in data_list:
                    indust=label_topic(i)
                    indust_list.append(indust)
                df["Industry"]=indust_list
                st.info("Labeling")
                st.markdown("**Topic Labeling Industry**")
                st.write(indust_list) 
                st.dataframe(df)
                
    