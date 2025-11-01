# ❓ Help & Community Page
# FAQ, contact options, and language selection

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import load_custom_css, header, render_faq_section, render_contact_section

st.set_page_config(
    page_title="Help - CropShield AI",
    page_icon="❓",
    layout="wide"
)

load_custom_css()

# Page header
header("❓ Help & Support", "Get answers to your questions and connect with experts")

st.markdown("<br>", unsafe_allow_html=True)

# Language selector
st.markdown("### 🌐 Select Language / भाषा चुनें")

col1, col2 = st.columns([1, 3])

with col1:
    language = st.radio(
        "Choose your preferred language:",
        options=["English", "हिंदी (Hindi)"],
        index=0,
        key="language_selector"
    )

st.markdown("<br>", unsafe_allow_html=True)

# FAQ content based on language
if language == "English":
    st.markdown("### ❓ Frequently Asked Questions")
    
    # FAQ 1: Why is my plant sick?
    with st.expander("🌱 Why is my plant getting sick?", expanded=False):
        st.markdown("""
        Plants can become sick due to various factors:
        
        **1. Pathogen Infections:**
        - **Fungi**: Most common cause (rust, blight, mildew)
        - **Bacteria**: Causes soft rot, leaf spots, wilting
        - **Viruses**: Leads to stunted growth, mosaic patterns
        - **Nematodes**: Microscopic worms attacking roots
        
        **2. Environmental Stress:**
        - Extreme temperatures (too hot or too cold)
        - Water stress (drought or waterlogging)
        - Poor soil quality or nutrient deficiency
        - Inadequate sunlight or excessive shade
        
        **3. Poor Agricultural Practices:**
        - Improper spacing leading to poor air circulation
        - Contaminated tools spreading diseases
        - Using infected seeds or seedlings
        - Over-fertilization or under-fertilization
        
        **4. Weather Conditions:**
        - High humidity promoting fungal growth
        - Excessive rainfall spreading pathogens
        - Sudden temperature changes
        
        **🔍 Early Detection is Key:** Use CropShield AI regularly to detect diseases early when treatment is most effective!
        """)
    
    # FAQ 2: How can I prevent diseases?
    with st.expander("🛡️ How can I prevent diseases?", expanded=False):
        st.markdown("""
        **Preventive measures are the best defense against plant diseases:**
        
        **Before Planting:**
        - ✅ Use certified, disease-free seeds
        - ✅ Choose disease-resistant crop varieties
        - ✅ Ensure proper soil preparation and pH balance
        - ✅ Plan crop rotation to break disease cycles
        
        **During Growing Season:**
        - ✅ Maintain proper plant spacing for air circulation
        - ✅ Water at soil level, avoid wetting leaves
        - ✅ Remove and destroy infected plant parts immediately
        - ✅ Keep the field clean, remove weeds and debris
        - ✅ Use clean, sterilized tools and equipment
        
        **Soil Management:**
        - ✅ Add organic matter (compost, manure)
        - ✅ Ensure good drainage
        - ✅ Test soil regularly and adjust nutrients
        - ✅ Practice green manuring
        
        **Regular Monitoring:**
        - ✅ Inspect crops weekly for early signs
        - ✅ Use CropShield AI for quick disease detection
        - ✅ Keep records of disease occurrences
        - ✅ Act quickly when symptoms appear
        
        **Biological Control:**
        - ✅ Encourage beneficial insects
        - ✅ Use bio-pesticides when appropriate
        - ✅ Apply neem-based products preventively
        
        **🌟 Remember:** Prevention is always cheaper and more effective than cure!
        """)
    
    # FAQ 3: How to apply treatments safely?
    with st.expander("💊 How to apply treatments safely?", expanded=False):
        st.markdown("""
        **Safety Guidelines for Treatment Application:**
        
        **Personal Protection (Always Wear):**
        - 🧤 Chemical-resistant gloves
        - 😷 Respirator mask or N95 mask
        - 🥽 Safety goggles or face shield
        - 👕 Long-sleeved shirt and long pants
        - 👢 Boots (not sandals)
        - 🧢 Hat or head covering
        
        **Before Application:**
        - 📖 Read product label carefully
        - ⚖️ Measure exact dosage - never estimate
        - 🌡️ Check weather - avoid windy or rainy days
        - ⏰ Plan to spray early morning (6-9 AM) or evening (4-6 PM)
        - 📱 Inform family members and neighbors
        
        **During Application:**
        - 🚶 Walk backward to avoid spray contact
        - 💨 Spray with wind at your back
        - 🚫 Don't eat, drink, or smoke while spraying
        - 🚸 Keep children and pets away
        - 🐝 Avoid spraying on flowering plants (protect bees)
        
        **After Application:**
        - 🚿 Wash hands and face thoroughly
        - 👔 Remove and wash contaminated clothing separately
        - 🗑️ Dispose of empty containers properly (never reuse)
        - 🚰 Don't contaminate water sources
        - ⏳ Follow pre-harvest intervals before harvesting
        
        **Storage:**
        - 🔒 Store in original containers with labels
        - 🏠 Keep in locked, dry, well-ventilated area
        - 🚫 Away from food, feed, and living areas
        - 👶 Out of reach of children
        
        **Emergency:**
        - 📞 Keep poison control number handy
        - 🏥 Know location of nearest hospital
        - 🧴 Have clean water available for washing
        - 📋 Keep product label for medical reference
        
        **⚠️ Important:** If you feel sick during or after application, seek medical help immediately!
        """)
    
    # FAQ 4: Additional questions
    with st.expander("📱 How accurate is CropShield AI?", expanded=False):
        st.markdown("""
        **About CropShield AI Accuracy:**
        
        **Detection Accuracy:**
        - 🎯 85-95% accuracy for common crop diseases
        - 📊 Confidence scores indicate reliability
        - 🔬 Trained on thousands of real crop images
        
        **Confidence Score Guide:**
        - **90-100%**: Very high confidence - take immediate action
        - **75-89%**: High confidence - monitor closely and prepare treatment
        - **60-74%**: Moderate confidence - continue monitoring
        - **Below 60%**: Low confidence - consider re-scanning with better image
        
        **Best Results Tips:**
        - 📸 Take clear, focused images in natural light
        - 🔍 Capture affected areas up close
        - 🌅 Avoid shadows and glare
        - 📐 Include multiple leaves/parts if possible
        
        **Limitations:**
        - ⚠️ AI is a tool to assist, not replace expert consultation
        - 🔬 For complex cases, always consult agricultural experts
        - 📚 Continuous learning improves accuracy over time
        
        **💡 Pro Tip:** Take multiple photos from different angles for better diagnosis!
        """)
    
    with st.expander("🌦️ Does weather affect disease detection?", expanded=False):
        st.markdown("""
        **Yes! Weather plays a crucial role:**
        
        **High Humidity (>80%):**
        - Increases fungal disease risk
        - Promotes spore germination
        - Our AI adjusts confidence scores accordingly
        
        **Temperature:**
        - Hot weather (>35°C): Some diseases slow down
        - Cool weather (<20°C): Others become more active
        - We factor this into recommendations
        
        **Rainfall:**
        - Heavy rain spreads pathogens
        - Standing water creates disease-friendly conditions
        - We suggest avoiding treatment during rain
        
        **How We Use Weather Data:**
        - 📊 Adjusts confidence scores based on conditions
        - 💡 Provides climate-aware treatment advice
        - ⏰ Suggests optimal treatment timing
        
        **Recommendation:** Always add weather data when available for more accurate suggestions!
        """)
    
    with st.expander("💰 How much does CropShield AI cost?", expanded=False):
        st.markdown("""
        **Pricing & Availability:**
        
        **Current Status:**
        - 🎉 This is a **prototype/demo version**
        - 🆓 Free to use for testing and evaluation
        - 📱 Access via web browser (no app installation needed)
        
        **Future Plans:**
        - 📱 Mobile app for Android and iOS
        - 🌐 Offline mode for areas with poor connectivity
        - 🇮🇳 Multi-language support (10+ Indian languages)
        - 👨‍🌾 Direct expert consultation integration
        
        **Our Mission:**
        - 🎯 Make AI-powered crop protection accessible to all farmers
        - 🌱 Promote sustainable agriculture practices
        - 📈 Help maximize yields while minimizing environmental impact
        
        **Stay Updated:**
        - Follow us for launch announcements
        - Beta testing opportunities coming soon
        
        **💚 Note:** We're committed to keeping core features affordable for small-scale farmers!
        """)

else:  # Hindi
    st.markdown("### ❓ अक्सर पूछे जाने वाले प्रश्न")
    
    # FAQ 1: Why is my plant sick?
    with st.expander("🌱 मेरा पौधा बीमार क्यों हो रहा है?", expanded=False):
        st.markdown("""
        पौधे विभिन्न कारणों से बीमार हो सकते हैं:
        
        **1. रोगजनक संक्रमण:**
        - **फफूंद (Fungi)**: सबसे आम कारण (रस्ट, ब्लाइट, मिल्ड्यू)
        - **बैक्टीरिया**: सड़न, पत्ती धब्बे, मुरझाना
        - **वायरस**: विकास रुकना, मोज़ेक पैटर्न
        - **नेमाटोड**: जड़ों पर हमला करने वाले सूक्ष्म कृमि
        
        **2. पर्यावरणीय तनाव:**
        - अत्यधिक तापमान (बहुत गर्म या ठंडा)
        - पानी की कमी (सूखा या जलभराव)
        - खराब मिट्टी की गुणवत्ता या पोषक तत्वों की कमी
        - अपर्याप्त सूर्य का प्रकाश या अत्यधिक छाया
        
        **3. खराब कृषि प्रथाएं:**
        - अनुचित दूरी से हवा का संचार खराब होना
        - दूषित उपकरणों से रोगों का फैलना
        - संक्रमित बीज या पौधों का उपयोग
        - अधिक या कम उर्वरक का उपयोग
        
        **4. मौसम की स्थिति:**
        - उच्च आर्द्रता से फफूंद का विकास
        - अत्यधिक वर्षा से रोगजनकों का फैलाव
        - अचानक तापमान में परिवर्तन
        
        **🔍 प्रारंभिक पहचान महत्वपूर्ण है:** रोगों का शीघ्र पता लगाने के लिए नियमित रूप से CropShield AI का उपयोग करें!
        """)
    
    # FAQ 2: How can I prevent diseases?
    with st.expander("🛡️ मैं रोगों को कैसे रोक सकता हूं?", expanded=False):
        st.markdown("""
        **रोकथाम पौधों की बीमारियों के खिलाफ सबसे अच्छी रक्षा है:**
        
        **रोपण से पहले:**
        - ✅ प्रमाणित, रोग-मुक्त बीजों का उपयोग करें
        - ✅ रोग-प्रतिरोधी फसल किस्मों का चयन करें
        - ✅ उचित मिट्टी की तैयारी और pH संतुलन सुनिश्चित करें
        - ✅ रोग चक्र को तोड़ने के लिए फसल चक्र की योजना बनाएं
        
        **बढ़ते मौसम के दौरान:**
        - ✅ हवा के संचार के लिए उचित पौधों की दूरी बनाए रखें
        - ✅ मिट्टी के स्तर पर पानी दें, पत्तियों को गीला करने से बचें
        - ✅ संक्रमित पौधों के हिस्सों को तुरंत हटाएं और नष्ट करें
        - ✅ खेत को साफ रखें, खरपतवार और मलबा हटाएं
        - ✅ साफ, स्टरलाइज़्ड उपकरणों का उपयोग करें
        
        **मिट्टी प्रबंधन:**
        - ✅ जैविक पदार्थ जोड़ें (खाद, गोबर)
        - ✅ अच्छी निकासी सुनिश्चित करें
        - ✅ नियमित रूप से मिट्टी का परीक्षण करें और पोषक तत्वों को समायोजित करें
        - ✅ हरी खाद का अभ्यास करें
        
        **नियमित निगरानी:**
        - ✅ शुरुआती संकेतों के लिए साप्ताहिक फसलों का निरीक्षण करें
        - ✅ त्वरित रोग का पता लगाने के लिए CropShield AI का उपयोग करें
        - ✅ रोग की घटनाओं का रिकॉर्ड रखें
        - ✅ लक्षण दिखाई देने पर जल्दी से कार्य करें
        
        **जैविक नियंत्रण:**
        - ✅ लाभकारी कीड़ों को प्रोत्साहित करें
        - ✅ उपयुक्त होने पर जैव-कीटनाशकों का उपयोग करें
        - ✅ निवारक रूप से नीम-आधारित उत्पादों को लागू करें
        
        **🌟 याद रखें:** रोकथाम हमेशा इलाज से सस्ती और अधिक प्रभावी होती है!
        """)
    
    # FAQ 3: How to apply treatments safely?
    with st.expander("💊 उपचार सुरक्षित रूप से कैसे लागू करें?", expanded=False):
        st.markdown("""
        **उपचार अनुप्रयोग के लिए सुरक्षा दिशानिर्देश:**
        
        **व्यक्तिगत सुरक्षा (हमेशा पहनें):**
        - 🧤 रसायन-प्रतिरोधी दस्ताने
        - 😷 रेस्पिरेटर मास्क या N95 मास्क
        - 🥽 सुरक्षा चश्मा या फेस शील्ड
        - 👕 लंबी बाजू की शर्ट और लंबी पैंट
        - 👢 जूते (चप्पल नहीं)
        - 🧢 टोपी या सिर ढकना
        
        **अनुप्रयोग से पहले:**
        - 📖 उत्पाद लेबल को ध्यान से पढ़ें
        - ⚖️ सटीक खुराक मापें - कभी अनुमान न लगाएं
        - 🌡️ मौसम की जाँच करें - हवादार या बारिश के दिनों से बचें
        - ⏰ सुबह जल्दी (6-9 AM) या शाम (4-6 PM) छिड़काव की योजना बनाएं
        - 📱 परिवार के सदस्यों और पड़ोसियों को सूचित करें
        
        **अनुप्रयोग के दौरान:**
        - 🚶 स्प्रे संपर्क से बचने के लिए पीछे की ओर चलें
        - 💨 अपनी पीठ पर हवा के साथ स्प्रे करें
        - 🚫 छिड़काव करते समय न खाएं, न पीएं या धूम्रपान न करें
        - 🚸 बच्चों और पालतू जानवरों को दूर रखें
        - 🐝 फूलों वाले पौधों पर छिड़काव करने से बचें (मधुमक्खियों की रक्षा करें)
        
        **अनुप्रयोग के बाद:**
        - 🚿 हाथ और चेहरे को अच्छी तरह से धोएं
        - 👔 दूषित कपड़ों को हटाएं और अलग से धोएं
        - 🗑️ खाली कंटेनरों का ठीक से निपटान करें (कभी पुन: उपयोग न करें)
        - 🚰 जल स्रोतों को दूषित न करें
        - ⏳ कटाई से पहले पूर्व-कटाई अंतराल का पालन करें
        
        **भंडारण:**
        - 🔒 लेबल के साथ मूल कंटेनरों में स्टोर करें
        - 🏠 बंद, सूखे, अच्छी तरह हवादार क्षेत्र में रखें
        - 🚫 भोजन, चारा और रहने के क्षेत्रों से दूर
        - 👶 बच्चों की पहुंच से बाहर
        
        **आपातकाल:**
        - 📞 जहर नियंत्रण नंबर हाथ में रखें
        - 🏥 निकटतम अस्पताल का स्थान जानें
        - 🧴 धोने के लिए साफ पानी उपलब्ध रखें
        - 📋 चिकित्सा संदर्भ के लिए उत्पाद लेबल रखें
        
        **⚠️ महत्वपूर्ण:** यदि आप अनुप्रयोग के दौरान या बाद में बीमार महसूस करते हैं, तुरंत चिकित्सा सहायता लें!
        """)
    
    # FAQ 4: Additional questions
    with st.expander("📱 CropShield AI कितना सटीक है?", expanded=False):
        st.markdown("""
        **CropShield AI सटीकता के बारे में:**
        
        **पहचान सटीकता:**
        - 🎯 सामान्य फसल रोगों के लिए 85-95% सटीकता
        - 📊 विश्वास स्कोर विश्वसनीयता को इंगित करते हैं
        - 🔬 हजारों वास्तविक फसल छवियों पर प्रशिक्षित
        
        **विश्वास स्कोर गाइड:**
        - **90-100%**: बहुत उच्च विश्वास - तत्काल कार्रवाई करें
        - **75-89%**: उच्च विश्वास - बारीकी से निगरानी करें और उपचार तैयार करें
        - **60-74%**: मध्यम विश्वास - निगरानी जारी रखें
        - **60 से नीचे**: कम विश्वास - बेहतर छवि के साथ फिर से स्कैन करने पर विचार करें
        
        **सर्वोत्तम परिणाम युक्तियाँ:**
        - 📸 प्राकृतिक प्रकाश में स्पष्ट, केंद्रित छवियां लें
        - 🔍 प्रभावित क्षेत्रों को करीब से कैप्चर करें
        - 🌅 छाया और चमक से बचें
        - 📐 संभव हो तो कई पत्तियां/भाग शामिल करें
        
        **💡 प्रो टिप:** बेहतर निदान के लिए विभिन्न कोणों से कई फ़ोटो लें!
        """)
    
    with st.expander("💰 CropShield AI की लागत कितनी है?", expanded=False):
        st.markdown("""
        **मूल्य निर्धारण और उपलब्धता:**
        
        **वर्तमान स्थिति:**
        - 🎉 यह एक **प्रोटोटाइप/डेमो संस्करण** है
        - 🆓 परीक्षण और मूल्यांकन के लिए निःशुल्क उपयोग
        - 📱 वेब ब्राउज़र के माध्यम से एक्सेस (कोई ऐप इंस्टॉलेशन की आवश्यकता नहीं)
        
        **भविष्य की योजनाएं:**
        - 📱 Android और iOS के लिए मोबाइल ऐप
        - 🌐 खराब कनेक्टिविटी वाले क्षेत्रों के लिए ऑफ़लाइन मोड
        - 🇮🇳 बहु-भाषा समर्थन (10+ भारतीय भाषाएं)
        - 👨‍🌾 प्रत्यक्ष विशेषज्ञ परामर्श एकीकरण
        
        **हमारा मिशन:**
        - 🎯 सभी किसानों के लिए AI-संचालित फसल सुरक्षा को सुलभ बनाना
        - 🌱 टिकाऊ कृषि प्रथाओं को बढ़ावा देना
        - 📈 पर्यावरणीय प्रभाव को कम करते हुए पैदावार को अधिकतम करने में मदद करना
        
        **💚 नोट:** हम छोटे पैमाने के किसानों के लिए मुख्य सुविधाओं को किफायती रखने के लिए प्रतिबद्ध हैं!
        """)

st.markdown("<br><br>", unsafe_allow_html=True)

# Contact Expert Section
st.markdown("### 📞 Get Expert Help / विशेषज्ञ सहायता प्राप्त करें")

if language == "English":
    st.markdown(
        """
        <div class="metric-card" style="padding: 2rem; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);">
            <p style="color: #2d5016; font-size: 1.1rem; line-height: 1.8; margin: 0;">
                Need personalized advice? Connect with our agricultural experts for consultation 
                on complex disease cases, treatment plans, and farming best practices.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="metric-card" style="padding: 2rem; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);">
            <p style="color: #2d5016; font-size: 1.1rem; line-height: 1.8; margin: 0;">
                व्यक्तिगत सलाह चाहिए? जटिल रोग मामलों, उपचार योजनाओं और कृषि सर्वोत्तम प्रथाओं पर 
                परामर्श के लिए हमारे कृषि विशेषज्ञों से जुड़ें।
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

contact_col1, contact_col2, contact_col3 = st.columns(3)

with contact_col1:
    st.markdown(
        """
        <div class="metric-card" style="padding: 2rem; text-align: center; min-height: 250px;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">💬</div>
            <h4 style="color: #2d5016; margin-bottom: 1rem;">WhatsApp Expert</h4>
            <p style="color: #666; font-size: 0.95rem; margin-bottom: 1.5rem;">
                Chat with agricultural experts on WhatsApp for quick answers
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # WhatsApp button with mock URL
    whatsapp_url = "https://wa.me/919999999999?text=Hello%20CropShield%20AI,%20I%20need%20help%20with%20my%20crops"
    if st.button("📱 Chat on WhatsApp", type="primary", use_container_width=True, key="whatsapp"):
        st.success("Opening WhatsApp... (Demo: +91-9999999999)")
        st.markdown(f"[Click here to open WhatsApp]({whatsapp_url})")

with contact_col2:
    st.markdown(
        """
        <div class="metric-card" style="padding: 2rem; text-align: center; min-height: 250px;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">📞</div>
            <h4 style="color: #2d5016; margin-bottom: 1rem;">Helpline</h4>
            <p style="color: #666; font-size: 0.95rem; margin-bottom: 1.5rem;">
                24/7 toll-free helpline for urgent crop protection issues
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("☎️ Call Helpline", use_container_width=True, key="helpline"):
        st.info("📞 Toll-Free: 1800-XXX-XXXX\n\n⏰ Available 24/7")

with contact_col3:
    st.markdown(
        """
        <div class="metric-card" style="padding: 2rem; text-align: center; min-height: 250px;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">✉️</div>
            <h4 style="color: #2d5016; margin-bottom: 1rem;">Email Support</h4>
            <p style="color: #666; font-size: 0.95rem; margin-bottom: 1.5rem;">
                Send detailed queries and receive expert responses via email
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("📧 Send Email", use_container_width=True, key="email"):
        st.info("📧 Email: support@cropshield.ai\n\n⏱️ Response within 24 hours")

st.markdown("<br><br>", unsafe_allow_html=True)

# Video tutorials section
if language == "English":
    st.markdown("### 🎥 Video Tutorials")
    
    tutorial_col1, tutorial_col2, tutorial_col3 = st.columns(3)
    
    with tutorial_col1:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">▶️</div>
                <h5 style="color: #2d5016;">How to Use CropShield AI</h5>
                <p style="color: #666; font-size: 0.85rem;">5 min tutorial</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Watch Tutorial", use_container_width=True, key="tut1"):
            st.info("🎬 Tutorial video coming soon!")
    
    with tutorial_col2:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">▶️</div>
                <h5 style="color: #2d5016;">Disease Prevention Tips</h5>
                <p style="color: #666; font-size: 0.85rem;">10 min guide</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Watch Guide", use_container_width=True, key="tut2"):
            st.info("🎬 Guide video coming soon!")
    
    with tutorial_col3:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">▶️</div>
                <h5 style="color: #2d5016;">Safe Treatment Application</h5>
                <p style="color: #666; font-size: 0.85rem;">8 min demo</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Watch Demo", use_container_width=True, key="tut3"):
            st.info("🎬 Demo video coming soon!")
else:
    st.markdown("### 🎥 वीडियो ट्यूटोरियल")
    
    tutorial_col1, tutorial_col2, tutorial_col3 = st.columns(3)
    
    with tutorial_col1:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">▶️</div>
                <h5 style="color: #2d5016;">CropShield AI का उपयोग कैसे करें</h5>
                <p style="color: #666; font-size: 0.85rem;">5 मिनट का ट्यूटोरियल</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("ट्यूटोरियल देखें", use_container_width=True, key="tut1_hi"):
            st.info("🎬 ट्यूटोरियल वीडियो जल्द आ रहा है!")
    
    with tutorial_col2:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">▶️</div>
                <h5 style="color: #2d5016;">रोग रोकथाम टिप्स</h5>
                <p style="color: #666; font-size: 0.85rem;">10 मिनट की गाइड</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("गाइड देखें", use_container_width=True, key="tut2_hi"):
            st.info("🎬 गाइड वीडियो जल्द आ रहा है!")
    
    with tutorial_col3:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">▶️</div>
                <h5 style="color: #2d5016;">सुरक्षित उपचार अनुप्रयोग</h5>
                <p style="color: #666; font-size: 0.85rem;">8 मिनट का डेमो</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("डेमो देखें", use_container_width=True, key="tut3_hi"):
            st.info("🎬 डेमो वीडियो जल्द आ रहा है!")

st.markdown("<br><br>", unsafe_allow_html=True)

# Community section
render_contact_section()

st.markdown("<br><br>", unsafe_allow_html=True)

# Additional resources
if language == "English":
    st.markdown("### 📚 Additional Resources")
    
    resource_col1, resource_col2 = st.columns(2)
    
    with resource_col1:
        with st.expander("🔗 Useful Links"):
            st.markdown("""
            - [Ministry of Agriculture & Farmers Welfare](https://agricoop.gov.in/)
            - [Indian Council of Agricultural Research](https://icar.org.in/)
            - [Krishi Vigyan Kendras (KVKs)](https://kvk.icar.gov.in/)
            - [PM Kisan Portal](https://pmkisan.gov.in/)
            - [Soil Health Card](https://soilhealth.dac.gov.in/)
            """)
    
    with resource_col2:
        with st.expander("📱 Government Apps"):
            st.markdown("""
            - **Kisan Suvidha**: Weather, market prices, input dealers
            - **Crop Insurance**: PM Fasal Bima Yojana
            - **mKisan**: SMS-based advisory services
            - **AgriMarket**: Commodity prices and trends
            - **Kisan Rath**: Transportation of agricultural produce
            """)
else:
    st.markdown("### 📚 अतिरिक्त संसाधन")
    
    resource_col1, resource_col2 = st.columns(2)
    
    with resource_col1:
        with st.expander("🔗 उपयोगी लिंक"):
            st.markdown("""
            - [कृषि और किसान कल्याण मंत्रालय](https://agricoop.gov.in/)
            - [भारतीय कृषि अनुसंधान परिषद](https://icar.org.in/)
            - [कृषि विज्ञान केंद्र (KVK)](https://kvk.icar.gov.in/)
            - [पीएम किसान पोर्टल](https://pmkisan.gov.in/)
            - [मृदा स्वास्थ्य कार्ड](https://soilhealth.dac.gov.in/)
            """)
    
    with resource_col2:
        with st.expander("📱 सरकारी ऐप्स"):
            st.markdown("""
            - **किसान सुविधा**: मौसम, बाजार मूल्य, इनपुट डीलर
            - **फसल बीमा**: पीएम फसल बीमा योजना
            - **mKisan**: SMS-आधारित सलाहकार सेवाएं
            - **AgriMarket**: कमोडिटी कीमतें और रुझान
            - **किसान रथ**: कृषि उपज का परिवहन
            """)

st.markdown("<br><br>", unsafe_allow_html=True)

# Navigation
st.markdown("---")
if language == "English":
    st.markdown("### 🔗 Quick Navigation")
else:
    st.markdown("### 🔗 त्वरित नेविगेशन")

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    home_label = "🏠 Home" if language == "English" else "🏠 होम"
    if st.button(home_label, use_container_width=True, key="nav_home"):
        st.switch_page("pages/1_Home.py")

with nav_col2:
    diagnosis_label = "🌿 Diagnosis" if language == "English" else "🌿 निदान"
    if st.button(diagnosis_label, use_container_width=True, key="nav_diagnosis"):
        st.switch_page("pages/2_Diagnosis.py")

with nav_col3:
    rec_label = "💧 Recommendations" if language == "English" else "💧 सिफारिशें"
    if st.button(rec_label, use_container_width=True, key="nav_recommendations"):
        st.switch_page("pages/3_Recommendations.py")

with nav_col4:
    impact_label = "📊 Impact" if language == "English" else "📊 प्रभाव"
    if st.button(impact_label, use_container_width=True, key="nav_impact"):
        st.switch_page("pages/4_Impact_Metrics.py")

st.markdown("<br><br><br>", unsafe_allow_html=True)

# Closing note
if language == "English":
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f1f8e9 0%, #c8e6c9 100%); 
             border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #2d5016; margin-bottom: 1rem;">🌱 Together, we grow healthier crops.</h2>
            <p style="color: #558b2f; font-size: 1.1rem; line-height: 1.8;">
                Thank you for using CropShield AI. We're committed to supporting farmers 
                with cutting-edge technology for sustainable and profitable agriculture.
            </p>
            <p style="color: #7cb342; font-weight: bold; margin-top: 1rem;">
                Happy Farming! 🌾
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f1f8e9 0%, #c8e6c9 100%); 
             border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #2d5016; margin-bottom: 1rem;">🌱 साथ मिलकर, हम स्वस्थ फसलें उगाते हैं।</h2>
            <p style="color: #558b2f; font-size: 1.1rem; line-height: 1.8;">
                CropShield AI का उपयोग करने के लिए धन्यवाद। हम टिकाऊ और लाभदायक कृषि के लिए 
                अत्याधुनिक तकनीक के साथ किसानों का समर्थन करने के लिए प्रतिबद्ध हैं।
            </p>
            <p style="color: #7cb342; font-weight: bold; margin-top: 1rem;">
                खुशहाल खेती! 🌾
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Sidebar
with st.sidebar:
    if language == "English":
        st.markdown("### 💡 Quick Tips")
        st.info("""
        **For Best Results:**
        - Take clear photos in daylight
        - Add weather data when available
        - Follow treatment instructions carefully
        - Monitor crops regularly
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Emergency Contacts")
        st.warning("""
        **Helpline:** 1800-XXX-XXXX
        
        **WhatsApp:** +91-9999999999
        
        **Email:** support@cropshield.ai
        """)
    else:
        st.markdown("### 💡 त्वरित सुझाव")
        st.info("""
        **सर्वोत्तम परिणामों के लिए:**
        - दिन के उजाले में स्पष्ट फ़ोटो लें
        - उपलब्ध होने पर मौसम डेटा जोड़ें
        - उपचार निर्देशों का सावधानीपूर्वक पालन करें
        - नियमित रूप से फसलों की निगरानी करें
        """)
        
        st.markdown("---")
        st.markdown("### 📞 आपातकालीन संपर्क")
        st.warning("""
        **हेल्पलाइन:** 1800-XXX-XXXX
        
        **WhatsApp:** +91-9999999999
        
        **ईमेल:** support@cropshield.ai
        """)

# Footer
st.markdown("---")
if language == "English":
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem; color: #999; font-size: 0.9rem;">
            © 2025 CropShield AI. All rights reserved. | Made with ❤️ for Indian Farmers
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem; color: #999; font-size: 0.9rem;">
            © 2025 CropShield AI. सर्वाधिकार सुरक्षित। | भारतीय किसानों के लिए ❤️ के साथ बनाया गया
        </div>
        """,
        unsafe_allow_html=True
    )
