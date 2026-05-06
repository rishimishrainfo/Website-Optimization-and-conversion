from flask import Flask, render_template, request, jsonify
import pickle, json, numpy as np, os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
with open(f"{BASE}/model/lr_model.pkl","rb") as f: lr_model=pickle.load(f)
with open(f"{BASE}/model/dt_model.pkl","rb") as f: dt_model=pickle.load(f)
with open(f"{BASE}/model/encoders.pkl","rb") as f: encoders=pickle.load(f)
with open(f"{BASE}/model/feature_options.json") as f: FEATURE_OPTIONS=json.load(f)

AUDIT_DATA=[
    {"page":"Homepage","type":"Homepage","load":"Fast (<1s)","mobile":"Partial","clarity":"Clear","potential":"High","score":72},
    {"page":"Contact","type":"Contact","load":"Fast (<1s)","mobile":"Partial","clarity":"Clear","potential":"High","score":68},
    {"page":"Partner","type":"Partner","load":"Fast (<1s)","mobile":"Partial","clarity":"Clear","potential":"Medium","score":45},
    {"page":"Sitemap","type":"Other","load":"Fast (<1s)","mobile":"Partial","clarity":"Average","potential":"Medium","score":28},
    {"page":"Verify","type":"Other","load":"Medium (1-3s)","mobile":"Partial","clarity":"Average","potential":"Low","score":12},
]
CHECKLIST=[
    {"category":"CTA Optimization","priority":"High","tasks":["Place CTA above the fold on every key page","Use action-oriented button text (e.g. Apply Now, Get Started)","Ensure CTA is visible on mobile without scrolling","Add CTA in multiple positions — top, middle, and footer","Use contrasting button color against page background"]},
    {"category":"Form Optimization","priority":"High","tasks":["Reduce form fields to 3 or fewer for initial inquiry","Remove optional fields from primary form","Add inline validation to show errors immediately","Auto-fill name/email where possible","Add a privacy assurance line below the submit button"]},
    {"category":"Page Speed","priority":"High","tasks":["Compress all images to WebP format","Enable browser caching for static assets","Minify CSS and JavaScript files","Lazy-load below-the-fold images","Target load time under 1 second for all pages"]},
    {"category":"Mobile Friendliness","priority":"Medium","tasks":["Ensure all buttons are at least 44x44px touch targets","Test layout on 375px (iPhone SE) and 390px (iPhone 14)","Use fluid typography to scale text","Avoid horizontal scrolling on any page","Make navigation accessible via hamburger on small screens"]},
    {"category":"Content Clarity","priority":"Medium","tasks":["Write a clear H1 on every page (missing on 3 pages)","Add a one-line value proposition at the top of homepage","Break content into scannable sections with subheadings","Remove jargon and use plain language","Add social proof (testimonials, partner logos, numbers)"]},
    {"category":"SEO & Metadata","priority":"Low","tasks":["Ensure every page has a unique meta description","Add alt text to all images","Use structured data markup for organization info","Add XML sitemap and submit to Google Search Console","Fix missing H1 on Sitemap, Partner, and Verify pages"]},
]
WIREFRAMES=[
    {"page":"Homepage","sections":["Hero: Bold headline + subtext + Apply Now CTA (above fold)","Social proof bar: partner college logos / intern count","3-column feature grid: What you learn, Duration, Certificate","Testimonial carousel: intern quotes with photos","FAQ accordion","Footer: contact + links + social media"],"tip":"Move the CTA into the hero section. Currently it appears too far down the page."},
    {"page":"Contact","sections":["Short 3-field form: Name, Email, Message (above fold)","Phone + email clickable links in sidebar","Embedded Google Map","Office hours block","Response time assurance: We reply within 24 hours"],"tip":"Reduce form from 4-6 fields to 3. This alone can increase submissions by 30-50%."},
    {"page":"Service / Internship","sections":["Program overview: Duration, Mode, Certificate badge","Step-by-step: Apply → Onboard → Work → Certify","Tech stack icons / skills learned","Apply Now button (sticky on scroll)","FAQ specific to internship queries"],"tip":"Add a sticky Apply Now button that follows the user as they scroll down."},
    {"page":"Partner","sections":["Hero: Partner with us headline + CTA","Benefits grid: Brand visibility, Student access, Easy setup","Partner logos — make them larger and more prominent","Request a Proposal form: 3 fields only","Testimonial from a current partner"],"tip":"Add an H1 tag. Currently missing — hurts both SEO and screen reader accessibility."},
]

@app.route("/")
def dashboard(): return render_template("dashboard.html", audit=AUDIT_DATA)

@app.route("/predict")
def predict_page(): return render_template("predict.html", options=FEATURE_OPTIONS)

@app.route("/checklist")
def checklist_page(): return render_template("checklist.html", checklist=CHECKLIST, wireframes=WIREFRAMES)

@app.route("/api/predict", methods=["POST"])
def predict():
    data=request.json
    try:
        X=np.array([[encoders["page_type"].transform([data["page_type"]])[0],int(data["cta_presence"]),encoders["cta_position"].transform([data["cta_position"]])[0],encoders["form_length"].transform([data["form_length"]])[0],encoders["load_speed"].transform([data["load_speed"]])[0],encoders["mobile_friendly"].transform([data["mobile_friendly"]])[0],encoders["content_length"].transform([data["content_length"]])[0],encoders["content_clarity"].transform([data["content_clarity"]])[0]]])
        model=dt_model if data.get("model","dt")=="dt" else lr_model
        pred=model.predict(X)[0]; proba=model.predict_proba(X)[0]
        label="High Conversion" if pred==1 else "Low Conversion"
        conf=round(float(max(proba))*100,1)
        recs=[]
        if not int(data["cta_presence"]): recs.append("Add a clear Call-To-Action button to the page")
        if data["cta_position"] not in ["Multiple","Middle"]: recs.append("Place CTA in multiple/center positions for visibility")
        if data["load_speed"]!="Fast (<1s)": recs.append("Optimize page load speed to under 1 second")
        if data["mobile_friendly"]!="Yes": recs.append("Improve mobile responsiveness")
        if data["content_clarity"]!="Clear": recs.append("Rewrite content for better clarity")
        if data["form_length"]=="Long (7+)": recs.append("Shorten the form to max 3 fields to reduce friction")
        if not recs: recs.append("Page is well-optimised! Keep monitoring analytics.")
        return jsonify({"label":label,"confidence":conf,"recommendations":recs})
    except Exception as e: return jsonify({"error":str(e)}),400

if __name__=="__main__": app.run(debug=True,port=5000)
