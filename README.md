# SafePay — Project README

## Project summary
**Primary project:** HepaSense — AI-Based Hepatitis B Prediction and Liver Analysis System (original project idea).  
**Submitted project (this repository):** SafePay — AI-Based Credit Card Fraud Detection system (implemented using Streamlit and several ML models).

This repository contains the SafePay implementation which I submitted because I encountered a major unresolved issue while working on the original Hepatitis B (HepaSense) project. SafePay is a complete, working project that demonstrates data preprocessing, modeling, evaluation, and a Streamlit interface for predicting fraud risk.

---

## Why SafePay was submitted instead of HepaSense
- The original project (HepaSense) focused on Hepatitis B prediction using medical indicators and liver imaging.
- During development I faced a significant technical/data issue in the HepaSense pipeline that prevented reliable results (details below).
- Due to time constraints and the need to deliver a complete, testable submission, I pivoted and completed SafePay which is fully functional and documented.

**Note:** Work on HepaSense continues separately (we sumbmited the cv and ml of the hepatitis b) ;
**link of repo of hepasense( https://github.com/omarsalama4/DEPI-Virus-B )**

---

## Notes about the HepaSense issue (brief)
- Problem encountered: very extreme overlap ploblem and imbalance classes ( we can't solve it using the SMOTE , CLASS WEIGHT , PCA) and very low recall and percision.
- Impact: Could not reach reliable, reproducible performance for medical predictions within the project timeline.
- Action taken: Documented the problem, saved experimental artifacts, and postponed further HepaSense work to a follow-up iteration.
