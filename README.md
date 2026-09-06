E-Commerce Customer Segmentation and Dashboard
An end-to-end data analytics and machine learning project that processes e-commerce data, clusters customers using RFM analysis and K-Means, and displays results in an interactive Streamlit dashboard.

Tech Stack
  Data Processing: pandas, numpy
  Machine Learning: scikit-learn (KMeans, StandardScaler, silhouette_score)
  Model Persistence: joblib
  Visualization: matplotlib, plotly
  Dashboard: streamlit

Key Features
  Automated Data Preprocessing: Cleans bad transaction data so leadership makes business decisions based on accurate figures.
  RFM Feature Extraction: Converts sales history into clear customer behavior scores so marketing teams can target specific buying habits.
  Machine Learning Clustering: Automatically groups customers into categories like Champions or At-Risk so companies can run targeted ad campaigns.
  Saved ML Models: Saves trained algorithms to disk so the system can classify new customers instantly without extra compute costs.
  Interactive Dashboard: Gives non-technical managers a visual tool to track sales metrics and check customer segments without writing code.

Installation and Running
Clone repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies:
pip install -r requirements.txt

Add dataset:
Place raw dataset files inside data/raw/.

Run Streamlit app:
streamlit run app/app.py

link: [https://e-commerce-customer-analytics-segmentation-jlg4wkzzezwd6kkfpjv.streamlit.app/?fbclid=IwY2xjawUE8qhwZG9mBWV4dG4DYWVtAjEwAGJyaWQRMWJlU2dSVFpPOXE2cWJNV0ZzcnRjBmFwcF9pZBAyMjIwMzkxNzg4MjAwODkyAAEe3EYx4geuqhSctNz1N1_TbWdTrGSAL5xST619bzkQfDS38NiMvgO2KwpmDPY_aem_fWRQ9-cy7UCSBefLXy5UBw]
<img width="1882" height="858" alt="image" src="https://github.com/user-attachments/assets/dfe7383c-e3e1-4dd9-bec4-deb18c0e6fad" />
<img width="1902" height="840" alt="image" src="https://github.com/user-attachments/assets/634ac2fc-2c56-4879-9968-ffd8f1c25bcb" />
<img width="1895" height="843" alt="image" src="https://github.com/user-attachments/assets/52e3db3b-8576-4405-a7b1-d529cca468a1" />


