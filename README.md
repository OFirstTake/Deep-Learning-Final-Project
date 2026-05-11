# CS313: Deep Learning for AI - Final Project
## Topic: Time-series Data and Application to Stock Markets

### 1. Project Introduction
This project focuses on analyzing time-series data and applying Deep Learning techniques to solve real-world problems in financial markets. The primary goal is to build predictive models for stock prices and trading signals, then optimize investment portfolios.

The project explores two major markets (Tech sector focus):
* **International Market:** Nasdaq.
* **Vietnam Market:** HOSE, HNX, and UPCOM.

**Author:** Phạm Nguyên Khang  

---

### 2. Project Tasks Description

The project is structured into 6 main tasks, covering the full pipeline from data engineering to model deployment:

#### Task 1: Nasdaq Stock Price Prediction
* **Objective:** Develop a Deep Learning model to predict the opening price of tech companies on the Nasdaq exchange.

#### Task 2: Vietnam Stock Price Prediction
* **Objective:** Adapt the prediction models to the Vietnamese stock market.

#### Task 3: Trading Signal Identification (Vietnam Market)
* **Objective:** Build a classification model to provide Buy/Sell signals based on historical data.

#### Task 4: Portfolio Optimization
* **Objective:** Design an system to select the best stocks for an investment portfolio.

#### Task 5: System Implementation (Extra Credit)
* **Objective:** Deloy a production-ready AI system.
* **Components:**
    * **Backend:** RESTful API built with FastAPI.
    * **Frontend:** Interactive dashboard using Streamlit for data visualization and inference.
    * **Data Pipeline:** Concepts of automation using Airflow and dbt for robust data workflows.

#### Task 6: Reporting & Open Source
* **Objective:** Documenting the entire research process and sharing the codebase.
* **Deliverables:** A comprehensive technical report (>2000 words) and a structured GitHub repository.
  
---

### 3. Setup and Running Instructions

#### Tasks 1-4: Model Training and Evaluation (Jupyter Notebook)
The core deep learning models, data analysis, and portfolio optimization steps are implemented in the Jupyter Notebook (`240017-project-notebook.ipynb`).

**1. Choose your environment:**
You can run this notebook on **Kaggle**, **Google Colab**, or a **Local Machine**. Using Kaggle or Colab is highly recommended to leverage GPU acceleration for faster training.

**2. Data Preparation:**
* **Kaggle/Colab:** Upload the required datasets (Nasdaq and Vietnam stock data) to your workspace.
* **Local:** Ensure the datasets are extracted into a local directory.

**3. Path Configuration:**
Open the notebook and locate the data loading cell at the beginning. Update the `folder_path` variable to point to your specific dataset directory.
```python
# Example of changing the path
folder_path = '/kaggle/input/dl4ai-dataset' # Change this to your actual path
```

#### Task 5: System Deployment (FastAPI & Streamlit)
This section guides you through running the RESTful API backend and the interactive Streamlit web UI.

**1. Setup Virtual Environment:**
Open your terminal and create a virtual environment to prevent package conflicts.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**2. Install Dependencies:**
Install all necessary Python packages listed in the requirements file.

```bash
pip install -r requirements.txt
```

**3. Host the Backend API (FastAPI):**
Start the FastAPI server to serve the deep learning models. Ensure you are in the directory containing your backend code.

```bash
uvicorn API:app --reload
```

**4. Host the Web UI (Streamlit):**
Open a new terminal window, activate the virtual environment again, and run the frontend application.

```bash
streamlit run Web.py
```

The interactive dashboard will automatically open in your default web browser, typically at `http://localhost:8501`.
